"""
OaK Agent - Main agent class for the Options and Knowledge framework.

This module provides the OaKAgent class, which integrates all OaK components
to create an agent compatible with the Crafter environment's Gymnasium API.

The agent uses:
    - Feature extraction from privileged info dict
    - Technology tree for subtask ordering
    - GVFs for reward-respecting value estimation
    - Options for temporally extended actions
    - Interrupt manager for survival priorities

Usage:
    from oak import OaKAgent
    import crafter

    env = crafter.Env()
    agent = OaKAgent()

    obs, info = env.reset()
    agent.reset()

    while not done:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, info, terminated or truncated)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from oak.constants import (
    ACHIEVEMENTS,
    NUM_ACTIONS,
    TOTAL_FEATURE_DIM,
)
from oak.features import FeatureExtractor
from oak.gvf import GeneralValueFunction, GVFFactory
from oak.interrupts import InterruptManager, InterruptRequest
from oak.options import (
    Option,
    OptionManager,
    OptionResult,
    RandomOption,
    create_options_for_subtasks,
)
from oak.subtasks import Subtask, get_all_subtasks
from oak.tech_tree import TechnologyTree


@dataclass
class AgentState:
    """
    Internal state of the OaK agent.

    Attributes:
        step_count: Total steps taken in current episode.
        completed_achievements: Set of completed achievement names.
        current_features: Most recent feature vector.
        previous_features: Previous feature vector.
        previous_info: Previous info dictionary.
    """

    step_count: int = 0
    completed_achievements: Set[str] = None
    current_features: Optional[np.ndarray] = None
    previous_features: Optional[np.ndarray] = None
    previous_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.completed_achievements is None:
            self.completed_achievements = set()


class OaKAgent:
    """
    OaK (Options and Knowledge) Agent for Crafter.

    Implements the OaK architecture with reward-respecting subtasks
    of feature attainment, using the Crafter environment as testbed.

    The agent balances:
    1. Technology tree progression (collecting resources, crafting tools)
    2. Survival maintenance (food, water, rest, combat)

    Using priority-based interrupt system and temporally extended options.

    Attributes:
        feature_extractor: Converts observations to feature vectors.
        tech_tree: Manages achievement dependencies.
        option_manager: Handles option selection and execution.
        interrupt_manager: Manages survival interrupts.
        gvfs: Dictionary of GVFs for each achievement.
        state: Current agent state.
    """

    def __init__(
        self,
        use_privileged_features: bool = True,
        option_type: str = "random",
        enable_interrupts: bool = True,
        learning_rate: float = 0.01,
        discount: float = 0.99,
    ) -> None:
        """
        Initialize the OaK agent.

        Args:
            use_privileged_features: Use info dict features (True) or pixels (False).
            option_type: Type of options ("random", "heuristic", "learned").
            enable_interrupts: Enable survival interrupt system.
            learning_rate: Learning rate for GVFs.
            discount: Discount factor for value functions.
        """
        # Feature extraction
        if use_privileged_features:
            self.feature_extractor = FeatureExtractor()
        else:
            raise NotImplementedError(
                "Pixel-based features not yet supported. "
                "Use use_privileged_features=True."
            )

        # Technology tree
        self.tech_tree = TechnologyTree()

        # Create subtasks and GVFs
        self.subtasks = get_all_subtasks()
        self.gvf_factory = GVFFactory(
            feature_dim=TOTAL_FEATURE_DIM,
            learning_rate=learning_rate,
            discount=discount,
        )
        self.gvfs = self.gvf_factory.create_all(
            self.feature_extractor.achievement_indices
        )

        # Create options
        options = create_options_for_subtasks(
            self.subtasks,
            self.gvfs,
            option_type=option_type,
        )
        self.option_manager = OptionManager()
        self.option_manager.register_all(list(options.values()))

        # Interrupt manager
        self.interrupt_manager = InterruptManager()
        self.enable_interrupts = enable_interrupts

        # Agent state
        self.state = AgentState()

        # Configuration
        self.option_type = option_type
        self.learning_rate = learning_rate
        self.discount = discount

    def reset(self) -> None:
        """
        Reset the agent for a new episode.

        Should be called when environment is reset.
        """
        self.state = AgentState()
        self.option_manager.reset()
        self.interrupt_manager.reset()

        # Reset GVF traces
        for gvf in self.gvfs.values():
            gvf.reset_traces()

    def act(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
    ) -> int:
        """
        Select an action given current observation and info.

        Implements the priority-based action selection:
        1. Check for survival interrupts (highest priority)
        2. Continue active option if present
        3. Select new option from available subtasks

        Args:
            obs: Current observation (64, 64, 3) RGB image.
            info: Info dictionary from environment.

        Returns:
            Action index (0-16).
        """
        # Extract features
        features = self.feature_extractor.extract(obs, info)
        self.state.current_features = features

        # Update completed achievements
        self._update_completed_achievements(info)

        # Step 1: Check for survival interrupts
        if self.enable_interrupts:
            interrupt = self._check_interrupts(info)
            if interrupt is not None:
                return self._handle_interrupt(interrupt, obs, info)

        # Step 2: Continue active option if present
        if self.option_manager.has_active_option():
            result = self.option_manager.step(obs, info)
            if not result.terminated:
                return result.action
            # Option terminated, fall through to select new one

        # Step 3: Select and initiate new option
        action = self._select_and_initiate_option(obs, info)

        # Update state for next step
        self.state.previous_features = features
        self.state.previous_info = info.copy()
        self.state.step_count += 1

        return action

    def _check_interrupts(
        self,
        info: Dict[str, Any],
    ) -> Optional[InterruptRequest]:
        """
        Check for survival interrupts.

        Args:
            info: Current info dictionary.

        Returns:
            InterruptRequest if interrupt needed, None otherwise.
        """
        # Check vital-based interrupts
        interrupt = self.interrupt_manager.check_interrupt(info)

        if interrupt is not None:
            # Check if we should preempt current option
            if self.option_manager.has_active_option():
                current = self.option_manager.active_option.name
                if not self.interrupt_manager.should_preempt(current, interrupt):
                    return None

            return interrupt

        # Check combat interrupt (health decreasing)
        combat_interrupt = self.interrupt_manager.check_combat_interrupt(
            info, self.state.previous_info
        )

        return combat_interrupt

    def _handle_interrupt(
        self,
        interrupt: InterruptRequest,
        obs: np.ndarray,
        info: Dict[str, Any],
    ) -> int:
        """
        Handle a survival interrupt by activating appropriate option.

        Args:
            interrupt: The interrupt request.
            obs: Current observation.
            info: Current info dictionary.

        Returns:
            Action from the interrupt option.
        """
        option = self.option_manager.get_option(interrupt.survival_option)

        if option is None:
            # Fallback to random action if option not found
            return np.random.randint(0, NUM_ACTIONS)

        # Preempt current option and activate survival option
        self.option_manager.set_active(option, preempt=True)
        result = self.option_manager.step(obs, info)

        return result.action

    def _select_and_initiate_option(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
    ) -> int:
        """
        Select and initiate a new option.

        Args:
            obs: Current observation.
            info: Current info dictionary.

        Returns:
            First action from the new option.
        """
        # Get available subtasks based on tech tree
        available_names = self.tech_tree.get_available_subtasks(
            self.state.completed_achievements
        )

        # Get corresponding options
        available_options = [
            self.option_manager.get_option(name)
            for name in available_names
            if self.option_manager.get_option(name) is not None
        ]

        # Select best option
        option = self.option_manager.select_next_option(info, available_options)

        if option is None:
            # No options available - take random action
            return np.random.randint(0, NUM_ACTIONS)

        # Initiate option and get first action
        self.option_manager.set_active(option)
        result = self.option_manager.step(obs, info)

        return result.action

    def _update_completed_achievements(self, info: Dict[str, Any]) -> None:
        """Update set of completed achievements from info dict."""
        achievements = info.get("achievements", {})
        for name, count in achievements.items():
            if count > 0:
                self.state.completed_achievements.add(name)

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        info: Dict[str, Any],
        done: bool,
    ) -> None:
        """
        Update agent after taking an action.

        Used for learning from experience.

        Args:
            obs: Observation after action.
            action: Action taken.
            reward: Reward received.
            info: Info dictionary after action.
            done: Whether episode ended.
        """
        # Extract features for next state
        next_features = self.feature_extractor.extract(obs, info)

        if self.state.current_features is not None:
            # Update GVFs with experience
            self._update_gvfs(
                self.state.current_features,
                action,
                reward,
                next_features,
                done,
            )

    def _update_gvfs(
        self,
        features: np.ndarray,
        action: int,
        reward: float,
        next_features: np.ndarray,
        done: bool,
    ) -> None:
        """
        Update all GVFs with the latest experience.

        Args:
            features: Current state features.
            action: Action taken.
            reward: Reward received.
            next_features: Next state features.
            done: Whether episode ended.
        """
        for name, gvf in self.gvfs.items():
            try:
                gvf.update(reward, features, next_features, done)
            except NotImplementedError:
                # GVF update not implemented - skip
                pass

    def select_optimistic_weights(
        self,
        method: str = "uniform",
        value: float = 10.0,
    ) -> None:
        """
        Set optimistic bonus weights for all GVFs.

        Args:
            method: Method for selecting weights ("uniform", "adaptive").
            value: Weight value for uniform method.

        Raises:
            NotImplementedError: Adaptive weight selection not implemented.
        """
        if method == "uniform":
            for gvf in self.gvfs.values():
                gvf.select_optimistic_weight(value)
        else:
            raise NotImplementedError(
                f"Weight selection method '{method}' not implemented. "
                "See STRATEGY.md 'Unresolved Issues' section 1."
            )

    def rank_features(
        self,
        method: str = "value_contribution",
    ) -> List[Tuple[str, float]]:
        """
        Rank features by their contribution to value function.

        Args:
            method: Ranking method to use.

        Returns:
            List of (feature_name, score) tuples, sorted by score.

        Raises:
            NotImplementedError: Feature ranking not implemented.
        """
        raise NotImplementedError(
            "Feature ranking algorithm not implemented. "
            "See STRATEGY.md 'Unresolved Issues' section 2."
        )

    def off_policy_correction(
        self,
        behavioral_policy: Any,
        target_policy: Any,
    ) -> float:
        """
        Compute importance sampling ratio for off-policy correction.

        Args:
            behavioral_policy: The policy that generated the data.
            target_policy: The policy being learned.

        Returns:
            Importance sampling ratio.

        Raises:
            NotImplementedError: Off-policy correction not implemented.
        """
        raise NotImplementedError(
            "Off-policy correction not implemented. "
            "See STRATEGY.md 'Unresolved Issues' section 5."
        )

    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics for logging.

        Returns:
            Dictionary of agent statistics.
        """
        return {
            "step_count": self.state.step_count,
            "completed_achievements": len(self.state.completed_achievements),
            "active_option": (
                self.option_manager.active_option.name
                if self.option_manager.has_active_option()
                else None
            ),
        }
