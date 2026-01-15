"""
Options and Option Management for the OaK framework.

This module implements temporally extended options as defined in the
options framework. Each option consists of:
    - Initiation set: States where the option can start
    - Policy: Behavior while option is active
    - Termination condition: When the option completes (β(s) = 1)

Options in OaK are derived from reward-respecting GVFs and execute
until their termination condition is satisfied.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from oak.constants import ACTIONS, ACTION_TO_INDEX, NUM_ACTIONS
from oak.gvf import GeneralValueFunction
from oak.subtasks import Subtask, SubtaskDefinition, SUBTASK_DEFINITIONS


class OptionState(Enum):
    """State of an option's execution."""

    INACTIVE = auto()  # Option not currently executing
    ACTIVE = auto()  # Option is executing
    TERMINATED = auto()  # Option just completed


@dataclass
class OptionResult:
    """
    Result of an option step.

    Attributes:
        action: The primitive action to execute.
        terminated: Whether the option terminated this step.
        info: Additional information about the step.
    """

    action: int
    terminated: bool
    info: Dict[str, Any]


class Option(ABC):
    """
    Abstract base class for temporally extended options.

    An option encapsulates a policy that executes over multiple timesteps
    until a termination condition is met.

    Attributes:
        name: Identifier for this option.
        subtask: Associated subtask definition.
        gvf: General Value Function for this option.
        state: Current execution state.
    """

    def __init__(
        self,
        name: str,
        subtask: Subtask,
        gvf: Optional[GeneralValueFunction] = None,
    ) -> None:
        """
        Initialize an option.

        Args:
            name: Name identifier (typically achievement name).
            subtask: Associated Subtask instance.
            gvf: Optional GVF for value estimation.
        """
        self.name = name
        self.subtask = subtask
        self.gvf = gvf
        self.state = OptionState.INACTIVE
        self._previous_info: Optional[Dict[str, Any]] = None

    @abstractmethod
    def can_initiate(self, info: Dict[str, Any]) -> bool:
        """
        Check if this option can be initiated in the current state.

        Args:
            info: Current info dictionary from environment.

        Returns:
            True if option can start (state is in initiation set).
        """
        pass

    @abstractmethod
    def select_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        """
        Select a primitive action according to the option's policy.

        Args:
            obs: Current observation.
            info: Current info dictionary.

        Returns:
            Action index (0-16).
        """
        pass

    def initiate(self, info: Dict[str, Any]) -> None:
        """
        Start executing this option.

        Args:
            info: Current info dictionary.
        """
        self.state = OptionState.ACTIVE
        self._previous_info = info.copy() if info else None

    def step(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
    ) -> OptionResult:
        """
        Execute one step of the option.

        Args:
            obs: Current observation.
            info: Current info dictionary.

        Returns:
            OptionResult containing action and termination status.
        """
        if self.state != OptionState.ACTIVE:
            raise RuntimeError(f"Option {self.name} is not active")

        # Check termination condition β(s)
        terminated = self.subtask.check_termination(info, self._previous_info)

        if terminated:
            self.state = OptionState.TERMINATED
            action = 0  # noop on termination
        else:
            action = self.select_action(obs, info)

        self._previous_info = info.copy() if info else None

        return OptionResult(
            action=action,
            terminated=terminated,
            info={"option": self.name},
        )

    def terminate(self) -> None:
        """Force termination of the option."""
        self.state = OptionState.TERMINATED

    def reset(self) -> None:
        """Reset option to inactive state."""
        self.state = OptionState.INACTIVE
        self._previous_info = None

    def is_active(self) -> bool:
        """Check if option is currently executing."""
        return self.state == OptionState.ACTIVE

    def get_priority(self) -> int:
        """Get priority level from subtask definition."""
        return self.subtask.get_priority()


class RandomOption(Option):
    """
    Option with random action selection (placeholder policy).

    Used as a baseline before learned policies are available.
    """

    def can_initiate(self, info: Dict[str, Any]) -> bool:
        """Random option can always initiate if subtask not completed."""
        return not self.subtask.is_completed(info)

    def select_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        """Select a random valid action."""
        return np.random.randint(0, NUM_ACTIONS)


class HeuristicOption(Option):
    """
    Option with heuristic action selection.

    Provides hand-coded policies for specific subtasks.
    This is a placeholder until learned policies are available.
    """

    def __init__(
        self,
        name: str,
        subtask: Subtask,
        gvf: Optional[GeneralValueFunction] = None,
        action_sequence: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize heuristic option.

        Args:
            name: Option name.
            subtask: Associated subtask.
            gvf: Optional GVF.
            action_sequence: Optional sequence of actions to execute.
        """
        super().__init__(name, subtask, gvf)
        self.action_sequence = action_sequence or []
        self._step_count = 0

    def can_initiate(self, info: Dict[str, Any]) -> bool:
        """Check if prerequisites are met and subtask not completed."""
        # For now, just check if not already completed
        return not self.subtask.is_completed(info)

    def select_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        """Select action from heuristic sequence or random fallback."""
        if self.action_sequence and self._step_count < len(self.action_sequence):
            action = self.action_sequence[self._step_count]
            self._step_count += 1
            return action

        # Random exploration when sequence exhausted
        return np.random.randint(0, NUM_ACTIONS)

    def reset(self) -> None:
        """Reset option and step counter."""
        super().reset()
        self._step_count = 0


class LearnedOption(Option):
    """
    Option with learned policy from GVF.

    Uses the GVF's value function to select actions that
    maximize expected return while respecting rewards.
    """

    def can_initiate(self, info: Dict[str, Any]) -> bool:
        """Check if option can start based on prerequisites."""
        return not self.subtask.is_completed(info)

    def select_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        """
        Select action using learned policy.

        Raises:
            NotImplementedError: Learned policy not yet implemented.
        """
        raise NotImplementedError(
            "Learned option policy not implemented. "
            "Requires trained GVF and action-value function. "
            "Use RandomOption or HeuristicOption instead."
        )


class OptionManager:
    """
    Manages option selection and execution.

    Handles:
    - Tracking active option
    - Option switching and preemption
    - Coordinating option execution

    Attributes:
        options: Dictionary of available options.
        active_option: Currently executing option (if any).
        completed_options: Set of completed option names.
    """

    def __init__(self) -> None:
        """Initialize the option manager."""
        self.options: Dict[str, Option] = {}
        self.active_option: Optional[Option] = None
        self.completed_options: Set[str] = set()
        self._step_count = 0

    def register_option(self, option: Option) -> None:
        """
        Register an option with the manager.

        Args:
            option: Option instance to register.
        """
        self.options[option.name] = option

    def register_all(self, options: List[Option]) -> None:
        """
        Register multiple options.

        Args:
            options: List of Option instances.
        """
        for option in options:
            self.register_option(option)

    def get_option(self, name: str) -> Optional[Option]:
        """Get an option by name."""
        return self.options.get(name)

    def has_active_option(self) -> bool:
        """Check if there is an active option."""
        return self.active_option is not None and self.active_option.is_active()

    def set_active(
        self,
        option: Option,
        preempt: bool = False,
    ) -> None:
        """
        Set the active option.

        Args:
            option: Option to activate.
            preempt: If True, terminate current option before switching.
        """
        if self.active_option is not None and self.active_option.is_active():
            if preempt:
                self.active_option.terminate()
            else:
                return  # Don't switch if active and not preempting

        option.initiate({})  # Will be updated with real info on step
        self.active_option = option

    def clear_active(self) -> None:
        """Clear the active option."""
        if self.active_option is not None:
            self.active_option.reset()
        self.active_option = None

    def step(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
    ) -> OptionResult:
        """
        Execute one step of the active option.

        Args:
            obs: Current observation.
            info: Current info dictionary.

        Returns:
            OptionResult from active option.

        Raises:
            RuntimeError: If no active option.
        """
        if self.active_option is None:
            raise RuntimeError("No active option to step")

        result = self.active_option.step(obs, info)
        self._step_count += 1

        if result.terminated:
            self.completed_options.add(self.active_option.name)
            self.clear_active()

        return result

    def get_available_options(
        self,
        info: Dict[str, Any],
        exclude_completed: bool = True,
    ) -> List[Option]:
        """
        Get options that can be initiated in current state.

        Args:
            info: Current info dictionary.
            exclude_completed: Whether to exclude completed options.

        Returns:
            List of available options, sorted by priority.
        """
        available = []
        for name, option in self.options.items():
            if exclude_completed and name in self.completed_options:
                continue
            if option.can_initiate(info):
                available.append(option)

        # Sort by priority (descending)
        return sorted(available, key=lambda o: o.get_priority(), reverse=True)

    def select_next_option(
        self,
        info: Dict[str, Any],
        available: Optional[List[Option]] = None,
    ) -> Optional[Option]:
        """
        Select the next option to execute.

        Args:
            info: Current info dictionary.
            available: Optional pre-filtered list of available options.

        Returns:
            Selected option or None if no options available.

        Note:
            Current implementation uses priority-based selection.
            Feature-based ranking is not yet implemented.
        """
        if available is None:
            available = self.get_available_options(info)

        if not available:
            return None

        # Select highest priority option
        # TODO: Implement feature ranking (see STRATEGY.md Unresolved Issues #2)
        return available[0]

    def compose_options(
        self,
        low_level_options: List[Option],
        target_achievement: str,
    ) -> Option:
        """
        Compose low-level options into a higher-level option.

        Args:
            low_level_options: Constituent options.
            target_achievement: Target achievement for composed option.

        Returns:
            Composed option.

        Raises:
            NotImplementedError: Option composition not implemented.
        """
        raise NotImplementedError(
            "Option composition not implemented. "
            "Requires STOMP hierarchy specification. "
            "See STRATEGY.md 'Unresolved Issues' section 3."
        )

    def reset(self) -> None:
        """Reset the option manager for new episode."""
        self.clear_active()
        self.completed_options.clear()
        self._step_count = 0
        for option in self.options.values():
            option.reset()


def create_options_for_subtasks(
    subtasks: Dict[str, Subtask],
    gvfs: Optional[Dict[str, GeneralValueFunction]] = None,
    option_type: str = "random",
) -> Dict[str, Option]:
    """
    Create options for all subtasks.

    Args:
        subtasks: Dictionary of Subtask instances.
        gvfs: Optional dictionary of GVF instances.
        option_type: Type of option to create ("random", "heuristic", "learned").

    Returns:
        Dictionary mapping subtask names to Option instances.
    """
    options = {}

    for name, subtask in subtasks.items():
        gvf = gvfs.get(name) if gvfs else None

        if option_type == "random":
            option = RandomOption(name, subtask, gvf)
        elif option_type == "heuristic":
            option = HeuristicOption(name, subtask, gvf)
        elif option_type == "learned":
            option = LearnedOption(name, subtask, gvf)
        else:
            raise ValueError(f"Unknown option type: {option_type}")

        options[name] = option

    return options
