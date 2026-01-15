"""
General Value Functions (GVFs) for the OaK framework.

This module implements the GVF formalism as described in STRATEGY.md,
with the four key components: cumulant, stopping function, stopping value,
and policy.

GVFs enable reward-respecting subtasks of feature attainment, where the
agent pursues specific state features while still maximizing the primary
reward signal.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from oak.constants import TOTAL_FEATURE_DIM


@dataclass
class GVFComponents:
    """
    Container for the four components of a GVF.

    Attributes:
        cumulant_fn: Function that returns C_t given reward.
        stopping_fn: Function β(s) that returns termination probability.
        stopping_value_fn: Function z(s) for additional value at termination.
        policy: The behavior policy π for this GVF.
    """

    cumulant_fn: Callable[[float], float]
    stopping_fn: Callable[[np.ndarray, Optional[np.ndarray]], float]
    stopping_value_fn: Callable[[np.ndarray, np.ndarray], float]
    policy: Optional[Callable[[np.ndarray], int]] = None


class GeneralValueFunction:
    """
    General Value Function for reward-respecting feature attainment.

    A GVF predicts the expected sum of a cumulant signal plus a stopping
    value, discounted by continuing probability. For reward-respecting
    subtasks, the cumulant equals the environment reward (C_t = R_t).

    The stopping value z_i(s) is defined as:
        z_i(s) = w^T x(s) - w_i x_i(s) + w̄_i x_i(s)

    This creates an optimistic bonus for terminating in states where
    the target feature x_i is high.

    Attributes:
        target_feature_index: Index of the target feature in x(s).
        feature_dim: Dimension of the feature vector.
        weights: Current weight vector w for value function.
        optimistic_weight: Bonus weight w̄_i for target feature.
        traces: Eligibility traces for learning.
    """

    def __init__(
        self,
        target_feature_index: int,
        feature_dim: int = TOTAL_FEATURE_DIM,
        learning_rate: float = 0.01,
        discount: float = 0.99,
        trace_decay: float = 0.9,
    ) -> None:
        """
        Initialize a GVF for feature attainment.

        Args:
            target_feature_index: Index i of target feature x_i in x(s).
            feature_dim: Dimension of feature vector (default 40).
            learning_rate: Learning rate α for weight updates.
            discount: Discount factor γ for future returns.
            trace_decay: Eligibility trace decay λ.
        """
        self.target_feature_index = target_feature_index
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.discount = discount
        self.trace_decay = trace_decay

        # Initialize weights
        self.weights = np.zeros(feature_dim, dtype=np.float32)
        self.traces = np.zeros(feature_dim, dtype=np.float32)

        # Optimistic bonus weight (to be set via select_optimistic_weight)
        self.optimistic_weight: Optional[float] = None

    def cumulant(self, reward: float) -> float:
        """
        Compute the cumulant C_t for this GVF.

        For reward-respecting subtasks, C_t = R_t (environment reward).

        Args:
            reward: Environment reward at time t.

        Returns:
            Cumulant value (equals reward for reward-respecting GVFs).
        """
        return reward

    def stopping_condition(
        self,
        current_features: np.ndarray,
        previous_features: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the stopping probability β(s).

        Returns 1.0 when the target feature has been attained (increased),
        0.0 otherwise.

        Args:
            current_features: Current feature vector x(s).
            previous_features: Previous feature vector x(s') for change detection.

        Returns:
            Stopping probability β(s) ∈ {0.0, 1.0}.
        """
        current_value = current_features[self.target_feature_index]

        if previous_features is None:
            # No previous state - check if feature is already high
            return 1.0 if current_value > 0.5 else 0.0

        previous_value = previous_features[self.target_feature_index]

        # Terminate when feature increases (attainment detected)
        return 1.0 if current_value > previous_value else 0.0

    def stopping_value(
        self,
        features: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the stopping value z_i(s).

        z_i(s) = w^T x(s) - w_i x_i(s) + w̄_i x_i(s)
               = w^T x(s) + (w̄_i - w_i) x_i(s)

        This replaces the standard weight of the target feature with
        an optimistic bonus weight, creating incentive to terminate
        in states where the target feature is present.

        Args:
            features: Feature vector x(s).
            weights: Optional weight vector (uses self.weights if None).

        Returns:
            Stopping value z_i(s).

        Raises:
            NotImplementedError: If optimistic weight not set.
        """
        if self.optimistic_weight is None:
            raise NotImplementedError(
                "Optimistic bonus weight w̄_i not set. "
                "Call select_optimistic_weight() first. "
                "See STRATEGY.md 'Unresolved Issues' section 1."
            )

        w = weights if weights is not None else self.weights
        i = self.target_feature_index

        # z_i(s) = w^T x(s) + (w̄_i - w_i) * x_i(s)
        base_value = np.dot(w, features)
        bonus = (self.optimistic_weight - w[i]) * features[i]

        return base_value + bonus

    def value(self, features: np.ndarray) -> float:
        """
        Compute the current value estimate V(s) = w^T x(s).

        Args:
            features: Feature vector x(s).

        Returns:
            Value estimate.
        """
        return float(np.dot(self.weights, features))

    def select_optimistic_weight(self, weight: float) -> None:
        """
        Set the optimistic bonus weight w̄_i for the target feature.

        The choice of this weight affects the trade-off between
        exploration (high bonus) and exploitation (low bonus).

        Args:
            weight: The optimistic weight value (should be > w_i).

        Note:
            The optimal selection of w̄_i is domain-dependent and
            currently requires manual tuning. See STRATEGY.md
            'Unresolved Issues' section 1.
        """
        self.optimistic_weight = weight

    def td_error(
        self,
        reward: float,
        current_features: np.ndarray,
        next_features: np.ndarray,
        done: bool,
    ) -> float:
        """
        Compute the TD error δ for this GVF.

        δ = C_t + γ (1 - β) V(s') + β z(s') - V(s)

        Args:
            reward: Environment reward r_t.
            current_features: Features x(s) at current state.
            next_features: Features x(s') at next state.
            done: Whether episode terminated.

        Returns:
            TD error value.
        """
        cumulant = self.cumulant(reward)
        beta = self.stopping_condition(next_features, current_features)

        current_value = self.value(current_features)

        if done:
            # Episode terminated - no future value
            target = cumulant
        elif beta > 0.5:
            # Option terminates - use stopping value
            try:
                target = cumulant + self.discount * self.stopping_value(next_features)
            except NotImplementedError:
                # Fall back to standard value if optimistic weight not set
                target = cumulant + self.discount * self.value(next_features)
        else:
            # Option continues - use standard value
            target = cumulant + self.discount * self.value(next_features)

        return target - current_value

    def update(
        self,
        reward: float,
        current_features: np.ndarray,
        next_features: np.ndarray,
        done: bool,
    ) -> float:
        """
        Update weights using TD learning with eligibility traces.

        Uses the UWT (UpdateWeights&Traces) procedure to learn
        off-policy from experience.

        Args:
            reward: Environment reward r_t.
            current_features: Features x(s) at current state.
            next_features: Features x(s') at next state.
            done: Whether episode terminated.

        Returns:
            TD error for logging.

        Raises:
            NotImplementedError: UWT update not fully implemented.
        """
        raise NotImplementedError(
            "UWT (UpdateWeights&Traces) update procedure not implemented. "
            "Requires specific hyperparameters and eligibility trace mechanisms. "
            "See STRATEGY.md 'Unresolved Issues' section 4."
        )

    def reset_traces(self) -> None:
        """Reset eligibility traces to zero (start of episode)."""
        self.traces = np.zeros(self.feature_dim, dtype=np.float32)


class GVFFactory:
    """
    Factory for creating GVFs for Crafter subtasks.

    Provides convenient methods to create properly configured GVFs
    for each of the 22 achievements.
    """

    def __init__(
        self,
        feature_dim: int = TOTAL_FEATURE_DIM,
        learning_rate: float = 0.01,
        discount: float = 0.99,
    ) -> None:
        """
        Initialize the GVF factory.

        Args:
            feature_dim: Dimension of feature vectors.
            learning_rate: Default learning rate for GVFs.
            discount: Default discount factor.
        """
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.discount = discount

    def create_for_achievement(
        self,
        achievement_name: str,
        feature_index: int,
    ) -> GeneralValueFunction:
        """
        Create a GVF for a specific achievement.

        Args:
            achievement_name: Name of the achievement (for logging).
            feature_index: Index of the achievement in feature vector.

        Returns:
            Configured GeneralValueFunction instance.
        """
        gvf = GeneralValueFunction(
            target_feature_index=feature_index,
            feature_dim=self.feature_dim,
            learning_rate=self.learning_rate,
            discount=self.discount,
        )
        return gvf

    def create_all(
        self,
        achievement_to_index: Dict[str, int],
    ) -> Dict[str, GeneralValueFunction]:
        """
        Create GVFs for all achievements.

        Args:
            achievement_to_index: Mapping from achievement names to feature indices.

        Returns:
            Dictionary mapping achievement names to GVF instances.
        """
        return {
            name: self.create_for_achievement(name, idx)
            for name, idx in achievement_to_index.items()
        }
