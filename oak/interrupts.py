"""
Interrupt Manager for survival options.

This module implements a priority-based interrupt system that monitors
vital statistics (health, food, water, rest) and triggers survival
options when they fall below critical thresholds.

The interrupt system can preempt ongoing progression options to ensure
agent survival, implementing the survival-vs-progression tradeoff
described in STRATEGY.md.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oak.constants import (
    CRITICAL_VITAL_THRESHOLD,
    WARNING_VITAL_THRESHOLD,
    MAX_VITAL_VALUE,
    SurvivalPriority,
    VITAL_TO_SURVIVAL_ACHIEVEMENT,
    FOOD_ACHIEVEMENTS,
)
from oak.subtasks import Subtask, get_survival_subtasks


@dataclass
class VitalStatus:
    """
    Current status of a vital statistic.

    Attributes:
        name: Vital stat name (health, food, drink, energy).
        value: Current value (0-9).
        is_critical: Whether below critical threshold.
        is_warning: Whether below warning threshold.
        priority: Interrupt priority level.
    """

    name: str
    value: int
    is_critical: bool
    is_warning: bool
    priority: int


@dataclass
class InterruptRequest:
    """
    Request to interrupt current option for survival.

    Attributes:
        vital: The vital statistic triggering the interrupt.
        survival_option: Name of the survival option to activate.
        priority: Priority level of the interrupt.
        urgent: Whether this is an urgent (critical) interrupt.
    """

    vital: str
    survival_option: str
    priority: int
    urgent: bool


class InterruptManager:
    """
    Manages survival interrupts based on vital statistics.

    Monitors health, food, water (drink), and energy levels, triggering
    appropriate survival options when levels fall below thresholds.

    Priority ordering (highest to lowest):
    1. Health (if being attacked)
    2. Food (prevents health loss)
    3. Water/Drink (prevents health loss)
    4. Energy (prevents health loss)

    Attributes:
        critical_threshold: Level below which urgent interrupt triggers.
        warning_threshold: Level below which normal interrupt triggers.
        survival_subtasks: Available survival subtask instances.
    """

    def __init__(
        self,
        critical_threshold: int = CRITICAL_VITAL_THRESHOLD,
        warning_threshold: int = WARNING_VITAL_THRESHOLD,
    ) -> None:
        """
        Initialize the interrupt manager.

        Args:
            critical_threshold: Trigger urgent interrupt below this level.
            warning_threshold: Trigger normal interrupt below this level.
        """
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold

        # Initialize survival subtasks
        self.survival_subtasks = get_survival_subtasks()

        # Priority mapping for vitals
        self.vital_priorities = {
            "health": SurvivalPriority.HEALTH.value,
            "food": SurvivalPriority.FOOD.value,
            "drink": SurvivalPriority.DRINK.value,
            "energy": SurvivalPriority.ENERGY.value,
        }

        # Mapping from vital to preferred survival action
        self.vital_to_action = {
            "food": ["eat_cow", "eat_plant"],
            "drink": ["collect_drink"],
            "energy": ["wake_up"],
        }

        # Track previous vital levels for trend detection
        self._previous_vitals: Optional[Dict[str, int]] = None

    def get_vital_status(self, info: Dict[str, Any]) -> Dict[str, VitalStatus]:
        """
        Extract vital status from info dictionary.

        Args:
            info: Info dictionary from Crafter environment.

        Returns:
            Dictionary mapping vital names to VitalStatus instances.
        """
        inventory = info.get("inventory", {})
        statuses = {}

        for vital in ["health", "food", "drink", "energy"]:
            value = inventory.get(vital, MAX_VITAL_VALUE)
            statuses[vital] = VitalStatus(
                name=vital,
                value=value,
                is_critical=value <= self.critical_threshold,
                is_warning=value <= self.warning_threshold,
                priority=self.vital_priorities[vital],
            )

        return statuses

    def check_interrupt(
        self,
        info: Dict[str, Any],
        allow_warning: bool = False,
    ) -> Optional[InterruptRequest]:
        """
        Check if a survival interrupt should be triggered.

        Args:
            info: Current info dictionary from environment.
            allow_warning: If True, also trigger on warning (not just critical).

        Returns:
            InterruptRequest if interrupt needed, None otherwise.
        """
        statuses = self.get_vital_status(info)

        # Collect all triggering conditions
        interrupts: List[InterruptRequest] = []

        for vital, status in statuses.items():
            # Skip health - we can't directly restore it
            if vital == "health":
                continue

            should_trigger = status.is_critical or (allow_warning and status.is_warning)

            if should_trigger:
                # Get available survival actions for this vital
                actions = self.vital_to_action.get(vital, [])
                if actions:
                    interrupts.append(
                        InterruptRequest(
                            vital=vital,
                            survival_option=actions[0],  # Primary action
                            priority=status.priority,
                            urgent=status.is_critical,
                        )
                    )

        if not interrupts:
            return None

        # Return highest priority interrupt
        return max(interrupts, key=lambda x: x.priority)

    def check_combat_interrupt(
        self,
        info: Dict[str, Any],
        previous_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[InterruptRequest]:
        """
        Check if combat interrupt is needed (health decreasing from attack).

        Args:
            info: Current info dictionary.
            previous_info: Previous info dictionary.

        Returns:
            Combat InterruptRequest if under attack, None otherwise.
        """
        if previous_info is None:
            return None

        current_health = info.get("inventory", {}).get("health", MAX_VITAL_VALUE)
        previous_health = previous_info.get("inventory", {}).get("health", MAX_VITAL_VALUE)

        # Health decreased - likely under attack
        if current_health < previous_health:
            # Could be zombie or skeleton - check semantic grid if available
            # For now, assume zombie (more common in open areas)
            return InterruptRequest(
                vital="health",
                survival_option="defeat_zombie",
                priority=SurvivalPriority.COMBAT.value,
                urgent=current_health <= self.critical_threshold,
            )

        return None

    def get_recommended_survival_action(
        self,
        info: Dict[str, Any],
    ) -> Optional[str]:
        """
        Get the recommended survival action based on current vitals.

        Considers which survival options have already been completed
        and which vitals are lowest.

        Args:
            info: Current info dictionary.

        Returns:
            Name of recommended survival subtask, or None if all vitals OK.
        """
        interrupt = self.check_interrupt(info, allow_warning=True)

        if interrupt is None:
            return None

        # Check if primary action is available, otherwise try alternatives
        vital = interrupt.vital
        actions = self.vital_to_action.get(vital, [])

        for action in actions:
            subtask = self.survival_subtasks.get(action)
            if subtask is not None:
                # Could check if action is feasible (e.g., cow nearby)
                return action

        return None

    def get_vital_trend(
        self,
        info: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        Get trend (change) in vital levels.

        Args:
            info: Current info dictionary.

        Returns:
            Dictionary mapping vital names to change since last check.
        """
        inventory = info.get("inventory", {})
        trends = {}

        for vital in ["health", "food", "drink", "energy"]:
            current = inventory.get(vital, MAX_VITAL_VALUE)
            if self._previous_vitals is not None:
                previous = self._previous_vitals.get(vital, MAX_VITAL_VALUE)
                trends[vital] = current - previous
            else:
                trends[vital] = 0

        # Update previous vitals
        self._previous_vitals = {
            vital: inventory.get(vital, MAX_VITAL_VALUE)
            for vital in ["health", "food", "drink", "energy"]
        }

        return trends

    def reset(self) -> None:
        """Reset interrupt manager state for new episode."""
        self._previous_vitals = None

    def should_preempt(
        self,
        current_option: str,
        interrupt: InterruptRequest,
    ) -> bool:
        """
        Determine if interrupt should preempt the current option.

        Args:
            current_option: Name of currently active option.
            interrupt: The interrupt request.

        Returns:
            True if interrupt should preempt current option.
        """
        # Always preempt for urgent (critical) interrupts
        if interrupt.urgent:
            return True

        # Don't preempt if already doing a survival action
        if current_option in self.survival_subtasks:
            return False

        # Preempt for high-priority warnings if current is low priority
        # This is a simplified heuristic
        return interrupt.priority >= SurvivalPriority.FOOD.value
