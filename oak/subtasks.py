"""
Subtask definitions for the OaK framework.

This module defines the 22 semantically meaningful achievements in Crafter
as subtasks, each with specific termination conditions based on the
STRATEGY.md specification.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

from oak.constants import ACHIEVEMENTS, VITAL_TO_SURVIVAL_ACHIEVEMENT


class SubtaskType(Enum):
    """Categories of subtasks in Crafter."""

    RESOURCE_COLLECTION = auto()  # Collect wood, stone, coal, iron, diamond
    CRAFTING = auto()  # Make pickaxes and swords
    PLACEMENT = auto()  # Place table, furnace, stone, plant
    SURVIVAL = auto()  # Eat, drink, sleep
    COMBAT = auto()  # Defeat zombie, skeleton


@dataclass
class SubtaskDefinition:
    """
    Definition of a subtask including its termination condition.

    Attributes:
        name: Achievement name (e.g., "collect_wood").
        subtask_type: Category of the subtask.
        description: Human-readable description.
        inventory_key: Key to check in inventory for termination (if applicable).
        is_survival: Whether this is a survival-related subtask.
        priority: Priority level for option selection (higher = more important).
    """

    name: str
    subtask_type: SubtaskType
    description: str
    inventory_key: Optional[str] = None
    is_survival: bool = False
    priority: int = 50


# =============================================================================
# SUBTASK DEFINITIONS
# =============================================================================

SUBTASK_DEFINITIONS: Dict[str, SubtaskDefinition] = {
    # Resource Collection
    "collect_wood": SubtaskDefinition(
        name="collect_wood",
        subtask_type=SubtaskType.RESOURCE_COLLECTION,
        description="Collect wood by interacting with trees",
        inventory_key="wood",
        priority=90,  # High priority - root of tech tree
    ),
    "collect_stone": SubtaskDefinition(
        name="collect_stone",
        subtask_type=SubtaskType.RESOURCE_COLLECTION,
        description="Collect stone using a wood pickaxe",
        inventory_key="stone",
        priority=80,
    ),
    "collect_coal": SubtaskDefinition(
        name="collect_coal",
        subtask_type=SubtaskType.RESOURCE_COLLECTION,
        description="Collect coal using a stone pickaxe",
        inventory_key="coal",
        priority=70,
    ),
    "collect_iron": SubtaskDefinition(
        name="collect_iron",
        subtask_type=SubtaskType.RESOURCE_COLLECTION,
        description="Collect iron using a stone pickaxe (requires furnace)",
        inventory_key="iron",
        priority=60,
    ),
    "collect_diamond": SubtaskDefinition(
        name="collect_diamond",
        subtask_type=SubtaskType.RESOURCE_COLLECTION,
        description="Collect diamond using an iron pickaxe",
        inventory_key="diamond",
        priority=100,  # Terminal achievement - highest reward
    ),
    "collect_sapling": SubtaskDefinition(
        name="collect_sapling",
        subtask_type=SubtaskType.RESOURCE_COLLECTION,
        description="Collect a sapling from grass",
        inventory_key="sapling",
        priority=30,
    ),
    "collect_drink": SubtaskDefinition(
        name="collect_drink",
        subtask_type=SubtaskType.SURVIVAL,
        description="Drink water from a lake to restore hydration",
        inventory_key="drink",
        is_survival=True,
        priority=75,
    ),
    # Crafting - Pickaxes
    "make_wood_pickaxe": SubtaskDefinition(
        name="make_wood_pickaxe",
        subtask_type=SubtaskType.CRAFTING,
        description="Craft a wood pickaxe at a table",
        inventory_key="wood_pickaxe",
        priority=85,
    ),
    "make_stone_pickaxe": SubtaskDefinition(
        name="make_stone_pickaxe",
        subtask_type=SubtaskType.CRAFTING,
        description="Craft a stone pickaxe at a table",
        inventory_key="stone_pickaxe",
        priority=75,
    ),
    "make_iron_pickaxe": SubtaskDefinition(
        name="make_iron_pickaxe",
        subtask_type=SubtaskType.CRAFTING,
        description="Craft an iron pickaxe at a furnace",
        inventory_key="iron_pickaxe",
        priority=65,
    ),
    # Crafting - Swords
    "make_wood_sword": SubtaskDefinition(
        name="make_wood_sword",
        subtask_type=SubtaskType.CRAFTING,
        description="Craft a wood sword at a table",
        inventory_key="wood_sword",
        priority=40,
    ),
    "make_stone_sword": SubtaskDefinition(
        name="make_stone_sword",
        subtask_type=SubtaskType.CRAFTING,
        description="Craft a stone sword at a table",
        inventory_key="stone_sword",
        priority=35,
    ),
    "make_iron_sword": SubtaskDefinition(
        name="make_iron_sword",
        subtask_type=SubtaskType.CRAFTING,
        description="Craft an iron sword at a furnace",
        inventory_key="iron_sword",
        priority=30,
    ),
    # Placement
    "place_table": SubtaskDefinition(
        name="place_table",
        subtask_type=SubtaskType.PLACEMENT,
        description="Place a crafting table (requires wood)",
        priority=88,  # Critical for all crafting
    ),
    "place_furnace": SubtaskDefinition(
        name="place_furnace",
        subtask_type=SubtaskType.PLACEMENT,
        description="Place a furnace (requires coal and stone)",
        priority=68,  # Required for iron tier
    ),
    "place_stone": SubtaskDefinition(
        name="place_stone",
        subtask_type=SubtaskType.PLACEMENT,
        description="Place a stone block",
        priority=20,
    ),
    "place_plant": SubtaskDefinition(
        name="place_plant",
        subtask_type=SubtaskType.PLACEMENT,
        description="Plant a sapling",
        priority=25,
    ),
    # Survival - Food
    "eat_cow": SubtaskDefinition(
        name="eat_cow",
        subtask_type=SubtaskType.SURVIVAL,
        description="Hunt and eat a cow to restore food",
        is_survival=True,
        priority=70,
    ),
    "eat_plant": SubtaskDefinition(
        name="eat_plant",
        subtask_type=SubtaskType.SURVIVAL,
        description="Eat a grown plant to restore food",
        is_survival=True,
        priority=65,
    ),
    "wake_up": SubtaskDefinition(
        name="wake_up",
        subtask_type=SubtaskType.SURVIVAL,
        description="Sleep and wake up safely to restore energy",
        is_survival=True,
        priority=60,
    ),
    # Combat
    "defeat_zombie": SubtaskDefinition(
        name="defeat_zombie",
        subtask_type=SubtaskType.COMBAT,
        description="Defeat a zombie in combat",
        is_survival=True,  # Defensive action
        priority=55,
    ),
    "defeat_skeleton": SubtaskDefinition(
        name="defeat_skeleton",
        subtask_type=SubtaskType.COMBAT,
        description="Defeat a skeleton in combat",
        is_survival=True,  # Defensive action
        priority=50,
    ),
}


class Subtask:
    """
    Represents a subtask with its termination condition.

    A subtask encapsulates the logic for detecting when a specific
    achievement has been attained, based on inventory/state changes.

    Attributes:
        definition: The SubtaskDefinition for this subtask.
        name: Achievement name.
        subtask_type: Category of the subtask.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a subtask by name.

        Args:
            name: Achievement name (must be in SUBTASK_DEFINITIONS).

        Raises:
            ValueError: If name is not a valid achievement.
        """
        if name not in SUBTASK_DEFINITIONS:
            raise ValueError(
                f"Unknown subtask: {name}. "
                f"Valid subtasks: {list(SUBTASK_DEFINITIONS.keys())}"
            )

        self.definition = SUBTASK_DEFINITIONS[name]
        self.name = name
        self.subtask_type = self.definition.subtask_type

    def check_termination(
        self,
        current_info: Dict[str, Any],
        previous_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if the subtask termination condition is met.

        The termination condition β(s) = 1 when the target feature
        has been attained (achievement unlocked or inventory increased).

        Args:
            current_info: Current info dict from environment.
            previous_info: Previous info dict (for detecting changes).

        Returns:
            True if termination condition is met (β(s) = 1).
        """
        achievements = current_info.get("achievements", {})
        inventory = current_info.get("inventory", {})

        # Check if achievement was just unlocked
        current_count = achievements.get(self.name, 0)
        if previous_info is not None:
            prev_achievements = previous_info.get("achievements", {})
            prev_count = prev_achievements.get(self.name, 0)
            if current_count > prev_count:
                return True

        # For inventory-based subtasks, check for increase
        if self.definition.inventory_key is not None:
            current_value = inventory.get(self.definition.inventory_key, 0)
            if previous_info is not None:
                prev_inventory = previous_info.get("inventory", {})
                prev_value = prev_inventory.get(self.definition.inventory_key, 0)
                if current_value > prev_value:
                    return True

        # For vital-based subtasks (drink, food, energy)
        if self.name == "collect_drink":
            current_drink = inventory.get("drink", 0)
            if previous_info is not None:
                prev_drink = previous_info.get("inventory", {}).get("drink", 0)
                if current_drink > prev_drink:
                    return True

        if self.name in ("eat_cow", "eat_plant"):
            current_food = inventory.get("food", 0)
            if previous_info is not None:
                prev_food = previous_info.get("inventory", {}).get("food", 0)
                if current_food > prev_food:
                    return True

        if self.name == "wake_up":
            current_energy = inventory.get("energy", 0)
            if previous_info is not None:
                prev_energy = previous_info.get("inventory", {}).get("energy", 0)
                if current_energy > prev_energy:
                    return True

        return False

    def is_completed(self, info: Dict[str, Any]) -> bool:
        """
        Check if this subtask has ever been completed in the episode.

        Args:
            info: Current info dict from environment.

        Returns:
            True if the achievement has been unlocked at least once.
        """
        achievements = info.get("achievements", {})
        return achievements.get(self.name, 0) > 0

    def get_priority(self) -> int:
        """Get the priority level for this subtask."""
        return self.definition.priority

    def is_survival_task(self) -> bool:
        """Check if this is a survival-related subtask."""
        return self.definition.is_survival

    def __repr__(self) -> str:
        return f"Subtask({self.name}, type={self.subtask_type.name})"


def get_all_subtasks() -> Dict[str, Subtask]:
    """
    Create Subtask instances for all 22 achievements.

    Returns:
        Dictionary mapping achievement names to Subtask instances.
    """
    return {name: Subtask(name) for name in SUBTASK_DEFINITIONS}


def get_survival_subtasks() -> Dict[str, Subtask]:
    """
    Get only survival-related subtasks.

    Returns:
        Dictionary of subtasks that are marked as survival tasks.
    """
    return {
        name: Subtask(name)
        for name, defn in SUBTASK_DEFINITIONS.items()
        if defn.is_survival
    }


def get_progression_subtasks() -> Dict[str, Subtask]:
    """
    Get non-survival subtasks (tech tree progression).

    Returns:
        Dictionary of subtasks for technology tree progression.
    """
    return {
        name: Subtask(name)
        for name, defn in SUBTASK_DEFINITIONS.items()
        if not defn.is_survival
    }
