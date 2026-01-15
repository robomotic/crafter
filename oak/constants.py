"""
Constants for the OaK (Options and Knowledge) framework in Crafter.

This module defines all achievements, actions, inventory items, vital statistics,
and threshold values used throughout the OaK agent implementation.
"""

from enum import Enum, auto
from typing import Dict, List, Set, Tuple

# =============================================================================
# ACHIEVEMENTS (22 semantically meaningful milestones)
# =============================================================================

ACHIEVEMENTS: Tuple[str, ...] = (
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
)

NUM_ACHIEVEMENTS: int = len(ACHIEVEMENTS)

# =============================================================================
# ACTIONS (17 discrete actions)
# =============================================================================

ACTIONS: Tuple[str, ...] = (
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
)

NUM_ACTIONS: int = len(ACTIONS)

# Action index mapping for quick lookup
ACTION_TO_INDEX: Dict[str, int] = {action: i for i, action in enumerate(ACTIONS)}

# =============================================================================
# INVENTORY ITEMS
# =============================================================================

# Resource items that can be collected
RESOURCE_ITEMS: Tuple[str, ...] = (
    "sapling",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
)

# Tool items that can be crafted
TOOL_ITEMS: Tuple[str, ...] = (
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
)

# All inventory items
INVENTORY_ITEMS: Tuple[str, ...] = RESOURCE_ITEMS + TOOL_ITEMS

# Maximum value for any inventory item
MAX_INVENTORY_VALUE: int = 9

# =============================================================================
# VITAL STATISTICS
# =============================================================================

VITAL_STATS: Tuple[str, ...] = (
    "health",
    "food",
    "drink",
    "energy",
)

# Maximum value for vital stats
MAX_VITAL_VALUE: int = 9

# Initial vital values
INITIAL_VITAL_VALUE: int = 9

# =============================================================================
# THRESHOLDS
# =============================================================================

# Critical threshold below which survival options trigger (out of 9)
CRITICAL_VITAL_THRESHOLD: int = 3

# Warning threshold for early intervention (out of 9)
WARNING_VITAL_THRESHOLD: int = 5

# =============================================================================
# SURVIVAL OPTION PRIORITIES (higher = more urgent)
# =============================================================================


class SurvivalPriority(Enum):
    """Priority levels for survival options. Higher values = more urgent."""

    HEALTH = 100  # Most critical - direct death risk
    FOOD = 80  # Causes health loss when depleted
    DRINK = 70  # Causes health loss when depleted
    ENERGY = 60  # Causes health loss when depleted
    COMBAT = 50  # Reactive to monster attacks


# Mapping from vital stat to corresponding survival achievement
VITAL_TO_SURVIVAL_ACHIEVEMENT: Dict[str, str] = {
    "food": "eat_cow",  # or eat_plant
    "drink": "collect_drink",
    "energy": "wake_up",
}

# Alternative food sources
FOOD_ACHIEVEMENTS: Tuple[str, ...] = ("eat_cow", "eat_plant")

# =============================================================================
# TECHNOLOGY TREE PREREQUISITES
# From STRATEGY.md Prerequisite Table
# =============================================================================

PREREQUISITES: Dict[str, Set[str]] = {
    # Root tasks (no prerequisites)
    "collect_wood": set(),
    "collect_sapling": set(),
    "collect_drink": set(),
    "eat_cow": set(),
    "defeat_zombie": set(),
    "defeat_skeleton": set(),
    # Wood-dependent tasks
    "place_table": {"collect_wood"},
    "make_wood_pickaxe": {"collect_wood", "place_table"},
    "make_wood_sword": {"collect_wood", "place_table"},
    # Stone-dependent tasks
    "collect_stone": {"make_wood_pickaxe"},
    "make_stone_pickaxe": {"collect_stone", "place_table"},
    "make_stone_sword": {"collect_stone", "place_table"},
    "place_stone": {"collect_stone"},
    # Coal and furnace
    "collect_coal": {"make_stone_pickaxe"},
    "place_furnace": {"collect_coal", "collect_stone"},
    # Iron-dependent tasks
    "collect_iron": {"make_stone_pickaxe", "place_furnace"},
    "make_iron_pickaxe": {"collect_iron", "place_furnace"},
    "make_iron_sword": {"collect_iron", "place_furnace"},
    # Diamond (terminal achievement)
    "collect_diamond": {"make_iron_pickaxe"},
    # Plant-related
    "place_plant": {"collect_sapling"},
    "eat_plant": {"place_plant"},
    # Sleep
    "wake_up": set(),  # Can attempt anytime but risky without shelter
}

# =============================================================================
# REWARD STRUCTURE
# =============================================================================

# Reward for unlocking an achievement for the first time
ACHIEVEMENT_REWARD: float = 1.0

# Reward/penalty per health point change
HEALTH_REWARD_PER_POINT: float = 0.1

# =============================================================================
# FEATURE DIMENSIONS
# =============================================================================

# Number of binary achievement features
NUM_ACHIEVEMENT_FEATURES: int = NUM_ACHIEVEMENTS  # 22

# Number of inventory count features
NUM_INVENTORY_FEATURES: int = len(INVENTORY_ITEMS)  # 12

# Number of vital status features
NUM_VITAL_FEATURES: int = len(VITAL_STATS)  # 4

# Number of placed object features (table, furnace)
NUM_PLACED_FEATURES: int = 2

# Total feature vector dimension
TOTAL_FEATURE_DIM: int = (
    NUM_ACHIEVEMENT_FEATURES
    + NUM_INVENTORY_FEATURES
    + NUM_VITAL_FEATURES
    + NUM_PLACED_FEATURES
)  # 40
