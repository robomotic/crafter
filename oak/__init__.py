"""
OaK (Options and Knowledge) Framework for Crafter.

This package implements Richard Sutton's OaK architecture for the Crafter
environment, using reward-respecting subtasks of feature attainment.

Key Components:
    - OaKAgent: Main agent class compatible with Crafter's Gymnasium API
    - GeneralValueFunction: GVF implementation with 4 components
    - Option: Temporally extended action policies
    - FeatureExtractor: Converts Crafter state to feature vectors
    - TechnologyTree: Manages achievement dependencies
    - InterruptManager: Priority-based survival option system
"""

from oak.agent import OaKAgent
from oak.gvf import GeneralValueFunction
from oak.options import Option, OptionManager
from oak.features import FeatureExtractor
from oak.tech_tree import TechnologyTree
from oak.interrupts import InterruptManager
from oak.subtasks import Subtask, SubtaskType
from oak.constants import (
    ACHIEVEMENTS,
    ACTIONS,
    NUM_ACTIONS,
    INVENTORY_ITEMS,
    VITAL_STATS,
    CRITICAL_VITAL_THRESHOLD,
    TOTAL_FEATURE_DIM,
    PREREQUISITES,
)

__version__ = "0.1.0"
__author__ = "OaK Research Team"

__all__ = [
    "OaKAgent",
    "GeneralValueFunction",
    "Option",
    "OptionManager",
    "FeatureExtractor",
    "TechnologyTree",
    "InterruptManager",
    "Subtask",
    "SubtaskType",
    "ACHIEVEMENTS",
    "ACTIONS",
    "NUM_ACTIONS",
    "INVENTORY_ITEMS",
    "VITAL_STATS",
    "CRITICAL_VITAL_THRESHOLD",
    "TOTAL_FEATURE_DIM",
    "PREREQUISITES",
]
