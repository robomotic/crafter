"""
Feature extraction for the OaK framework.

This module provides the FeatureExtractor class that converts Crafter's
info dictionary into a feature vector x(s) suitable for GVF computation.

Feature Vector Structure (40 dimensions):
    - Binary achievement features (22)
    - Inventory count features (12)
    - Vital status features (4)
    - Placed object features (2)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from oak.constants import (
    ACHIEVEMENTS,
    INVENTORY_ITEMS,
    VITAL_STATS,
    MAX_INVENTORY_VALUE,
    MAX_VITAL_VALUE,
    TOTAL_FEATURE_DIM,
    NUM_ACHIEVEMENT_FEATURES,
    NUM_INVENTORY_FEATURES,
    NUM_VITAL_FEATURES,
)


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    This allows for different feature extraction strategies:
    - Privileged info dict features (implemented)
    - Pixel-based learned features (future work)
    """

    @abstractmethod
    def extract(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Extract feature vector from observation and info.

        Args:
            obs: Raw observation array (64, 64, 3) RGB image.
            info: Info dictionary from Crafter environment.

        Returns:
            Feature vector x(s) as numpy array.
        """
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimension of the feature vector."""
        pass


class FeatureExtractor(BaseFeatureExtractor):
    """
    Feature extractor using privileged info dictionary.

    Extracts features directly from Crafter's info dict, which provides
    ground-truth inventory counts and achievement states.

    Attributes:
        feature_dim: Total dimension of feature vector (40).
        achievement_indices: Mapping from achievement name to feature index.
        inventory_indices: Mapping from inventory item to feature index.
        vital_indices: Mapping from vital stat to feature index.
    """

    def __init__(self) -> None:
        """Initialize the feature extractor with index mappings."""
        self.feature_dim = TOTAL_FEATURE_DIM

        # Build index mappings for each feature group
        offset = 0

        # Achievement features (0-21)
        self.achievement_indices: Dict[str, int] = {
            name: offset + i for i, name in enumerate(ACHIEVEMENTS)
        }
        offset += NUM_ACHIEVEMENT_FEATURES

        # Inventory features (22-33)
        self.inventory_indices: Dict[str, int] = {
            name: offset + i for i, name in enumerate(INVENTORY_ITEMS)
        }
        offset += NUM_INVENTORY_FEATURES

        # Vital features (34-37)
        self.vital_indices: Dict[str, int] = {
            name: offset + i for i, name in enumerate(VITAL_STATS)
        }
        offset += NUM_VITAL_FEATURES

        # Placed object features (38-39)
        self.table_placed_index = offset
        self.furnace_placed_index = offset + 1

    def extract(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Extract feature vector from Crafter info dictionary.

        Args:
            obs: Raw observation (unused in privileged mode).
            info: Info dictionary containing 'inventory' and 'achievements'.

        Returns:
            Feature vector of shape (40,) with:
                - Binary achievement flags (0 or 1)
                - Normalized inventory counts (0 to 1)
                - Normalized vital levels (0 to 1)
                - Binary placed object flags (0 or 1)
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)

        inventory = info.get("inventory", {})
        achievements = info.get("achievements", {})

        # Extract achievement features (binary: unlocked in this episode)
        for name, idx in self.achievement_indices.items():
            # Achievement is unlocked if count > 0
            features[idx] = 1.0 if achievements.get(name, 0) > 0 else 0.0

        # Extract inventory features (normalized to [0, 1])
        for name, idx in self.inventory_indices.items():
            count = inventory.get(name, 0)
            features[idx] = count / MAX_INVENTORY_VALUE

        # Extract vital features (normalized to [0, 1])
        for name, idx in self.vital_indices.items():
            level = inventory.get(name, MAX_VITAL_VALUE)
            features[idx] = level / MAX_VITAL_VALUE

        # Extract placed object features
        # These are inferred from semantic grid if available,
        # otherwise we track them based on achievement history
        # For now, use achievements as proxy
        features[self.table_placed_index] = (
            1.0 if achievements.get("place_table", 0) > 0 else 0.0
        )
        features[self.furnace_placed_index] = (
            1.0 if achievements.get("place_furnace", 0) > 0 else 0.0
        )

        return features

    def get_feature_dim(self) -> int:
        """Return the dimension of the feature vector."""
        return self.feature_dim

    def get_feature_index(self, feature_name: str) -> Optional[int]:
        """
        Get the index of a named feature in the feature vector.

        Args:
            feature_name: Name of the feature (achievement, inventory item, or vital).

        Returns:
            Index in the feature vector, or None if not found.
        """
        if feature_name in self.achievement_indices:
            return self.achievement_indices[feature_name]
        if feature_name in self.inventory_indices:
            return self.inventory_indices[feature_name]
        if feature_name in self.vital_indices:
            return self.vital_indices[feature_name]
        if feature_name == "table_placed":
            return self.table_placed_index
        if feature_name == "furnace_placed":
            return self.furnace_placed_index
        return None

    def get_achievement_features(self, features: np.ndarray) -> np.ndarray:
        """Extract only the achievement portion of the feature vector."""
        return features[:NUM_ACHIEVEMENT_FEATURES]

    def get_inventory_features(self, features: np.ndarray) -> np.ndarray:
        """Extract only the inventory portion of the feature vector."""
        start = NUM_ACHIEVEMENT_FEATURES
        end = start + NUM_INVENTORY_FEATURES
        return features[start:end]

    def get_vital_features(self, features: np.ndarray) -> np.ndarray:
        """Extract only the vital stats portion of the feature vector."""
        start = NUM_ACHIEVEMENT_FEATURES + NUM_INVENTORY_FEATURES
        end = start + NUM_VITAL_FEATURES
        return features[start:end]


class PixelFeatureExtractor(BaseFeatureExtractor):
    """
    Pixel-based feature extractor using learned representations.

    This class is a placeholder for future implementation using CNNs
    or other learned encoders to extract features from raw pixel observations.
    """

    def __init__(self, encoder: Optional[Any] = None) -> None:
        """
        Initialize with optional pre-trained encoder.

        Args:
            encoder: Neural network encoder (e.g., CNN, VAE).
        """
        self.encoder = encoder
        self._feature_dim = 256  # Default latent dimension

    def extract(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from pixel observations.

        Args:
            obs: RGB image observation of shape (64, 64, 3).
            info: Info dictionary (unused in pixel mode).

        Returns:
            Learned feature representation.

        Raises:
            NotImplementedError: Pixel-based extraction not yet implemented.
        """
        raise NotImplementedError(
            "Pixel-based feature extraction requires a trained encoder. "
            "This is planned for future implementation. "
            "Use FeatureExtractor with privileged info dict instead."
        )

    def get_feature_dim(self) -> int:
        """Return the dimension of the learned feature vector."""
        return self._feature_dim
