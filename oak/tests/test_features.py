"""
Unit tests for FeatureExtractor.
"""

import pytest
import numpy as np

from oak.features import FeatureExtractor, PixelFeatureExtractor
from oak.constants import (
    ACHIEVEMENTS,
    TOTAL_FEATURE_DIM,
    MAX_INVENTORY_VALUE,
    MAX_VITAL_VALUE,
)


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a feature extractor."""
        return FeatureExtractor()

    @pytest.fixture
    def mock_obs(self):
        """Create a mock observation."""
        return np.zeros((64, 64, 3), dtype=np.uint8)

    @pytest.fixture
    def mock_info(self):
        """Create a mock info dictionary with default values."""
        return {
            "inventory": {
                "health": 9,
                "food": 9,
                "drink": 9,
                "energy": 9,
                "wood": 0,
                "stone": 0,
                "coal": 0,
                "iron": 0,
                "diamond": 0,
                "sapling": 0,
                "wood_pickaxe": 0,
                "stone_pickaxe": 0,
                "iron_pickaxe": 0,
                "wood_sword": 0,
                "stone_sword": 0,
                "iron_sword": 0,
            },
            "achievements": {name: 0 for name in ACHIEVEMENTS},
        }

    def test_init(self, extractor):
        """Test extractor initialization."""
        assert extractor.feature_dim == TOTAL_FEATURE_DIM
        assert len(extractor.achievement_indices) == len(ACHIEVEMENTS)
        assert len(extractor.inventory_indices) == 12
        assert len(extractor.vital_indices) == 4

    def test_get_feature_dim(self, extractor):
        """Test feature dimension getter."""
        assert extractor.get_feature_dim() == TOTAL_FEATURE_DIM

    def test_extract_returns_correct_shape(self, extractor, mock_obs, mock_info):
        """Test that extract returns correct shape."""
        features = extractor.extract(mock_obs, mock_info)
        assert features.shape == (TOTAL_FEATURE_DIM,)
        assert features.dtype == np.float32

    def test_extract_achievements_zero(self, extractor, mock_obs, mock_info):
        """Test achievement features are 0 when not unlocked."""
        features = extractor.extract(mock_obs, mock_info)

        # First 22 features should be 0 (no achievements)
        for i in range(22):
            assert features[i] == 0.0

    def test_extract_achievements_one(self, extractor, mock_obs, mock_info):
        """Test achievement features are 1 when unlocked."""
        mock_info["achievements"]["collect_wood"] = 1

        features = extractor.extract(mock_obs, mock_info)

        idx = extractor.achievement_indices["collect_wood"]
        assert features[idx] == 1.0

    def test_extract_inventory_normalized(self, extractor, mock_obs, mock_info):
        """Test inventory features are normalized to [0, 1]."""
        mock_info["inventory"]["wood"] = 5

        features = extractor.extract(mock_obs, mock_info)

        idx = extractor.inventory_indices["wood"]
        expected = 5 / MAX_INVENTORY_VALUE
        assert features[idx] == pytest.approx(expected)

    def test_extract_vitals_normalized(self, extractor, mock_obs, mock_info):
        """Test vital features are normalized to [0, 1]."""
        mock_info["inventory"]["health"] = 6

        features = extractor.extract(mock_obs, mock_info)

        idx = extractor.vital_indices["health"]
        expected = 6 / MAX_VITAL_VALUE
        assert features[idx] == pytest.approx(expected)

    def test_extract_placed_objects(self, extractor, mock_obs, mock_info):
        """Test placed object features."""
        mock_info["achievements"]["place_table"] = 1

        features = extractor.extract(mock_obs, mock_info)

        assert features[extractor.table_placed_index] == 1.0
        assert features[extractor.furnace_placed_index] == 0.0

    def test_get_feature_index(self, extractor):
        """Test getting feature index by name."""
        assert extractor.get_feature_index("collect_wood") is not None
        assert extractor.get_feature_index("wood") is not None
        assert extractor.get_feature_index("health") is not None
        assert extractor.get_feature_index("invalid_name") is None

    def test_get_achievement_features(self, extractor, mock_obs, mock_info):
        """Test extracting achievement feature slice."""
        features = extractor.extract(mock_obs, mock_info)
        achievement_features = extractor.get_achievement_features(features)

        assert len(achievement_features) == 22

    def test_get_inventory_features(self, extractor, mock_obs, mock_info):
        """Test extracting inventory feature slice."""
        features = extractor.extract(mock_obs, mock_info)
        inventory_features = extractor.get_inventory_features(features)

        assert len(inventory_features) == 12

    def test_get_vital_features(self, extractor, mock_obs, mock_info):
        """Test extracting vital feature slice."""
        features = extractor.extract(mock_obs, mock_info)
        vital_features = extractor.get_vital_features(features)

        assert len(vital_features) == 4


class TestPixelFeatureExtractor:
    """Tests for PixelFeatureExtractor class."""

    def test_init(self):
        """Test initialization."""
        extractor = PixelFeatureExtractor()
        assert extractor.encoder is None
        assert extractor.get_feature_dim() == 256

    def test_extract_not_implemented(self):
        """Test that extract raises NotImplementedError."""
        extractor = PixelFeatureExtractor()
        obs = np.zeros((64, 64, 3))
        info = {}

        with pytest.raises(NotImplementedError):
            extractor.extract(obs, info)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
