"""
Unit tests for General Value Functions (GVFs).
"""

import pytest
import numpy as np

from oak.gvf import GeneralValueFunction, GVFFactory
from oak.constants import TOTAL_FEATURE_DIM, ACHIEVEMENTS


class TestGeneralValueFunction:
    """Tests for GeneralValueFunction class."""

    @pytest.fixture
    def gvf(self):
        """Create a GVF targeting feature index 0."""
        return GeneralValueFunction(
            target_feature_index=0,
            feature_dim=TOTAL_FEATURE_DIM,
            learning_rate=0.01,
            discount=0.99,
        )

    def test_init(self, gvf):
        """Test GVF initialization."""
        assert gvf.target_feature_index == 0
        assert gvf.feature_dim == TOTAL_FEATURE_DIM
        assert gvf.weights.shape == (TOTAL_FEATURE_DIM,)
        assert gvf.traces.shape == (TOTAL_FEATURE_DIM,)
        assert gvf.optimistic_weight is None

    def test_cumulant(self, gvf):
        """Test cumulant equals reward (reward-respecting)."""
        assert gvf.cumulant(1.0) == 1.0
        assert gvf.cumulant(-0.1) == -0.1
        assert gvf.cumulant(0.0) == 0.0

    def test_stopping_condition_feature_increase(self, gvf):
        """Test stopping condition when target feature increases."""
        current = np.zeros(TOTAL_FEATURE_DIM)
        current[0] = 1.0  # Target feature is high

        previous = np.zeros(TOTAL_FEATURE_DIM)
        previous[0] = 0.0  # Was low before

        beta = gvf.stopping_condition(current, previous)
        assert beta == 1.0

    def test_stopping_condition_no_change(self, gvf):
        """Test stopping condition when feature doesn't change."""
        current = np.zeros(TOTAL_FEATURE_DIM)
        previous = np.zeros(TOTAL_FEATURE_DIM)

        beta = gvf.stopping_condition(current, previous)
        assert beta == 0.0

    def test_stopping_condition_no_previous(self, gvf):
        """Test stopping condition without previous state."""
        features = np.zeros(TOTAL_FEATURE_DIM)
        features[0] = 1.0  # Target feature is high

        beta = gvf.stopping_condition(features)
        assert beta == 1.0

    def test_value_computation(self, gvf):
        """Test value computation V(s) = w^T x(s)."""
        gvf.weights = np.ones(TOTAL_FEATURE_DIM)
        features = np.ones(TOTAL_FEATURE_DIM)

        value = gvf.value(features)
        assert value == TOTAL_FEATURE_DIM

    def test_stopping_value_not_implemented(self, gvf):
        """Test stopping value raises error without optimistic weight."""
        features = np.zeros(TOTAL_FEATURE_DIM)

        with pytest.raises(NotImplementedError):
            gvf.stopping_value(features)

    def test_stopping_value_with_optimistic_weight(self, gvf):
        """Test stopping value with optimistic weight set."""
        gvf.select_optimistic_weight(10.0)
        gvf.weights = np.zeros(TOTAL_FEATURE_DIM)

        features = np.zeros(TOTAL_FEATURE_DIM)
        features[0] = 1.0  # Target feature is 1

        z = gvf.stopping_value(features)
        # z = w^T x + (w̄ - w_i) * x_i = 0 + (10 - 0) * 1 = 10
        assert z == 10.0

    def test_select_optimistic_weight(self, gvf):
        """Test setting optimistic weight."""
        gvf.select_optimistic_weight(5.0)
        assert gvf.optimistic_weight == 5.0

    def test_td_error_basic(self, gvf):
        """Test TD error computation."""
        gvf.weights = np.zeros(TOTAL_FEATURE_DIM)
        current = np.zeros(TOTAL_FEATURE_DIM)
        next_features = np.zeros(TOTAL_FEATURE_DIM)

        # δ = r + γ V(s') - V(s) = 1 + 0.99 * 0 - 0 = 1
        td_error = gvf.td_error(1.0, current, next_features, done=False)
        assert td_error == 1.0

    def test_update_not_implemented(self, gvf):
        """Test that update raises NotImplementedError."""
        current = np.zeros(TOTAL_FEATURE_DIM)
        next_features = np.zeros(TOTAL_FEATURE_DIM)

        with pytest.raises(NotImplementedError):
            gvf.update(1.0, current, next_features, done=False)

    def test_reset_traces(self, gvf):
        """Test resetting eligibility traces."""
        gvf.traces = np.ones(TOTAL_FEATURE_DIM)
        gvf.reset_traces()
        assert np.all(gvf.traces == 0)


class TestGVFFactory:
    """Tests for GVF factory."""

    def test_create_for_achievement(self):
        """Test creating GVF for a specific achievement."""
        factory = GVFFactory()
        gvf = factory.create_for_achievement("collect_wood", feature_index=6)

        assert gvf.target_feature_index == 6
        assert gvf.feature_dim == TOTAL_FEATURE_DIM

    def test_create_all(self):
        """Test creating GVFs for all achievements."""
        factory = GVFFactory()
        indices = {name: i for i, name in enumerate(ACHIEVEMENTS)}

        gvfs = factory.create_all(indices)

        assert len(gvfs) == len(ACHIEVEMENTS)
        assert "collect_wood" in gvfs
        assert "collect_diamond" in gvfs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
