"""
Unit tests for the OaKAgent class.
"""

import pytest
import numpy as np

from oak.agent import OaKAgent, AgentState
from oak.constants import NUM_ACTIONS, ACHIEVEMENTS


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = AgentState()
        assert state.step_count == 0
        assert state.completed_achievements == set()
        assert state.current_features is None
        assert state.previous_features is None
        assert state.previous_info is None

    def test_state_with_values(self):
        """Test state with provided values."""
        features = np.zeros(40)
        state = AgentState(
            step_count=10,
            completed_achievements={"collect_wood"},
            current_features=features,
        )
        assert state.step_count == 10
        assert "collect_wood" in state.completed_achievements
        assert state.current_features is not None


class TestOaKAgentInit:
    """Tests for OaKAgent initialization."""

    def test_default_init(self):
        """Test agent initializes with default parameters."""
        agent = OaKAgent()
        assert agent.feature_extractor is not None
        assert agent.tech_tree is not None
        assert agent.option_manager is not None
        assert agent.interrupt_manager is not None
        assert len(agent.gvfs) == len(ACHIEVEMENTS)

    def test_init_with_params(self):
        """Test agent initializes with custom parameters."""
        agent = OaKAgent(
            use_privileged_features=True,
            option_type="random",
            enable_interrupts=True,
            learning_rate=0.05,
            discount=0.95,
        )
        assert agent.learning_rate == 0.05
        assert agent.discount == 0.95
        assert agent.enable_interrupts is True

    def test_init_pixel_features_not_implemented(self):
        """Test that pixel features raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            OaKAgent(use_privileged_features=False)


class TestOaKAgentReset:
    """Tests for agent reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears agent state."""
        agent = OaKAgent()
        agent.state.step_count = 100
        agent.state.completed_achievements.add("collect_wood")

        agent.reset()

        assert agent.state.step_count == 0
        assert len(agent.state.completed_achievements) == 0


class TestOaKAgentAct:
    """Tests for agent action selection."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return OaKAgent()

    @pytest.fixture
    def mock_obs(self):
        """Create a mock observation."""
        return np.zeros((64, 64, 3), dtype=np.uint8)

    @pytest.fixture
    def mock_info(self):
        """Create a mock info dictionary."""
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

    def test_act_returns_valid_action(self, agent, mock_obs, mock_info):
        """Test that act returns a valid action index."""
        agent.reset()
        action = agent.act(mock_obs, mock_info)
        assert 0 <= action < NUM_ACTIONS

    def test_act_increments_step_count(self, agent, mock_obs, mock_info):
        """Test that act increments the step count."""
        agent.reset()
        assert agent.state.step_count == 0

        agent.act(mock_obs, mock_info)
        assert agent.state.step_count == 1

    def test_act_updates_features(self, agent, mock_obs, mock_info):
        """Test that act updates feature vectors."""
        agent.reset()
        assert agent.state.current_features is None

        agent.act(mock_obs, mock_info)
        assert agent.state.current_features is not None
        assert len(agent.state.current_features) == 40


class TestOaKAgentInterrupts:
    """Tests for interrupt handling."""

    @pytest.fixture
    def agent(self):
        """Create agent with interrupts enabled."""
        return OaKAgent(enable_interrupts=True)

    def test_low_food_triggers_interrupt(self, agent):
        """Test that low food triggers survival interrupt."""
        agent.reset()
        obs = np.zeros((64, 64, 3), dtype=np.uint8)
        info = {
            "inventory": {
                "health": 9,
                "food": 2,  # Below critical threshold
                "drink": 9,
                "energy": 9,
                "wood": 0,
            },
            "achievements": {},
        }

        interrupt = agent._check_interrupts(info)
        assert interrupt is not None
        assert interrupt.vital == "food"

    def test_no_interrupt_when_vitals_high(self, agent):
        """Test no interrupt when all vitals are high."""
        agent.reset()
        info = {
            "inventory": {
                "health": 9,
                "food": 9,
                "drink": 9,
                "energy": 9,
            },
            "achievements": {},
        }

        interrupt = agent._check_interrupts(info)
        assert interrupt is None


class TestOaKAgentNotImplemented:
    """Tests for NotImplementedError on unimplemented features."""

    def test_rank_features_not_implemented(self):
        """Test that rank_features raises NotImplementedError."""
        agent = OaKAgent()
        with pytest.raises(NotImplementedError):
            agent.rank_features()

    def test_off_policy_correction_not_implemented(self):
        """Test that off_policy_correction raises NotImplementedError."""
        agent = OaKAgent()
        with pytest.raises(NotImplementedError):
            agent.off_policy_correction(None, None)

    def test_adaptive_weight_selection_not_implemented(self):
        """Test that adaptive weight selection raises NotImplementedError."""
        agent = OaKAgent()
        with pytest.raises(NotImplementedError):
            agent.select_optimistic_weights(method="adaptive")


class TestOaKAgentStats:
    """Tests for agent statistics."""

    def test_get_stats(self):
        """Test getting agent statistics."""
        agent = OaKAgent()
        agent.reset()

        stats = agent.get_stats()

        assert "step_count" in stats
        assert "completed_achievements" in stats
        assert "active_option" in stats
        assert stats["step_count"] == 0
        assert stats["completed_achievements"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
