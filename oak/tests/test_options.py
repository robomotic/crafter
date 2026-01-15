"""
Unit tests for Options and OptionManager.
"""

import pytest
import numpy as np

from oak.options import (
    Option,
    OptionState,
    OptionResult,
    OptionManager,
    RandomOption,
    HeuristicOption,
    LearnedOption,
    create_options_for_subtasks,
)
from oak.subtasks import Subtask, get_all_subtasks
from oak.constants import NUM_ACTIONS


class TestOptionState:
    """Tests for OptionState enum."""

    def test_states_exist(self):
        """Test all option states are defined."""
        assert OptionState.INACTIVE is not None
        assert OptionState.ACTIVE is not None
        assert OptionState.TERMINATED is not None


class TestRandomOption:
    """Tests for RandomOption class."""

    @pytest.fixture
    def option(self):
        """Create a random option for testing."""
        subtask = Subtask("collect_wood")
        return RandomOption("collect_wood", subtask)

    def test_init(self, option):
        """Test option initialization."""
        assert option.name == "collect_wood"
        assert option.state == OptionState.INACTIVE

    def test_can_initiate(self, option):
        """Test initiation check."""
        info = {"achievements": {}}
        assert option.can_initiate(info) is True

    def test_can_initiate_false_when_completed(self, option):
        """Test initiation fails when already completed."""
        info = {"achievements": {"collect_wood": 1}}
        assert option.can_initiate(info) is False

    def test_select_action_returns_valid(self, option):
        """Test action selection returns valid action."""
        obs = np.zeros((64, 64, 3))
        info = {}
        action = option.select_action(obs, info)
        assert 0 <= action < NUM_ACTIONS

    def test_initiate_sets_active(self, option):
        """Test that initiate sets option to active."""
        option.initiate({})
        assert option.state == OptionState.ACTIVE

    def test_step_returns_result(self, option):
        """Test step returns OptionResult."""
        option.initiate({})
        obs = np.zeros((64, 64, 3))
        info = {"achievements": {}, "inventory": {}}

        result = option.step(obs, info)

        assert isinstance(result, OptionResult)
        assert 0 <= result.action < NUM_ACTIONS
        assert isinstance(result.terminated, bool)

    def test_step_raises_when_inactive(self, option):
        """Test step raises error when not active."""
        obs = np.zeros((64, 64, 3))
        info = {}

        with pytest.raises(RuntimeError):
            option.step(obs, info)

    def test_reset(self, option):
        """Test reset returns to inactive state."""
        option.initiate({})
        assert option.is_active()

        option.reset()
        assert not option.is_active()
        assert option.state == OptionState.INACTIVE


class TestHeuristicOption:
    """Tests for HeuristicOption class."""

    @pytest.fixture
    def option(self):
        """Create a heuristic option with action sequence."""
        subtask = Subtask("collect_wood")
        return HeuristicOption(
            "collect_wood",
            subtask,
            action_sequence=[1, 2, 3, 5],  # left, right, up, do
        )

    def test_follows_action_sequence(self, option):
        """Test that option follows action sequence."""
        option.initiate({})
        obs = np.zeros((64, 64, 3))
        info = {"achievements": {}, "inventory": {}}

        # Step through sequence
        assert option.select_action(obs, info) == 1
        assert option.select_action(obs, info) == 2
        assert option.select_action(obs, info) == 3
        assert option.select_action(obs, info) == 5

    def test_random_after_sequence(self, option):
        """Test random actions after sequence exhausted."""
        option.initiate({})
        obs = np.zeros((64, 64, 3))
        info = {"achievements": {}, "inventory": {}}

        # Exhaust sequence
        for _ in range(4):
            option.select_action(obs, info)

        # Now should be random
        action = option.select_action(obs, info)
        assert 0 <= action < NUM_ACTIONS


class TestLearnedOption:
    """Tests for LearnedOption class."""

    def test_select_action_not_implemented(self):
        """Test that learned option raises NotImplementedError."""
        subtask = Subtask("collect_wood")
        option = LearnedOption("collect_wood", subtask)
        option.initiate({})

        obs = np.zeros((64, 64, 3))
        info = {}

        with pytest.raises(NotImplementedError):
            option.select_action(obs, info)


class TestOptionManager:
    """Tests for OptionManager class."""

    @pytest.fixture
    def manager(self):
        """Create an option manager with some options."""
        manager = OptionManager()
        subtasks = get_all_subtasks()

        for name in ["collect_wood", "place_table", "collect_stone"]:
            option = RandomOption(name, subtasks[name])
            manager.register_option(option)

        return manager

    def test_register_option(self, manager):
        """Test option registration."""
        assert "collect_wood" in manager.options
        assert "place_table" in manager.options

    def test_get_option(self, manager):
        """Test getting option by name."""
        option = manager.get_option("collect_wood")
        assert option is not None
        assert option.name == "collect_wood"

    def test_has_active_option(self, manager):
        """Test checking for active option."""
        assert manager.has_active_option() is False

        option = manager.get_option("collect_wood")
        manager.set_active(option)

        assert manager.has_active_option() is True

    def test_set_active(self, manager):
        """Test setting active option."""
        option = manager.get_option("collect_wood")
        manager.set_active(option)

        assert manager.active_option == option
        assert option.is_active()

    def test_clear_active(self, manager):
        """Test clearing active option."""
        option = manager.get_option("collect_wood")
        manager.set_active(option)
        manager.clear_active()

        assert manager.active_option is None
        assert not option.is_active()

    def test_step(self, manager):
        """Test stepping active option."""
        option = manager.get_option("collect_wood")
        manager.set_active(option)

        obs = np.zeros((64, 64, 3))
        info = {"achievements": {}, "inventory": {}}

        result = manager.step(obs, info)

        assert isinstance(result, OptionResult)
        assert 0 <= result.action < NUM_ACTIONS

    def test_step_raises_without_active(self, manager):
        """Test step raises error without active option."""
        obs = np.zeros((64, 64, 3))
        info = {}

        with pytest.raises(RuntimeError):
            manager.step(obs, info)

    def test_get_available_options(self, manager):
        """Test getting available options."""
        info = {"achievements": {}}
        available = manager.get_available_options(info)

        assert len(available) > 0
        assert all(isinstance(o, Option) for o in available)

    def test_reset(self, manager):
        """Test manager reset."""
        option = manager.get_option("collect_wood")
        manager.set_active(option)
        manager.completed_options.add("collect_wood")

        manager.reset()

        assert manager.active_option is None
        assert len(manager.completed_options) == 0

    def test_compose_options_not_implemented(self, manager):
        """Test option composition raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            manager.compose_options([], "collect_diamond")


class TestCreateOptionsForSubtasks:
    """Tests for option creation utility."""

    def test_create_random_options(self):
        """Test creating random options for all subtasks."""
        subtasks = get_all_subtasks()
        options = create_options_for_subtasks(subtasks, option_type="random")

        assert len(options) == len(subtasks)
        assert all(isinstance(o, RandomOption) for o in options.values())

    def test_create_heuristic_options(self):
        """Test creating heuristic options."""
        subtasks = get_all_subtasks()
        options = create_options_for_subtasks(subtasks, option_type="heuristic")

        assert len(options) == len(subtasks)
        assert all(isinstance(o, HeuristicOption) for o in options.values())

    def test_invalid_option_type(self):
        """Test invalid option type raises error."""
        subtasks = get_all_subtasks()

        with pytest.raises(ValueError):
            create_options_for_subtasks(subtasks, option_type="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
