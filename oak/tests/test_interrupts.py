"""
Unit tests for InterruptManager.
"""

import pytest

from oak.interrupts import (
    InterruptManager,
    VitalStatus,
    InterruptRequest,
)
from oak.constants import (
    CRITICAL_VITAL_THRESHOLD,
    WARNING_VITAL_THRESHOLD,
    MAX_VITAL_VALUE,
    SurvivalPriority,
)


class TestVitalStatus:
    """Tests for VitalStatus dataclass."""

    def test_vital_status_critical(self):
        """Test vital status when critical."""
        status = VitalStatus(
            name="food",
            value=2,
            is_critical=True,
            is_warning=True,
            priority=80,
        )
        assert status.is_critical is True
        assert status.is_warning is True

    def test_vital_status_healthy(self):
        """Test vital status when healthy."""
        status = VitalStatus(
            name="food",
            value=9,
            is_critical=False,
            is_warning=False,
            priority=80,
        )
        assert status.is_critical is False
        assert status.is_warning is False


class TestInterruptRequest:
    """Tests for InterruptRequest dataclass."""

    def test_interrupt_request(self):
        """Test interrupt request creation."""
        request = InterruptRequest(
            vital="food",
            survival_option="eat_cow",
            priority=80,
            urgent=True,
        )
        assert request.vital == "food"
        assert request.survival_option == "eat_cow"
        assert request.urgent is True


class TestInterruptManager:
    """Tests for InterruptManager class."""

    @pytest.fixture
    def manager(self):
        """Create an interrupt manager."""
        return InterruptManager()

    def test_init(self, manager):
        """Test manager initialization."""
        assert manager.critical_threshold == CRITICAL_VITAL_THRESHOLD
        assert manager.warning_threshold == WARNING_VITAL_THRESHOLD
        assert len(manager.survival_subtasks) > 0

    def test_get_vital_status_healthy(self, manager):
        """Test vital status extraction when healthy."""
        info = {
            "inventory": {
                "health": 9,
                "food": 9,
                "drink": 9,
                "energy": 9,
            }
        }

        statuses = manager.get_vital_status(info)

        assert len(statuses) == 4
        assert statuses["health"].value == 9
        assert statuses["health"].is_critical is False
        assert statuses["food"].is_warning is False

    def test_get_vital_status_critical(self, manager):
        """Test vital status when critical."""
        info = {
            "inventory": {
                "health": 2,
                "food": 1,
                "drink": 9,
                "energy": 9,
            }
        }

        statuses = manager.get_vital_status(info)

        assert statuses["health"].is_critical is True
        assert statuses["food"].is_critical is True
        assert statuses["drink"].is_critical is False

    def test_check_interrupt_no_interrupt(self, manager):
        """Test no interrupt when vitals are healthy."""
        info = {
            "inventory": {
                "health": 9,
                "food": 9,
                "drink": 9,
                "energy": 9,
            }
        }

        interrupt = manager.check_interrupt(info)
        assert interrupt is None

    def test_check_interrupt_low_food(self, manager):
        """Test interrupt triggers for low food."""
        info = {
            "inventory": {
                "health": 9,
                "food": 2,  # Below critical threshold
                "drink": 9,
                "energy": 9,
            }
        }

        interrupt = manager.check_interrupt(info)

        assert interrupt is not None
        assert interrupt.vital == "food"
        assert interrupt.survival_option == "eat_cow"
        assert interrupt.urgent is True

    def test_check_interrupt_low_drink(self, manager):
        """Test interrupt triggers for low drink."""
        info = {
            "inventory": {
                "health": 9,
                "food": 9,
                "drink": 2,
                "energy": 9,
            }
        }

        interrupt = manager.check_interrupt(info)

        assert interrupt is not None
        assert interrupt.vital == "drink"
        assert interrupt.survival_option == "collect_drink"

    def test_check_interrupt_priority_order(self, manager):
        """Test that highest priority interrupt is returned."""
        info = {
            "inventory": {
                "health": 9,
                "food": 2,  # Critical
                "drink": 2,  # Also critical
                "energy": 2,  # Also critical
            }
        }

        interrupt = manager.check_interrupt(info)

        # Food has higher priority than drink and energy
        assert interrupt.vital == "food"
        assert interrupt.priority == SurvivalPriority.FOOD.value

    def test_check_combat_interrupt(self, manager):
        """Test combat interrupt when health decreases."""
        current_info = {
            "inventory": {
                "health": 7,
                "food": 9,
                "drink": 9,
                "energy": 9,
            }
        }
        previous_info = {
            "inventory": {
                "health": 9,
                "food": 9,
                "drink": 9,
                "energy": 9,
            }
        }

        interrupt = manager.check_combat_interrupt(current_info, previous_info)

        assert interrupt is not None
        assert interrupt.vital == "health"
        assert interrupt.survival_option == "defeat_zombie"

    def test_check_combat_interrupt_no_change(self, manager):
        """Test no combat interrupt when health unchanged."""
        info = {
            "inventory": {
                "health": 9,
                "food": 9,
                "drink": 9,
                "energy": 9,
            }
        }

        interrupt = manager.check_combat_interrupt(info, info)
        assert interrupt is None

    def test_get_recommended_survival_action(self, manager):
        """Test recommended action for low vitals."""
        info = {
            "inventory": {
                "health": 9,
                "food": 4,  # Warning level
                "drink": 9,
                "energy": 9,
            }
        }

        action = manager.get_recommended_survival_action(info)
        assert action == "eat_cow"

    def test_get_vital_trend(self, manager):
        """Test vital trend detection."""
        info1 = {"inventory": {"health": 9, "food": 9, "drink": 9, "energy": 9}}
        info2 = {"inventory": {"health": 7, "food": 8, "drink": 9, "energy": 9}}

        # First call establishes baseline
        trends1 = manager.get_vital_trend(info1)
        assert all(t == 0 for t in trends1.values())

        # Second call detects changes
        trends2 = manager.get_vital_trend(info2)
        assert trends2["health"] == -2
        assert trends2["food"] == -1
        assert trends2["drink"] == 0

    def test_should_preempt_urgent(self, manager):
        """Test preemption for urgent interrupts."""
        interrupt = InterruptRequest(
            vital="food",
            survival_option="eat_cow",
            priority=80,
            urgent=True,
        )

        # Urgent always preempts
        assert manager.should_preempt("collect_wood", interrupt) is True

    def test_should_preempt_survival_in_progress(self, manager):
        """Test no preemption when already doing survival."""
        interrupt = InterruptRequest(
            vital="drink",
            survival_option="collect_drink",
            priority=70,
            urgent=False,
        )

        # Don't preempt ongoing survival action
        assert manager.should_preempt("eat_cow", interrupt) is False

    def test_reset(self, manager):
        """Test manager reset."""
        # Set some state
        manager._previous_vitals = {"health": 5}

        manager.reset()

        assert manager._previous_vitals is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
