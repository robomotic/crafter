"""
Unit tests for TechnologyTree.
"""

import pytest
from pathlib import Path

from oak.tech_tree import TechnologyTree
from oak.constants import ACHIEVEMENTS, PREREQUISITES


class TestTechnologyTree:
    """Tests for TechnologyTree class."""

    @pytest.fixture
    def tree(self):
        """Create a technology tree."""
        return TechnologyTree()

    def test_init(self, tree):
        """Test tree initialization."""
        assert len(tree.prerequisites) == len(PREREQUISITES)
        assert len(tree.dependents) > 0
        assert len(tree.root_achievements) > 0

    def test_root_achievements(self, tree):
        """Test root achievements have no prerequisites."""
        for root in tree.root_achievements:
            prereqs = tree.get_prerequisites(root)
            assert len(prereqs) == 0

    def test_get_prerequisites(self, tree):
        """Test getting prerequisites for an achievement."""
        # collect_wood has no prerequisites
        assert len(tree.get_prerequisites("collect_wood")) == 0

        # place_table requires collect_wood
        prereqs = tree.get_prerequisites("place_table")
        assert "collect_wood" in prereqs

        # make_wood_pickaxe requires collect_wood and place_table
        prereqs = tree.get_prerequisites("make_wood_pickaxe")
        assert "collect_wood" in prereqs
        assert "place_table" in prereqs

    def test_get_all_prerequisites(self, tree):
        """Test getting transitive prerequisites."""
        # collect_diamond requires many prerequisites
        all_prereqs = tree.get_all_prerequisites("collect_diamond")

        assert "make_iron_pickaxe" in all_prereqs
        assert "collect_iron" in all_prereqs
        assert "collect_wood" in all_prereqs

    def test_get_dependents(self, tree):
        """Test getting dependents."""
        # collect_wood enables many things
        dependents = tree.get_dependents("collect_wood")

        assert "place_table" in dependents

    def test_are_prerequisites_met(self, tree):
        """Test prerequisite checking."""
        # collect_wood can always be done
        assert tree.are_prerequisites_met("collect_wood", set()) is True

        # place_table needs collect_wood
        assert tree.are_prerequisites_met("place_table", set()) is False
        assert tree.are_prerequisites_met("place_table", {"collect_wood"}) is True

    def test_get_available_subtasks_empty(self, tree):
        """Test available subtasks with no completions."""
        available = tree.get_available_subtasks(set())

        # Should include root achievements
        assert "collect_wood" in available
        assert "collect_sapling" in available

        # Should not include non-root achievements
        assert "collect_stone" not in available

    def test_get_available_subtasks_with_progress(self, tree):
        """Test available subtasks with some completions."""
        completed = {"collect_wood"}
        available = tree.get_available_subtasks(completed)

        # place_table should now be available
        assert "place_table" in available

        # collect_wood should be excluded (already done)
        assert "collect_wood" not in available

    def test_get_next_subtasks(self, tree):
        """Test next subtask recommendations."""
        next_tasks = tree.get_next_subtasks(set())

        assert len(next_tasks) > 0
        # Should prioritize by dependents
        assert "collect_wood" in next_tasks

    def test_topological_order(self, tree):
        """Test topological ordering."""
        order = tree.topological_order()

        assert len(order) == len(ACHIEVEMENTS)

        # collect_wood should come before place_table
        assert order.index("collect_wood") < order.index("place_table")

        # collect_stone should come before make_stone_pickaxe
        assert order.index("collect_stone") < order.index("make_stone_pickaxe")

    def test_get_depth(self, tree):
        """Test depth calculation."""
        # Root achievements have depth 0
        assert tree.get_depth("collect_wood") == 0
        assert tree.get_depth("collect_sapling") == 0

        # place_table has depth 1
        assert tree.get_depth("place_table") == 1

        # make_wood_pickaxe has depth 2
        assert tree.get_depth("make_wood_pickaxe") == 2

    def test_get_critical_path(self, tree):
        """Test critical path calculation."""
        path = tree.get_critical_path_to("collect_diamond")

        assert path[0] == "collect_wood"  # Starts at root
        assert path[-1] == "collect_diamond"  # Ends at target

        # Should include key intermediate steps
        assert "make_iron_pickaxe" in path

    def test_to_dict(self, tree):
        """Test dictionary export."""
        data = tree.to_dict()

        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == len(ACHIEVEMENTS)


class TestTechnologyTreeWithFile:
    """Tests for loading from dag.json file."""

    def test_load_from_nonexistent_file(self):
        """Test fallback when file doesn't exist."""
        tree = TechnologyTree(dag_path=Path("/nonexistent/path.json"))
        # Should use built-in prerequisites
        assert len(tree.prerequisites) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
