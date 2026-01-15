"""
Technology Tree management for Crafter achievements.

This module provides the TechnologyTree class that manages the directed
acyclic graph (DAG) of achievement dependencies, enabling queries for
available subtasks and prerequisite checking.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from oak.constants import ACHIEVEMENTS, PREREQUISITES


class TechnologyTree:
    """
    Manages the technology tree (DAG) of Crafter achievements.

    The technology tree defines which achievements depend on others,
    enabling the agent to determine which subtasks are currently available
    based on completed prerequisites.

    Attributes:
        prerequisites: Dict mapping each achievement to its required prerequisites.
        dependents: Dict mapping each achievement to achievements that depend on it.
        root_achievements: Set of achievements with no prerequisites.
    """

    def __init__(self, dag_path: Optional[Path] = None) -> None:
        """
        Initialize the technology tree.

        Args:
            dag_path: Optional path to dag.json file. If None, uses built-in
                     prerequisites from constants.py.
        """
        if dag_path is not None and dag_path.exists():
            self._load_from_json(dag_path)
        else:
            self.prerequisites = {k: v.copy() for k, v in PREREQUISITES.items()}

        # Build reverse mapping (what depends on each achievement)
        self.dependents: Dict[str, Set[str]] = {a: set() for a in ACHIEVEMENTS}
        for achievement, prereqs in self.prerequisites.items():
            for prereq in prereqs:
                if prereq in self.dependents:
                    self.dependents[prereq].add(achievement)

        # Identify root achievements (no prerequisites)
        self.root_achievements: Set[str] = {
            a for a, prereqs in self.prerequisites.items() if len(prereqs) == 0
        }

        # Cache for topological order
        self._topo_order: Optional[List[str]] = None

    def _load_from_json(self, dag_path: Path) -> None:
        """
        Load prerequisites from dag.json file.

        Args:
            dag_path: Path to the JSON file containing edges.
        """
        with open(dag_path, "r") as f:
            data = json.load(f)

        # Initialize all achievements with empty prerequisites
        self.prerequisites = {a: set() for a in ACHIEVEMENTS}

        # Parse edges from JSON
        for edge in data.get("edges", []):
            from_node = edge["from"].lower().replace(" ", "_")
            to_node = edge["to"].lower().replace(" ", "_")
            if to_node in self.prerequisites:
                self.prerequisites[to_node].add(from_node)

    def get_prerequisites(self, achievement: str) -> Set[str]:
        """
        Get the direct prerequisites for an achievement.

        Args:
            achievement: Name of the achievement.

        Returns:
            Set of achievement names that are direct prerequisites.
        """
        return self.prerequisites.get(achievement, set())

    def get_all_prerequisites(self, achievement: str) -> Set[str]:
        """
        Get all prerequisites (transitive closure) for an achievement.

        Args:
            achievement: Name of the achievement.

        Returns:
            Set of all achievements that must be completed first.
        """
        all_prereqs: Set[str] = set()
        stack = list(self.get_prerequisites(achievement))

        while stack:
            prereq = stack.pop()
            if prereq not in all_prereqs:
                all_prereqs.add(prereq)
                stack.extend(self.get_prerequisites(prereq))

        return all_prereqs

    def get_dependents(self, achievement: str) -> Set[str]:
        """
        Get achievements that directly depend on this one.

        Args:
            achievement: Name of the achievement.

        Returns:
            Set of achievement names that require this as a prerequisite.
        """
        return self.dependents.get(achievement, set())

    def are_prerequisites_met(
        self, achievement: str, completed: Set[str]
    ) -> bool:
        """
        Check if all prerequisites for an achievement are completed.

        Args:
            achievement: Name of the achievement to check.
            completed: Set of already completed achievement names.

        Returns:
            True if all prerequisites are in the completed set.
        """
        prereqs = self.get_prerequisites(achievement)
        return prereqs.issubset(completed)

    def get_available_subtasks(
        self, completed: Set[str], exclude_completed: bool = True
    ) -> List[str]:
        """
        Get subtasks that are currently available to pursue.

        An achievement is available if:
        1. All its prerequisites are completed
        2. It has not been completed yet (if exclude_completed=True)

        Args:
            completed: Set of completed achievement names.
            exclude_completed: Whether to exclude already completed achievements.

        Returns:
            List of available achievement names, ordered by topological sort.
        """
        available = []
        for achievement in self.topological_order():
            if exclude_completed and achievement in completed:
                continue
            if self.are_prerequisites_met(achievement, completed):
                available.append(achievement)

        return available

    def get_next_subtasks(self, completed: Set[str]) -> List[str]:
        """
        Get the next recommended subtasks based on tech tree progression.

        Prioritizes achievements that unlock the most dependents.

        Args:
            completed: Set of completed achievement names.

        Returns:
            List of recommended next achievements, sorted by priority.
        """
        available = self.get_available_subtasks(completed)

        # Score each available subtask by number of unlockable dependents
        def score(achievement: str) -> int:
            return len(self.get_dependents(achievement))

        return sorted(available, key=score, reverse=True)

    def topological_order(self) -> List[str]:
        """
        Get achievements in topological order (prerequisites before dependents).

        Returns:
            List of achievement names in topological order.
        """
        if self._topo_order is not None:
            return self._topo_order

        # Kahn's algorithm
        in_degree = {a: len(self.prerequisites.get(a, set())) for a in ACHIEVEMENTS}
        queue = [a for a, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for dependent in self.dependents.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        self._topo_order = result
        return result

    def get_depth(self, achievement: str) -> int:
        """
        Get the depth of an achievement in the technology tree.

        Depth is the length of the longest path from any root to this achievement.

        Args:
            achievement: Name of the achievement.

        Returns:
            Depth in the tree (0 for root achievements).
        """
        if achievement in self.root_achievements:
            return 0

        prereqs = self.get_prerequisites(achievement)
        if not prereqs:
            return 0

        return 1 + max(self.get_depth(p) for p in prereqs)

    def get_critical_path_to(self, achievement: str) -> List[str]:
        """
        Get the critical (longest) path to an achievement.

        Args:
            achievement: Target achievement name.

        Returns:
            List of achievements in order from root to target.
        """
        if achievement in self.root_achievements:
            return [achievement]

        prereqs = self.get_prerequisites(achievement)
        if not prereqs:
            return [achievement]

        # Find the prerequisite with the longest path
        longest_path: List[str] = []
        for prereq in prereqs:
            path = self.get_critical_path_to(prereq)
            if len(path) > len(longest_path):
                longest_path = path

        return longest_path + [achievement]

    def to_dict(self) -> Dict[str, Any]:
        """
        Export the technology tree as a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "nodes": list(ACHIEVEMENTS),
            "edges": [
                {"from": prereq, "to": achievement}
                for achievement, prereqs in self.prerequisites.items()
                for prereq in prereqs
            ],
        }
