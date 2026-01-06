"""
Unit tests for world generation.
"""
import unittest
import numpy as np
import crafter
from crafter import objects


class TestWorldGen(unittest.TestCase):
    """Test world generation."""

    def test_world_generation(self):
        """Test that world generates without errors."""
        env = crafter.Env(seed=42)
        env.reset()
        self.assertIsNotNone(env._world)
        self.assertGreater(len(env._world.objects), 0)

    def test_player_spawn(self):
        """Test that player spawns in the world."""
        env = crafter.Env()
        env.reset()
        
        self.assertIsNotNone(env._player)
        self.assertIn(env._player, env._world.objects)

    def test_deterministic_generation(self):
        """Test that same seed produces same world."""
        env1 = crafter.Env(seed=12345)
        obs1, _ = env1.reset()
        
        env2 = crafter.Env(seed=12345)
        obs2, _ = env2.reset()
        
        # Same seed should produce identical initial observations
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds_differ(self):
        """Test that different seeds produce different worlds."""
        env1 = crafter.Env(seed=111)
        obs1, _ = env1.reset()
        
        env2 = crafter.Env(seed=222)
        obs2, _ = env2.reset()
        
        # Different seeds should produce different observations
        self.assertFalse(np.array_equal(obs1, obs2))


if __name__ == '__main__':
    unittest.main()
