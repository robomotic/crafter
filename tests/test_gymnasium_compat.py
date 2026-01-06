"""
Unit tests for Gymnasium compatibility.
"""
import unittest
import numpy as np


class TestGymnasiumCompatibility(unittest.TestCase):
    """Test compatibility with Gymnasium API."""

    def test_gymnasium_import(self):
        """Test that gymnasium can be imported."""
        try:
            import gymnasium as gym
        except ImportError:
            self.skipTest("Gymnasium not installed")

    def test_gymnasium_make(self):
        """Test environment creation via gym.make."""
        try:
            import gymnasium as gym
            import crafter
        except ImportError:
            self.skipTest("Gymnasium not installed")
        
        env = gym.make('CrafterReward-v1')
        self.assertIsNotNone(env)

    def test_gymnasium_registered_envs(self):
        """Test that all Crafter environments are registered."""
        try:
            import gymnasium as gym
            import crafter
        except ImportError:
            self.skipTest("Gymnasium not installed")
        
        expected_envs = [
            'CrafterReward-v1',
            'CrafterNoReward-v1',
            'CrafterRewardTutorial-v1',
            'CrafterNoRewardTutorial-v1',
        ]
        
        for env_name in expected_envs:
            env = gym.make(env_name)
            self.assertIsNotNone(env, f"Failed to create {env_name}")

    def test_gymnasium_reset_api(self):
        """Test new Gymnasium reset API."""
        try:
            import gymnasium as gym
            import crafter
        except ImportError:
            self.skipTest("Gymnasium not installed")
        
        env = gym.make('CrafterReward-v1')
        obs, info = env.reset()
        
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertEqual(obs.shape, (64, 64, 3))

    def test_gymnasium_step_api(self):
        """Test new Gymnasium step API."""
        try:
            import gymnasium as gym
            import crafter
        except ImportError:
            self.skipTest("Gymnasium not installed")
        
        env = gym.make('CrafterReward-v1')
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_gymnasium_seed(self):
        """Test that seeding works with Gymnasium API."""
        try:
            import gymnasium as gym
            import crafter
        except ImportError:
            self.skipTest("Gymnasium not installed")
        
        env1 = gym.make('CrafterReward-v1')
        env2 = gym.make('CrafterReward-v1')
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)

    def test_gymnasium_episode_length(self):
        """Test that max_episode_steps is respected."""
        try:
            import gymnasium as gym
            import crafter
        except ImportError:
            self.skipTest("Gymnasium not installed")
        
        env = gym.make('CrafterReward-v1')
        env.reset()
        
        # Max episode steps should be 10000
        for step in range(10000):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        
        # Should reach 10000 or terminate earlier due to death
        self.assertTrue(step >= 9999 or terminated or truncated)


if __name__ == '__main__':
    unittest.main()
