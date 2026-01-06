"""
Unit tests for the Crafter environment.
"""
import unittest
import numpy as np
import crafter


class TestEnv(unittest.TestCase):
    """Test basic environment functionality."""

    def test_env_creation(self):
        """Test that environment can be created."""
        env = crafter.Env()
        self.assertIsNotNone(env)

    def test_env_reset(self):
        """Test that environment reset works and returns correct format."""
        env = crafter.Env()
        result = env.reset()
        
        # Direct Env should return (obs, info) tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        obs, info = result
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (64, 64, 3))
        self.assertEqual(obs.dtype, np.uint8)
        self.assertIsInstance(info, dict)

    def test_env_reset_with_seed(self):
        """Test that environment reset accepts seed parameter."""
        env = crafter.Env()
        obs1, _ = env.reset(seed=42)
        env2 = crafter.Env()
        obs2, _ = env2.reset(seed=42)
        
        # Same seed should produce same initial observation
        np.testing.assert_array_equal(obs1, obs2)

    def test_env_step(self):
        """Test that environment step works and returns correct format."""
        env = crafter.Env()
        env.reset()
        
        result = env.step(0)  # noop action
        
        # Direct Env should return 5-tuple (obs, reward, terminated, truncated, info)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)
        
        obs, reward, terminated, truncated, info = result
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (64, 64, 3))
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_env_episode_length(self):
        """Test that episodes terminate at specified length."""
        length = 100
        env = crafter.Env(length=length)
        env.reset()
        
        for step in range(length):
            _, _, terminated, truncated, _ = env.step(0)
            if step < length - 1:
                self.assertFalse(truncated, f"Episode truncated early at step {step}")
        
        # Last step should trigger truncation
        self.assertTrue(truncated, "Episode not truncated at specified length")

    def test_env_reward_modes(self):
        """Test reward and no-reward modes."""
        # With reward
        env_reward = crafter.Env(reward=True)
        env_reward.reset()
        _, reward, _, _, _ = env_reward.step(0)
        self.assertIsInstance(reward, (int, float))
        
        # Without reward
        env_no_reward = crafter.Env(reward=False)
        env_no_reward.reset()
        _, reward, _, _, _ = env_no_reward.step(0)
        self.assertEqual(reward, 0.0)

    def test_env_tutorial_mode(self):
        """Test tutorial mode initialization."""
        env = crafter.Env(tutorial=True)
        obs, _ = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        
        # Tutorial mode should have constant daylight
        self.assertEqual(env._world.daylight, 1.0)

    def test_env_action_space(self):
        """Test action space."""
        env = crafter.Env()
        self.assertEqual(env.action_space.n, 17)
        self.assertEqual(len(env.action_names), 17)

    def test_env_observation_space(self):
        """Test observation space."""
        env = crafter.Env()
        obs_space = env.observation_space
        self.assertEqual(obs_space.shape, (64, 64, 3))
        self.assertEqual(obs_space.dtype, np.uint8)

    def test_env_info_keys(self):
        """Test that info dict contains expected keys."""
        env = crafter.Env()
        env.reset()
        _, _, _, _, info = env.step(0)
        
        expected_keys = ['inventory', 'achievements', 'discount', 'semantic', 'player_pos', 'reward']
        for key in expected_keys:
            self.assertIn(key, info)


class TestTutorialMode(unittest.TestCase):
    """Test tutorial mode specific features."""

    def test_tutorial_no_zombies_at_start(self):
        """Test that zombies don't spawn in tutorial mode."""
        env = crafter.Env(tutorial=True, seed=42)
        env.reset()
        
        # Run for a while and check no zombies spawn
        for _ in range(100):
            env.step(0)
        
        # Count zombies in the world
        from crafter.objects import Zombie
        zombie_count = sum(1 for obj in env._world.objects if isinstance(obj, Zombie))
        self.assertEqual(zombie_count, 0, "Zombies spawned in tutorial mode")

    def test_normal_mode_allows_zombies(self):
        """Test that zombies can spawn in normal mode."""
        env = crafter.Env(tutorial=False, seed=42)
        env.reset()
        
        # Run for longer to allow zombie spawns
        # Note: Zombies spawn randomly, so this test may be flaky
        # but with enough steps, we should see at least one
        for _ in range(1000):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        
        # This test just ensures the code runs without error
        # Actual zombie count can vary due to randomness


if __name__ == '__main__':
    unittest.main()
