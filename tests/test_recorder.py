"""
Unit tests for the Recorder wrapper.
"""

import unittest
import tempfile
import pathlib
import json
import numpy as np
import crafter


class TestRecorder(unittest.TestCase):
    """Test recorder functionality."""

    def test_recorder_creation(self):
        """Test that recorder can be created."""
        env = crafter.Env()
        with tempfile.TemporaryDirectory() as tmpdir:
            env = crafter.Recorder(env, tmpdir)
            self.assertIsNotNone(env)

    def test_recorder_backward_compat_reset(self):
        """Test that recorder returns old API format for reset."""
        env = crafter.Env()
        with tempfile.TemporaryDirectory() as tmpdir:
            env = crafter.Recorder(
                env, tmpdir, save_stats=False, save_video=False, save_episode=False
            )
            result = env.reset()

            # Recorder should return just obs for backward compatibility
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (64, 64, 3))

    def test_recorder_backward_compat_step(self):
        """Test that recorder returns old API format for step."""
        env = crafter.Env()
        with tempfile.TemporaryDirectory() as tmpdir:
            env = crafter.Recorder(
                env, tmpdir, save_stats=False, save_video=False, save_episode=False
            )
            env.reset()
            result = env.step(0)

            # Recorder should return 4-tuple for backward compatibility
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 4)

            obs, reward, done, info = result
            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)

    def test_stats_recorder_saves_stats(self):
        """Test that stats are saved correctly."""
        env = crafter.Env(length=50)
        with tempfile.TemporaryDirectory() as tmpdir:
            env = crafter.Recorder(
                env, tmpdir, save_stats=True, save_video=False, save_episode=False
            )
            env.reset()

            # Run until episode ends
            done = False
            while not done:
                _, _, done, _ = env.step(0)

            # Check stats file exists
            stats_file = pathlib.Path(tmpdir) / "stats.jsonl"
            self.assertTrue(stats_file.exists())

            # Read and validate stats
            with open(stats_file, "r") as f:
                stats = json.loads(f.readline())

            self.assertIn("length", stats)
            self.assertIn("reward", stats)
            self.assertEqual(stats["length"], 50)

    def test_recorder_no_directory(self):
        """Test recorder with no directory (disabled recording)."""
        env = crafter.Env()
        env = crafter.Recorder(env, None)
        obs = env.reset()
        self.assertIsInstance(obs, np.ndarray)

        _, _, done, _ = env.step(0)
        self.assertIsInstance(done, bool)


if __name__ == "__main__":
    unittest.main()
