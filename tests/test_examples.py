"""
Tests to verify example scripts work correctly.
"""
import unittest
import subprocess
import sys
import pathlib
import tempfile


class TestExamples(unittest.TestCase):
    """Test that example scripts run without errors."""

    def test_run_random_example(self):
        """Test that run_random.py example works."""
        examples_dir = pathlib.Path(__file__).parent.parent / 'examples'
        script = examples_dir / 'run_random.py'
        
        if not script.exists():
            self.skipTest(f"Example script not found: {script}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, str(script), '--outdir', tmpdir, '--steps', '100'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            self.assertEqual(result.returncode, 0, 
                           f"run_random.py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    def test_run_random_module(self):
        """Test that crafter.run_random module works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, '-m', 'crafter.run_random', '--record', tmpdir, '--episodes', '1', '--length', '100'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            self.assertEqual(result.returncode, 0, 
                           f"crafter.run_random failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    def test_run_gui_help(self):
        """Test that run_gui shows help without errors."""
        result = subprocess.run(
            [sys.executable, '-m', 'crafter.run_gui', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('--fps', result.stdout)

    def test_run_terrain_help(self):
        """Test that run_terrain shows help without errors."""
        result = subprocess.run(
            [sys.executable, '-m', 'crafter.run_terrain', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        self.assertEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
