# OaK Framework - How To Guide

## Running Tests

The OaK framework includes comprehensive unit tests for all core modules. Tests are written using `pytest`.

### Prerequisites

Install pytest if not already installed:
```bash
pip install pytest pytest-cov
```

### Running All Tests

From the project root directory:
```bash
cd /home/robomotic/DevOps/github/crafter
python -m pytest oak/tests/ -v
```

Or from the oak directory:
```bash
cd oak
python -m pytest tests/ -v
```

### Running Specific Test Files

Run tests for a specific module:
```bash
# Test the agent
python -m pytest oak/tests/test_agent.py -v

# Test GVFs
python -m pytest oak/tests/test_gvf.py -v

# Test options
python -m pytest oak/tests/test_options.py -v

# Test features
python -m pytest oak/tests/test_features.py -v

# Test tech tree
python -m pytest oak/tests/test_tech_tree.py -v

# Test interrupts
python -m pytest oak/tests/test_interrupts.py -v
```

### Running Tests with Coverage

Generate a coverage report:
```bash
python -m pytest oak/tests/ --cov=oak --cov-report=html --cov-report=term
```

This will create an HTML coverage report in `htmlcov/index.html`.

### Running Specific Test Classes or Methods

```bash
# Run a specific test class
python -m pytest oak/tests/test_agent.py::TestOaKAgentInit -v

# Run a specific test method
python -m pytest oak/tests/test_agent.py::TestOaKAgentInit::test_default_init -v
```

### Quick Syntax Check

Check all files for syntax errors using Pylance (if available):
```bash
# This requires the Pylance MCP server
# Alternatively, use Python's built-in syntax checker:
python -m py_compile oak/*.py
```

Or check imports work:
```bash
python -c "from oak import OaKAgent, FeatureExtractor, TechnologyTree; print('✓ Imports successful')"
```

## Using the OaK Agent

### Basic Usage

```python
import numpy as np
from oak import OaKAgent, ACHIEVEMENTS

# Create the agent
agent = OaKAgent()

# Reset for new episode
agent.reset()

# Create mock observation and info (or use real Crafter environment)
obs = np.zeros((64, 64, 3), dtype=np.uint8)
info = {
    "inventory": {
        "health": 9, "food": 9, "drink": 9, "energy": 9,
        "wood": 0, "stone": 0, "coal": 0, "iron": 0, "diamond": 0,
        "sapling": 0, "wood_pickaxe": 0, "stone_pickaxe": 0, 
        "iron_pickaxe": 0, "wood_sword": 0, "stone_sword": 0, 
        "iron_sword": 0,
    },
    "achievements": {name: 0 for name in ACHIEVEMENTS},
}

# Get action
action = agent.act(obs, info)

# Update agent with experience (not fully implemented yet)
# agent.update(reward, next_obs, next_info, done)

# Get agent statistics
stats = agent.get_stats()
print(f"Steps: {stats['step_count']}, Achievements: {stats['completed_achievements']}")
```

### With Crafter Environment

```python
import crafter
from oak import OaKAgent

# Create environment
env = crafter.Env()

# Create agent
agent = OaKAgent()

# Run episode
obs = env.reset()
agent.reset()

total_reward = 0
done = False

while not done:
    # Get info from environment (Crafter provides this via env.step)
    info = env.step(0)[3]  # Get info dict
    
    # Agent selects action
    action = agent.act(obs, info)
    
    # Step environment
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    # Update agent (when learning is implemented)
    # agent.update(reward, obs, info, done)

print(f"Episode finished with reward: {total_reward}")
```

## Module Overview

### Core Components

- **`OaKAgent`**: Main agent class compatible with Crafter's Gymnasium API
- **`FeatureExtractor`**: Converts Crafter state to 40-dimensional feature vector
- **`TechnologyTree`**: Manages achievement dependencies and provides topological ordering
- **`GeneralValueFunction`**: GVF implementation with 4 components (cumulant, stopping function, stopping value, policy)
- **`Option`**: Temporally extended action policies (Random, Heuristic, Learned)
- **`OptionManager`**: Manages option lifecycle and composition
- **`InterruptManager`**: Priority-based survival option system
- **`Subtask`**: Defines subtasks with termination conditions

### Feature Vector (40 dimensions)

- Binary achievement features (22): One per achievement
- Inventory count features (12): wood, stone, coal, iron, diamond, sapling, pickaxes (3), swords (3)
- Vital status features (4): health, food, drink, energy (normalized to [0,1])
- Placed object features (2): table_placed, furnace_placed

## Known Limitations (NotImplementedError)

The following features are marked as unimplemented:

1. **`PixelFeatureExtractor.extract()`** - CNN encoder for raw pixels
2. **`GeneralValueFunction.update()`** - UWT weight updates
3. **`GeneralValueFunction.stopping_value()`** - without optimistic weights set
4. **`LearnedOption.select_action()`** - learned policy networks
5. **`OptionManager.compose_options()`** - hierarchical option composition
6. **`OaKAgent.rank_features()`** - feature importance ranking
7. **`OaKAgent.off_policy_correction()`** - importance sampling

## Documentation

- **[STRATEGY.md](docs/STRATEGY.md)**: Comprehensive strategy guide for OaK in Crafter
- **[dag.json](docs/dag.json)**: Achievement dependency graph in JSON format
- **[dag.dot](docs/dag.dot)**: Achievement dependency graph in DOT format
- **[achievements.png](docs/achievements.png)**: Visual representation of achievement dependencies

## Development Workflow

1. Make changes to source files in `oak/`
2. Run syntax checks: `python -m py_compile oak/*.py`
3. Run tests: `python -m pytest oak/tests/ -v`
4. Check coverage: `python -m pytest oak/tests/ --cov=oak --cov-report=term`
5. Commit changes

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:
```bash
cd /home/robomotic/DevOps/github/crafter
python -c "from oak import OaKAgent"
```

Or add the project to your Python path:
```bash
export PYTHONPATH="/home/robomotic/DevOps/github/crafter:$PYTHONPATH"
```

### Test Failures

If tests fail, check:
1. All dependencies are installed (`numpy`, `pytest`)
2. You're using the correct Python version
3. The Crafter environment is properly installed (for integration tests)

### Missing Dependencies

Install all required packages:
```bash
pip install numpy pytest pytest-cov
pip install crafter  # For running with actual environment
```
