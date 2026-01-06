# Crafter Test Suite

## Running Tests

Run all tests:
```bash
python tests/run_all_tests.py
```

Or with unittest discovery:
```bash
python -m unittest discover tests -v
```

Run specific test file:
```bash
python -m unittest tests.test_env
python -m unittest tests.test_recorder
python -m unittest tests.test_gymnasium_compat
```

Run specific test:
```bash
python -m unittest tests.test_env.TestEnv.test_env_reset
```

## Test Coverage

- **test_env.py**: Core environment functionality, reset/step API, episode termination, tutorial mode
- **test_recorder.py**: Recorder wrapper, backward compatibility, stats recording
- **test_gymnasium_compat.py**: Gymnasium integration and API compatibility
- **test_worldgen.py**: World generation and determinism
- **test_examples.py**: Example scripts validation

## Requirements

All tests require the base Crafter dependencies. For full coverage:
```bash
pip install gymnasium pytest
```

## Test Results

Tests validate:
- ✅ Gymnasium API compatibility (5-tuple step, seed parameter)
- ✅ Backward compatibility with old API (4-tuple step)
- ✅ Environment creation and registration
- ✅ Tutorial mode vs normal mode behavior
- ✅ Episode termination conditions
- ✅ Recorder wrapper functionality
- ✅ World generation determinism
- ✅ Example scripts execution
