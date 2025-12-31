# Feature Summary

This document summarizes the major features added to the noise-aware-qnn-identifiability project.

## 1. Command-Line Interface (CLI)

**Location**: `src/cli.py`, `src/__main__.py`

### Features
- Comprehensive argument parsing with `argparse`
- Configurable problem parameters (samples, dimensions, seed)
- Noise presets (minimal, default, extensive, high-noise)
- Custom noise grids
- Optimization controls
- Output management
- Verbosity levels (verbose/quiet)
- Configuration file support

### Usage
```bash
# Basic usage
python -m src --samples 1000 --dimension 16 --seed 42

# Using presets
python -m src --noise-presets minimal

# Custom noise
python -m src --noise-grid "0.0,0.0;0.1,0.1;0.2,0.2"

# Output control
python -m src --output-dir ./results/exp01 --save-results results.json

# Verbosity
python -m src --verbose  # or --quiet
```

### Documentation
- `CLI.md`: Complete CLI documentation
- 27 unit tests in `tests/test_cli.py`

## 2. Experiment Configuration System

**Location**: `src/config.py`

### Features
- YAML and JSON configuration file support
- Dataclass-based configuration objects
- Parameter validation
- Three configuration types:
  1. Single experiments
  2. Batch experiments
  3. Hyperparameter sweeps

### Configuration Classes
- `NoiseConfig`: Individual noise setting
- `ExperimentConfig`: Single experiment configuration
- `SweepConfig`: Hyperparameter sweep definition
- `BatchConfig`: Batch experiment collection

### Usage
```python
from src.config import ExperimentConfig, load_config, save_config

# Programmatic creation
config = ExperimentConfig(
    name="my_exp",
    samples=512,
    dimension=8,
    seed=42
)

# Save/load
save_config(config, "config.yaml")
config = load_config("config.yaml")
```

### CLI Integration
```bash
python -m src --config examples/single_experiment.yaml
```

## 3. Batch Experiment Runner

**Location**: `src/batch_runner.py`

### Features
- Sequential execution
- Parallel execution with `ProcessPoolExecutor`
- Progress tracking
- Error handling
- Automatic result saving
- Batch summaries

### Functions
- `run_single_experiment()`: Execute one experiment
- `run_batch_sequential()`: Sequential batch execution
- `run_batch_parallel()`: Parallel batch execution
- `run_batch()`: Unified batch interface
- `save_batch_summary()`: Save execution summary

### Usage
```python
from src.config import BatchConfig
from src.batch_runner import run_batch

batch = BatchConfig(
    name="my_batch",
    experiments=[...],
    parallel=True,
    max_workers=4
)

summary = run_batch(batch)
```

## 4. Hyperparameter Sweeps

**Location**: Integrated in `src/config.py`

### Features
- Automatic experiment generation
- Cartesian product of parameters
- Flexible parameter specification
- Automatic naming

### Example
```yaml
sweeps:
  - name: dim_seed_sweep
    base_config:
      name: base
      samples: 512
    sweep_params:
      dimension: [4, 8, 16]
      seed: [0, 1, 2, 3, 4]
```

Generates 3 × 5 = 15 experiments automatically.

## 5. Example Configurations

**Location**: `examples/`

### Provided Examples
1. `single_experiment.yaml`: Single experiment template
2. `batch_experiments.yaml`: Multi-experiment batch
3. `hyperparameter_sweep.yaml`: Parameter sweep
4. `minimal_example.yaml`: Quick start example

### Usage
```bash
python -m src --config examples/single_experiment.yaml
python -m src --config examples/batch_experiments.yaml
python -m src --config examples/hyperparameter_sweep.yaml
```

## 6. Comprehensive Documentation

### Documents
1. **CLI.md**: Command-line interface guide
   - All options explained
   - Usage examples
   - Noise presets
   - Batch processing tips

2. **CONFIGURATION.md**: Configuration system guide
   - Configuration types
   - Parameter reference
   - Output structure
   - Programmatic API
   - Advanced examples

3. **FEATURES.md**: This document
   - Feature overview
   - Quick reference

## 7. Unit Tests

**Location**: `tests/`

### Test Suites
1. `test_functions.py`: 25 tests for core functions
   - `feature_map` normalization
   - `qnn_forward` computation
   - `apply_depolarizing` noise
   - `apply_phase_noise` perturbation
   - `finite_diff_hessian_diag` calculation

2. `test_cli.py`: 27 tests for CLI
   - Parser configuration
   - Argument parsing
   - Noise grid parsing
   - Preset validation

### Total: 56 passing tests

## Quick Start Guide

### 1. Simple CLI Usage
```bash
# Run with defaults
python -m src

# Custom experiment
python -m src -n 1000 -d 16 --seed 42 --verbose
```

### 2. Configuration File
```bash
# Create config
cat > my_exp.yaml << EOF
name: test
samples: 256
dimension: 4
seed: 0
noise_grid:
  - p_dep: 0.0
    sigma_phase: 0.0
  - p_dep: 0.1
    sigma_phase: 0.1
EOF

# Run
python -m src --config my_exp.yaml
```

### 3. Batch Experiments
```bash
# Create batch config with multiple experiments
python -m src --config examples/batch_experiments.yaml
```

### 4. Hyperparameter Sweep
```bash
# Run parameter sweep (generates multiple experiments)
python -m src --config examples/hyperparameter_sweep.yaml
```

## Key Benefits

1. **Reproducibility**: All experiments are fully documented in config files
2. **Scalability**: Easy to run large sweeps and batches
3. **Flexibility**: Support for both CLI and configuration files
4. **Parallel Execution**: Speed up independent experiments
5. **Validation**: Automatic parameter validation prevents errors
6. **Organization**: Structured output with metadata
7. **Version Control**: Config files can be tracked in git

## Dependencies

New dependencies added:
- `pyyaml`: YAML file support (added to `pyproject.toml`)

## Future Enhancements

Potential additions:
- Database integration for results
- Web dashboard for visualization
- Cloud execution support
- Experiment comparison tools
- Automatic report generation

---

## Citations

If you use or build on this work, please cite:

> Noise-Aware QNN Identifiability
```bibtex
@software{altman2025noise-aware-qnn-identifiability,
  author  = {Christopher Altman},
  title   = {Noise-Aware QNN Identifiability},
  year    = {2025},
  version = {0.1.0},
  url     = {https://github.com/christopher-altman/noise-aware-qnn-identifiability},
}
```
---

## License

MIT License. See `LICENSE`.

---

## Contact

- **Website:** [christopheraltman.com](https://christopheraltman.com)
- **Research portfolio:** https://lab.christopheraltman.com/
- **Portfolio mirror:** https://christopher-altman.github.io/
- **GitHub:** [github.com/christopher-altman](https://github.com/christopher-altman)
- **Google Scholar:** [scholar.google.com/citations?user=tvwpCcgAAAAJ](https://scholar.google.com/citations?user=tvwpCcgAAAAJ)
- **Email:** x@christopheraltman.com

---

*Christopher Altman (2025)*