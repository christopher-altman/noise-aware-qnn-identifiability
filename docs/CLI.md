# Command-Line Interface Documentation

The `noise-aware-qnn-identifiability` project includes a comprehensive command-line interface for running experiments with configurable parameters.

## Quick Start

```bash
# Run with default settings
python -m src

# View all options
python -m src --help
```

## Usage

```bash
python -m src [OPTIONS]
```

## Options

### Problem Configuration

- `-n, --samples N`: Number of training samples (default: 512)
- `-d, --dimension D`: Parameter dimension (default: 8)
- `-s, --seed SEED`: Random seed for reproducibility (default: 0)

### Noise Configuration

- `--noise-grid GRID`: Custom noise grid as "p1,s1;p2,s2;..." where:
  - `p` = depolarizing noise probability [0, 1]
  - `s` = phase noise sigma (Gaussian scale)
  - Example: `"0.0,0.0;0.05,0.0;0.1,0.1"`

- `--noise-presets {minimal,default,extensive,high-noise}`: Use predefined noise grid preset
  - **minimal**: 2 noise settings for quick tests
  - **default**: 5 noise settings (recommended)
  - **extensive**: 9 noise settings for detailed analysis
  - **high-noise**: 5 settings focused on high noise regimes

### Optimization Configuration

- `--optimizer-iterations ITERS`: Number of random search iterations per noise setting (default: 2000)
- `--hessian-eps EPS`: Finite difference epsilon for Hessian computation (default: 1e-3)

### Output Configuration

- `-o, --output-dir DIR`: Output directory for figures and results (default: current directory)
- `--no-plots`: Disable plot generation
- `--save-results FILE`: Save results to JSON file (e.g., results.json)

### Verbosity

- `-v, --verbose`: Enable verbose output with detailed progress
- `-q, --quiet`: Suppress all non-essential output

## Examples

### Basic Usage

```bash
# Run with default settings
python -m src
```

### Custom Problem Size

```bash
# Larger problem with 1000 samples and 16 dimensions
python -m src --samples 1000 --dimension 16 --seed 42
```

### Custom Noise Grid

```bash
# Test specific noise combinations
python -m src --noise-grid "0.0,0.0;0.1,0.0;0.2,0.1;0.3,0.3"
```

### Using Noise Presets

```bash
# Quick test with minimal noise settings
python -m src --noise-presets minimal

# Comprehensive sweep with extensive preset
python -m src --noise-presets extensive
```

### Output Control

```bash
# Save to specific directory with JSON results
python -m src --output-dir ./results/exp01 --save-results results.json

# Run without generating plots (faster for batch processing)
python -m src --no-plots --quiet
```

### Optimization Tuning

```bash
# Increase optimizer iterations for better convergence
python -m src --optimizer-iterations 5000

# Use different Hessian epsilon for finite differences
python -m src --hessian-eps 1e-4
```

### Verbose and Quiet Modes

```bash
# See detailed progress during execution
python -m src --verbose

# Minimal output (useful for batch scripts)
python -m src --quiet
```

## Noise Preset Details

### Minimal
```python
[(0.0, 0.0), (0.1, 0.0)]
```
2 settings: baseline and moderate depolarizing noise

### Default
```python
[(0.0, 0.0), (0.05, 0.0), (0.10, 0.0), (0.10, 0.10), (0.20, 0.20)]
```
5 settings: gradual noise increase with combined noise types

### Extensive
```python
[
    (0.0, 0.0), (0.05, 0.0), (0.10, 0.0), (0.15, 0.0),
    (0.10, 0.05), (0.10, 0.10), (0.15, 0.15),
    (0.20, 0.20), (0.25, 0.25)
]
```
9 settings: comprehensive sweep of noise parameter space

### High-Noise
```python
[(0.0, 0.0), (0.20, 0.0), (0.30, 0.0), (0.30, 0.30), (0.40, 0.40)]
```
5 settings: focus on regime where identifiability collapse is expected

## Output Files

When plots are enabled, the following files are generated:

- `fig_accuracy_vs_identifiability.png`: Shows relationship between task performance and identifiability
- `fig_param_error_vs_noise.png`: Shows parameter recovery error vs. noise level

When `--save-results` is specified, a JSON file is created with detailed metrics:

```json
[
  {
    "p_dep": 0.0,
    "sigma_phase": 0.0,
    "acc": 0.8867,
    "param_l2": 0.3947,
    "ident_proxy": 0.0
  },
  ...
]
```

## Batch Processing Example

```bash
# Script to run multiple experiments with different seeds
for seed in 0 1 2 3 4; do
  python -m src \
    --seed $seed \
    --output-dir ./results/seed_$seed \
    --save-results results.json \
    --quiet
done
```

## Exit Codes

- `0`: Success
- `1`: Error (invalid parameters, runtime error, etc.)

## Tips

1. **Quick tests**: Use `--noise-presets minimal` with `--samples 256` for fast iterations
2. **Reproducibility**: Always specify `--seed` for reproducible results
3. **Large experiments**: Use `--verbose` to monitor progress on long runs
4. **Batch processing**: Use `--quiet --no-plots` for headless execution
5. **Data analysis**: Always use `--save-results` to preserve numerical results for post-processing
