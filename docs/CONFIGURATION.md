# Experiment Configuration System

The project includes a comprehensive configuration system for defining and running experiments using YAML or JSON files.

## Quick Start

```bash
# Run experiment from config file
python -m src --config examples/single_experiment.yaml

# Run batch experiments
python -m src --config examples/batch_experiments.yaml

# Run hyperparameter sweep
python -m src --config examples/hyperparameter_sweep.yaml
```

## Configuration Types

### 1. Single Experiment

Define a single experiment with all parameters.

**Example (`examples/single_experiment.yaml`):**

```yaml
name: example_experiment
samples: 512
dimension: 8
seed: 42
optimizer_iterations: 2000
hessian_eps: 0.001
output_dir: ./results/example
generate_plots: true
save_results: true
verbose: false

noise_grid:
  - p_dep: 0.0
    sigma_phase: 0.0
  - p_dep: 0.10
    sigma_phase: 0.0
  - p_dep: 0.20
    sigma_phase: 0.20
```

### 2. Batch Experiments

Run multiple independent experiments sequentially or in parallel.

**Example (`examples/batch_experiments.yaml`):**

```yaml
name: multi_seed_comparison
parallel: false  # Set to true for parallel execution
max_workers: 4   # Number of parallel workers

experiments:
  - name: seed_0
    samples: 512
    dimension: 8
    seed: 0
    noise_grid:
      - p_dep: 0.0
        sigma_phase: 0.0
      - p_dep: 0.10
        sigma_phase: 0.10
    output_dir: ./results/batch
    generate_plots: true
    save_results: true
  
  - name: seed_1
    samples: 512
    dimension: 8
    seed: 1
    noise_grid:
      - p_dep: 0.0
        sigma_phase: 0.0
      - p_dep: 0.10
        sigma_phase: 0.10
    output_dir: ./results/batch
    generate_plots: true
    save_results: true
```

### 3. Hyperparameter Sweeps

Automatically generate experiments by sweeping over parameter combinations.

**Example (`examples/hyperparameter_sweep.yaml`):**

```yaml
name: dimension_seed_sweep
parallel: true
max_workers: 4

sweeps:
  - name: dim_seed_sweep
    base_config:
      name: base
      samples: 512
      optimizer_iterations: 1000
      output_dir: ./results/sweep
      generate_plots: true
      save_results: true
      noise_grid:
        - p_dep: 0.0
          sigma_phase: 0.0
        - p_dep: 0.20
          sigma_phase: 0.20
    sweep_params:
      dimension: [4, 8, 16]      # Sweep over dimensions
      seed: [0, 1, 2, 3, 4]      # Sweep over seeds
```

This will generate 3 × 5 = 15 experiments automatically.

## Configuration Parameters

### Experiment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | required | Unique experiment name |
| `samples` | int | 512 | Number of training samples |
| `dimension` | int | 8 | Parameter dimension |
| `seed` | int | 0 | Random seed for reproducibility |
| `optimizer_iterations` | int | 2000 | Optimizer iterations per noise setting |
| `hessian_eps` | float | 0.001 | Finite difference epsilon for Hessian |
| `output_dir` | string | "./results" | Output directory for results |
| `generate_plots` | bool | true | Generate visualization plots |
| `save_results` | bool | true | Save results to JSON |
| `verbose` | bool | false | Enable verbose output |

### Noise Grid

Define noise settings as a list of configurations:

```yaml
noise_grid:
  - p_dep: 0.0        # Depolarizing probability [0, 1]
    sigma_phase: 0.0  # Phase noise sigma (>= 0)
  - p_dep: 0.05
    sigma_phase: 0.0
  - p_dep: 0.10
    sigma_phase: 0.10
```

### Batch Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | required | Batch name |
| `parallel` | bool | false | Run experiments in parallel |
| `max_workers` | int | 4 | Maximum parallel workers |
| `experiments` | list | [] | List of experiment configurations |
| `sweeps` | list | [] | List of sweep configurations |

### Sweep Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Sweep name prefix |
| `base_config` | object | Base experiment configuration |
| `sweep_params` | dict | Parameters to sweep (param: [values]) |

## Output Structure

### Single Experiment

```
output_dir/
└── experiment_name/
    ├── config.json                         # Experiment configuration
    ├── results.json                        # Numerical results
    ├── fig_accuracy_vs_identifiability.png # Visualization
    └── fig_param_error_vs_noise.png        # Visualization
```

### Batch Experiments

```
output_dir/
├── experiment_1/
│   ├── config.json
│   ├── results.json
│   └── *.png
├── experiment_2/
│   ├── config.json
│   ├── results.json
│   └── *.png
└── batch_summary.json  # Summary of all experiments
```

## JSON Format

Configurations can also be written in JSON:

```json
{
  "name": "example_experiment",
  "samples": 512,
  "dimension": 8,
  "seed": 42,
  "noise_grid": [
    {"p_dep": 0.0, "sigma_phase": 0.0},
    {"p_dep": 0.1, "sigma_phase": 0.0}
  ],
  "optimizer_iterations": 2000,
  "output_dir": "./results/example",
  "generate_plots": true,
  "save_results": true
}
```

## Programmatic Usage

### Python API

```python
from src.config import ExperimentConfig, BatchConfig, load_config, save_config
from src.batch_runner import run_batch, run_single_experiment

# Create config programmatically
config = ExperimentConfig(
    name="my_experiment",
    samples=512,
    dimension=8,
    seed=42,
    noise_grid=[
        NoiseConfig(0.0, 0.0),
        NoiseConfig(0.1, 0.0),
    ]
)

# Save config
save_config(config, "my_config.yaml")

# Load and run
config = load_config("my_config.yaml")
result = run_single_experiment(config)
```

### Create Sweeps Programmatically

```python
from src.config import ExperimentConfig, SweepConfig, BatchConfig

base = ExperimentConfig(name="base", samples=512, dimension=8)

sweep = SweepConfig(
    name="my_sweep",
    base_config=base,
    sweep_params={
        "seed": [0, 1, 2],
        "dimension": [4, 8, 16]
    }
)

# Generate all combinations
experiments = sweep.generate_experiments()
print(f"Generated {len(experiments)} experiments")
```

## Advanced Examples

### Mixed Batch and Sweep

```yaml
name: comprehensive_study
parallel: true
max_workers: 8

experiments:
  # Single baseline experiment
  - name: baseline
    samples: 1000
    dimension: 16
    seed: 0
    noise_grid:
      - p_dep: 0.0
        sigma_phase: 0.0

sweeps:
  # Sweep over noise levels
  - name: noise_sweep
    base_config:
      name: noise_base
      samples: 512
      dimension: 8
      seed: 42
    sweep_params:
      noise_grid:
        - [{"p_dep": 0.0, "sigma_phase": 0.0}, {"p_dep": 0.05, "sigma_phase": 0.0}]
        - [{"p_dep": 0.0, "sigma_phase": 0.0}, {"p_dep": 0.10, "sigma_phase": 0.10}]
        - [{"p_dep": 0.0, "sigma_phase": 0.0}, {"p_dep": 0.20, "sigma_phase": 0.20}]
```

### Optimizer Iterations Sweep

```yaml
name: convergence_study
parallel: false

sweeps:
  - name: optimizer_sweep
    base_config:
      name: base
      samples: 256
      dimension: 4
      seed: 0
      noise_grid:
        - p_dep: 0.0
          sigma_phase: 0.0
        - p_dep: 0.10
          sigma_phase: 0.10
    sweep_params:
      optimizer_iterations: [500, 1000, 2000, 5000]
```

## Tips and Best Practices

1. **Use descriptive names**: Choose experiment names that reflect the configuration
2. **Start small**: Test with minimal settings before large sweeps
3. **Parallel execution**: Enable `parallel: true` for independent experiments
4. **Save intermediate results**: Always set `save_results: true` for long runs
5. **Version control**: Keep configuration files in version control
6. **Documentation**: Add comments in YAML (use `# comment`) to document choices

## Validation

The configuration system validates all parameters:

- Sample counts and dimensions must be positive
- Noise probabilities must be in [0, 1]
- Phase noise sigma must be non-negative
- Required fields must be present

Errors are reported with clear messages if validation fails.

## Example Workflow

```bash
# 1. Create configuration
cat > my_experiment.yaml << EOF
name: quick_test
samples: 256
dimension: 4
seed: 0
noise_grid:
  - p_dep: 0.0
    sigma_phase: 0.0
  - p_dep: 0.1
    sigma_phase: 0.0
output_dir: ./results/test
EOF

# 2. Run experiment
python -m src --config my_experiment.yaml

# 3. Check results
cat results/test/quick_test/results.json

# 4. View plots
open results/test/quick_test/*.png
```
