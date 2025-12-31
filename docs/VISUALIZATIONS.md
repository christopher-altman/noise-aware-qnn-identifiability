# Extended Visualization System

The project includes a comprehensive visualization system with static (matplotlib) and interactive (plotly) visualizations.

## Quick Start

```bash
# Basic visualizations (default)
python -m src --samples 512 --dimension 8

# Extended visualizations (heatmaps, combined metrics)
python -m src --samples 512 --dimension 8 --extended-viz

# Interactive visualizations (HTML dashboards)
python -m src --samples 512 --dimension 8 --interactive

# All visualizations
python -m src --samples 512 --dimension 8 --extended-viz --interactive
```

## Visualization Types

### 1. Standard Visualizations (Default)

Always generated when plots are enabled:

#### `assets/figures/fig_accuracy_vs_identifiability.png`
- **Purpose**: Main result - shows identifiability collapse
- **X-axis**: Classification accuracy
- **Y-axis**: Identifiability proxy (log scale)
- **Key insight**: High accuracy can coexist with near-zero identifiability

#### `assets/figures/fig_param_error_vs_noise.png`
- **Purpose**: Parameter recovery degradation
- **X-axis**: Noise index (p + σ)
- **Y-axis**: Parameter L2 error
- **Key insight**: Parameter recovery generally degrades with noise

### 2. Extended Visualizations (`--extended-viz`)

Additional static visualizations providing deeper insights:

#### Heatmaps (3 files)
- `heatmap_acc.png`: Accuracy across noise parameter grid
- `heatmap_param_l2.png`: Parameter error across noise grid
- `heatmap_ident_proxy.png`: Identifiability across noise grid

**Features**:
- 2D grid: depolarizing probability (p) vs. phase noise (σ)
- Color-coded metric values
- Annotated cells with numerical values
- Useful for identifying noise regimes

#### `combined_metrics.png`
Comprehensive 2×2 grid showing:
1. **Accuracy vs Noise**: How task performance varies
2. **Parameter Error vs Noise**: Recovery quality trends
3. **Identifiability Collapse**: Log-scale identifiability decay
4. **Accuracy vs Identifiability**: Main result with noise annotations

**Use case**: Single-figure summary for papers/presentations

### 3. Interactive Visualizations (`--interactive`)

Interactive HTML dashboards using Plotly (requires `plotly` package):

#### `interactive_heatmaps.html`
- Three interactive heatmaps side-by-side
- Hover to see exact values
- Pan and zoom capabilities
- Export-ready

#### `interactive_dashboard.html`
- Comprehensive 4-panel dashboard
- Interactive hover tooltips
- Zoom and pan on all plots
- Color-coded by noise level or identifiability
- Annotations for each data point

**Advantages**:
- Explore data interactively
- Better for presentations
- Easier to identify specific noise settings
- Publication-quality exports

### 4. Advanced Visualizations (Programmatic)

Available through Python API:

#### Loss Landscape Visualizations

```python
from src.visualizations import plot_loss_landscape_2d, plot_loss_landscape_pca

# 2D slice of loss landscape
plot_loss_landscape_2d(
    loss_fn=lambda theta: my_loss(theta),
    theta_opt=best_theta,
    output_path=Path("loss_landscape.png"),
    n_points=50,
    direction_indices=(0, 1)  # Which parameter dimensions to visualize
)

# Loss landscape along principal components of trajectory
plot_loss_landscape_pca(
    loss_fn=lambda theta: my_loss(theta),
    theta_opt=best_theta,
    theta_history=optimization_trajectory,
    output_path=Path("loss_landscape_pca.png")
)
```

**Features**:
- Contour and 3D surface plots
- Optimum marked clearly
- Customizable dimensions
- PCA-based visualization shows trajectory in principal component space

#### Parameter Trajectory Plots

```python
from src.visualizations import plot_parameter_trajectory

plot_parameter_trajectory(
    theta_history=trajectory,
    theta_true=ground_truth,
    loss_history=losses,
    output_path=Path("trajectory.png")
)
```

**Shows**:
1. Parameter evolution over iterations
2. Distance to ground truth (log scale)
3. Loss evolution (log scale)
4. 2D parameter space trajectory

#### Curvature Spectrum Analysis

```python
from src.visualizations import plot_curvature_spectrum

plot_curvature_spectrum(
    hessian_diag=hessian_diagonal,
    output_path=Path("curvature.png"),
    title="Hessian Diagonal Spectrum"
)
```

**Features**:
- Bar chart by parameter index
- Sorted spectrum view
- Condition number annotation
- Identifies flat directions

#### Interactive Loss Landscape

```python
from src.interactive_viz import create_interactive_loss_landscape

create_interactive_loss_landscape(
    loss_fn=lambda theta: my_loss(theta),
    theta_opt=best_theta,
    output_path=Path("loss_landscape_interactive.html"),
    n_points=50
)
```

**Features**:
- Rotatable 3D surface
- Hover for exact loss values
- Optimum marked
- Export to image

## Usage Examples

### CLI Examples

```bash
# Quick visualization test
python -m src --noise-presets minimal --extended-viz

# Full analysis with all visualizations
python -m src \
  --samples 1000 \
  --dimension 16 \
  --noise-presets extensive \
  --extended-viz \
  --interactive \
  --output-dir ./results/full_analysis

# From configuration file
python -m src --config examples/extended_viz_example.yaml
```

### Programmatic Examples

#### Example 1: Generate All Visualizations for Results

```python
from src.plots import plot_results
from src.visualizations import *
from src.interactive_viz import *
from pathlib import Path

# Assume you have results from an experiment
results = [...]  # List of result dicts

output_dir = Path("./visualizations")

# Generate standard plots
plot_results(results, output_dir=output_dir)

# Generate heatmaps
for metric in ['acc', 'param_l2', 'ident_proxy']:
    plot_noise_heatmap(results, output_dir / f"heatmap_{metric}.png", metric=metric)

# Generate combined metrics
plot_combined_metrics_grid(results, output_dir / "combined.png")

# Generate interactive dashboard
create_interactive_metrics_dashboard(results, output_dir / "dashboard.html")
```

#### Example 2: Loss Landscape Analysis

```python
import numpy as np
from src.visualizations import plot_loss_landscape_2d

# Define your loss function
def my_loss(theta):
    # Your loss computation
    return np.sum(theta**2)

# Optimal parameters found
theta_opt = np.array([0.1, 0.2, 0.3, 0.4])

# Generate landscape
plot_loss_landscape_2d(
    loss_fn=my_loss,
    theta_opt=theta_opt,
    output_path=Path("landscape.png"),
    n_points=30,
    scale=1.0,
    direction_indices=(0, 1)
)
```

#### Example 3: Trajectory Visualization

```python
from src.visualizations import plot_parameter_trajectory

# Collect trajectory during optimization
theta_history = []
loss_history = []

for iteration in range(num_iterations):
    theta = optimize_step()
    loss = compute_loss(theta)
    theta_history.append(theta)
    loss_history.append(loss)

# Visualize
plot_parameter_trajectory(
    theta_history=theta_history,
    theta_true=true_parameters,
    loss_history=loss_history,
    output_path=Path("trajectory.png")
)
```

## Configuration File Support

Add visualization options to experiment configs:

```yaml
name: my_experiment
samples: 512
dimension: 8
# ... other params ...
generate_plots: true  # Standard plots
# Note: extended_viz and interactive are CLI-only for now
```

Then use CLI flags:

```bash
python -m src --config my_experiment.yaml --extended-viz --interactive
```

## Output Structure

### Standard Run
```
assets/figures/
├── fig_accuracy_vs_identifiability.png
└── fig_param_error_vs_noise.png
```

### With Extended Visualizations
```
assets/figures/
├── fig_accuracy_vs_identifiability.png
├── fig_param_error_vs_noise.png
├── heatmap_acc.png
├── heatmap_param_l2.png
├── heatmap_ident_proxy.png
└── combined_metrics.png
```

### With Interactive Visualizations
```
assets/figures/
├── fig_accuracy_vs_identifiability.png
├── fig_param_error_vs_noise.png
├── interactive_heatmaps.html
└── interactive_dashboard.html
```

## Dependencies

- **matplotlib**: Standard visualizations (included)
- **plotly**: Interactive visualizations (optional)

To install plotly:
```bash
pip install plotly
# or
pip install -e .  # If using editable install
```

## Tips and Best Practices

1. **Start with standard visualizations**: Understand basic patterns first
2. **Use extended viz for papers**: Heatmaps and combined metrics are publication-ready
3. **Interactive for exploration**: Best for detailed data analysis
4. **Loss landscapes are expensive**: Use smaller grid sizes (n_points=30) for faster generation
5. **Trajectory visualization requires history**: Modify optimizer to track parameters
6. **Heatmaps need grid data**: Works best with structured noise sweeps

## Performance Considerations

- **Loss landscape**: O(n²) loss evaluations - can be slow
- **PCA landscape**: Requires saved trajectory (memory overhead)
- **Interactive plots**: Larger HTML files but worth it for analysis
- **Heatmaps**: Fast for reasonable grid sizes (< 10×10)

## Customization

All visualization functions accept additional parameters:

```python
# Customize colors, labels, sizes, etc.
plot_noise_heatmap(
    results,
    output_path=Path("custom_heatmap.png"),
    metric='ident_proxy'
    # Modify source to add custom parameters
)
```

See source code in `src/visualizations.py` and `src/interactive_viz.py` for full customization options.

## Troubleshooting

### Plotly not found
```
Warning: plotly not available for interactive visualizations
```
**Solution**: `pip install plotly`

### Memory issues with large grids
**Solution**: Reduce `n_points` parameter or use smaller problem dimensions

### Figures too small/large
**Solution**: Modify `figsize` parameters in source code or adjust DPI

## Examples Gallery

See the `examples/` directory for configuration files that generate various visualizations:

- `extended_viz_example.yaml`: Full visualization suite
- `minimal_example.yaml`: Quick test
- `hyperparameter_sweep.yaml`: Large-scale visualization

## Future Enhancements

Potential additions:
- Animated trajectory videos
- 3D parameter space trajectories
- Parallel coordinates plots
- t-SNE/UMAP embeddings for high-dimensional parameters
- Real-time optimization visualization

---

**License:** MIT (see `LICENSE`) · **Contact:** x@christopheraltman.com  
Back to: `README.md`