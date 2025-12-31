# Quick Reference Guide

**One-page reference for the Noise-Aware QNN Identifiability project**

---

## 🚀 Quick Start

```bash
# Default run
python -m src

# With enhanced metrics
python -m src --enhanced-metrics

# Full visualization suite
python -m src --extended-viz --interactive

# From config file
python -m src --config examples/enhanced_metrics_example.yaml
```

---

## 📊 Core Metrics

| Metric | Formula | Interpretation | Threshold |
|--------|---------|----------------|-----------|
| **Accuracy** | Correct / Total | Task performance | > 0.90 good |
| **Param Error** | ‖θ̂ - θ*‖₂ | Recovery error | < 0.1 good |
| **Identifiability** | min\|Hᵢᵢ\| / max\|Hᵢᵢ\| | Curvature ratio | > 0.01 good |

---

## 🔬 Enhanced Metrics (Fisher Information)

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| **Fisher Condition Number** κ(F) | Ill-conditioning | < 1,000 |
| **Effective Rank** | True dimensionality | ≈ param_dim |
| **Effective Dimension** | Well-determined params | ≈ param_dim |
| **Participation Ratio** | Dimensional participation | > 0.7 |
| **Fisher Trace** | Total information | Higher = better |

### Quality Assessment

| κ(F) Range | Quality | Status |
|------------|---------|--------|
| < 100 | Excellent | ✅ Strongly identifiable |
| 100-1,000 | Good | ✅ Identifiable |
| 1,000-10,000 | Fair | ⚠️ Marginally identifiable |
| > 10,000 | Poor | ❌ Non-identifiable (collapse) |

---

## 🧮 Key Formulas

### Fisher Information Matrix
```
F_ij = E[(∂log p(y|x,θ)/∂θᵢ)(∂log p(y|x,θ)/∂θⱼ)]
```

### Condition Number
```
κ(F) = λ_max / λ_min
```

### Participation Ratio (Effective Rank)
```
PR = (Σλᵢ)² / Σλᵢ²
```

### Cramér-Rao Bound
```
Var(θ̂) ≥ F⁻¹
```

**Implication**: Large κ(F) → large variance → non-identifiable

---

## 📁 File Structure

```
├── src/
│   ├── enhanced_metrics.py      # Fisher Information (486 lines)
│   ├── cli.py                   # CLI interface (272 lines)
│   ├── config.py                # Configuration (283 lines)
│   ├── visualizations.py        # Static plots (420 lines)
│   ├── interactive_viz.py       # Plotly plots (508 lines)
│   └── ...
├── tests/                       # 56 passing tests
├── examples/                    # 6 YAML configs
└── *.md                         # 7 documentation files
```

---

## 📚 Documentation Map

| File | Purpose | Lines |
|------|---------|-------|
| `README.md` | Overview & quick start | 294 |
| `PROJECT_SUMMARY.md` | Complete feature list | 509 |
| `ENHANCED_METRICS.md` | Fisher Information guide | 572 |
| `CLI.md` | Command-line reference | 197 |
| `CONFIGURATION.md` | Config system guide | 366 |
| `VISUALIZATIONS.md` | Plotting guide | 395 |
| `DATA_EXPORT_LOGGING.md` | Data management | 646 |
| `QUICK_REFERENCE.md` | This file | - |

---

## 🐍 Python API Examples

### Basic Enhanced Metrics
```python
from src.enhanced_metrics import compute_all_enhanced_metrics

metrics = compute_all_enhanced_metrics(
    model_fn=model_fn,      # (x, theta) -> p(y=1|x,theta)
    loss_fn=loss_fn,        # (theta) -> scalar
    theta=best_theta,
    X=data,
    y=labels
)

print(f"κ(F) = {metrics['fisher_condition_number']:.2e}")
print(f"Effective rank = {metrics['fisher_effective_rank']:.2f}")
print(f"Quality: {metrics['information_geometry_quality']}")
```

### Pretty Report
```python
from src.enhanced_metrics import format_metrics_report

report = format_metrics_report(metrics)
print(report)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_functions.py -v

# Run with coverage
pytest tests/ --cov=src
```

**Current Status**: 56/56 tests passing ✅

---

## 🎨 Visualization Options

### Static Plots (matplotlib)
- `--extended-viz`: Loss landscapes, trajectories, heatmaps

### Interactive Plots (plotly)
- `--interactive`: Rotatable 3D plots, hoverable dashboards

### Example
```bash
python -m src --extended-viz --interactive --output-dir results/viz
```

---

## ⚙️ Configuration

### CLI Flags
```bash
--samples N              # Number of training samples
--input-dim D            # Input dimension
--param-dim D            # Parameter dimension
--iterations N           # Optimization iterations
--depolarizing p1 p2 ... # Depolarizing noise levels
--phase σ1 σ2 ...        # Phase noise levels
--enhanced-metrics       # Enable Fisher computation
--extended-viz           # Extended visualizations
--interactive            # Interactive plots
--config FILE            # Load from YAML/JSON
--output-dir DIR         # Output directory
--verbose                # Verbose output
```

### Config File Format
```yaml
experiment:
  n_samples: 100
  input_dim: 4
  param_dim: 4
  
  noise:
    depolarizing_probs: [0.0, 0.1, 0.2]
    phase_stds: [0.0, 0.05, 0.1]
  
  enhanced_metrics:
    enabled: true
    batch_size: 50
```

---

## 🔑 Key Insights

### The Core Thesis
> High predictive accuracy can coexist with a collapse of parameter identifiability.

### The Mathematical Explanation
> Identifiability collapses because the information geometry becomes ill-conditioned.

### How We Prove It
1. Compute Fisher Information Matrix F
2. Calculate condition number κ(F) = λ_max / λ_min
3. Show κ(F) → ∞ as noise increases
4. Measure effective rank dropping: PR << d

### Why This Matters
- **Cramér-Rao bound**: Var(θ̂) ≥ F⁻¹
- Large κ(F) → huge variance → non-identifiable
- Rigorous mathematical foundation
- Publication-ready language

---

## 📊 Expected Results

### Clean (No Noise)
```
Accuracy: > 95%
Fisher κ(F): < 100
Effective Rank: ≈ param_dim
Quality: "excellent"
```

### High Noise
```
Accuracy: > 90% (still good!)
Fisher κ(F): > 10,000 (collapse!)
Effective Rank: << param_dim
Quality: "poor_ill_conditioned"
```

**This is the identifiability collapse we're demonstrating.**

---

## 🏷️ Dependencies

### Required
- `numpy` - Numerical computation
- `matplotlib` - Plotting
- `scipy` - Fisher computation

### Optional
- `plotly` - Interactive plots
- `pyyaml` - Config files

### Install
```bash
pip install numpy matplotlib scipy plotly pyyaml pytest
```

---

## 🎯 Common Tasks

### Run Default Experiment
```bash
python -m src
```

### Custom Noise Sweep
```bash
python -m src --depolarizing 0.0 0.05 0.1 0.15 0.2 --phase 0.0 0.02 0.05
```

### Enable Everything
```bash
python -m src --enhanced-metrics --extended-viz --interactive --verbose
```

### Batch Experiments
```bash
python -m src --config examples/batch_experiments.yaml
```

### Run Tests
```bash
pytest tests/ -v
```

---

## 📈 Performance Notes

### Fisher Computation Complexity
- **Full FIM**: O(n × d²) - use for d ≤ 16
- **Batch FIM**: O(b × d²) - use for d > 16
- **Hessian only**: O(d) - fastest, less rigorous

### Recommended Settings
```python
# Small problems (d ≤ 16)
compute_full_fisher: true
batch_size: null

# Medium problems (16 < d ≤ 64)
compute_full_fisher: false
batch_size: 100

# Large problems (d > 64)
enhanced_metrics: false  # Use Hessian only
```

---

## 🚨 Troubleshooting

### Fisher matrix is singular
- Add regularization: F + εI
- Increase batch_size
- Use SVD-based condition number

### Computation is slow
- Reduce batch_size
- Use Hessian-only metrics
- Disable enhanced_metrics

### Tests failing
```bash
# Check environment
python -c "import numpy, scipy, matplotlib; print('OK')"

# Re-run tests
pytest tests/ -v --tb=short
```

---

## 📝 Citation

```bibtex
@software{altman2025noise-aware-qnn,
  author  = {Christopher Altman},
  title   = {Noise-Aware QNN Identifiability},
  year    = {2025},
  url     = {https://github.com/christopher-altman/noise-aware-qnn-identifiability}
}
```

---

**License:** MIT (see `LICENSE`) · **Contact:** x@christopheraltman.com  
Back to: `README.md`
