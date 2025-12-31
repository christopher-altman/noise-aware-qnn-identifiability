# Enhanced Identifiability Metrics

**Rigorous mathematical analysis of identifiability using information geometry.**

> "Identifiability collapses because the information geometry becomes ill-conditioned."

This module provides Fisher Information Matrix analysis, condition numbers, effective dimensionality, and gradient stability metrics for deep identifiability analysis.

## Quick Start

```python
from src.enhanced_metrics import compute_all_enhanced_metrics, format_metrics_report

# Define your model and loss
def model_fn(x, theta):
    # Returns p(y=1|x,theta)
    phi = feature_map(x)
    score = qnn_forward(phi, theta)
    return 1.0 / (1.0 + np.exp(-score))

def loss_fn(theta):
    # Your loss function
    return compute_loss(theta, X, y)

# Compute all metrics
metrics = compute_all_enhanced_metrics(
    model_fn=model_fn,
    loss_fn=loss_fn,
    theta=best_theta,
    X=training_data,
    y=training_labels
)

# Print report
print(format_metrics_report(metrics))
```

## Core Metrics

### 1. Fisher Information Matrix (FIM)

**Mathematical Definition:**

For a parametric model p(y|x,θ), the Fisher Information Matrix is:

```
F_ij = E[(∂log p(y|x,θ)/∂θ_i)(∂log p(y|x,θ)/∂θ_j)]
```

**Interpretation:**
- Measures the amount of information that data carries about parameters
- High eigenvalues → parameter is well-determined by data
- Low eigenvalues → parameter is poorly constrained (flat directions)

**What We Compute:**
- **Fisher Trace**: Total information mass (sum of eigenvalues)
- **Fisher Condition Number**: κ(F) = λ_max / λ_min (measures ill-conditioning)
- **Effective Rank**: Participation ratio PR = (Σλ_i)² / Σλ_i²
- **Effective Dimension**: Number of eigenvalues capturing 99% of spectral mass
- **Participation Ratio**: How many dimensions carry significant information
- **Spectral Gap**: λ_max / λ_min (large gap → ill-conditioning)

### 2. Condition Number

**Definition:**

```
κ(F) = λ_max / λ_min
```

**Interpretation:**
- κ < 100: Excellent conditioning, parameters well-identified
- κ < 1,000: Good conditioning
- κ < 10,000: Fair conditioning  
- κ > 10,000: Poor conditioning, identifiability collapse

**Why It Matters:**

Large condition numbers indicate that small changes in data can lead to large changes in parameter estimates → **non-identifiability**.

### 3. Effective Rank (Participation Ratio)

**Definition:**

```
PR = (Σλ_i)² / Σλ_i²
```

**Interpretation:**
- Measures how many eigenvalue directions participate
- PR = d (full rank) → all parameters independently identifiable
- PR << d → many flat directions, parameters are coupled/unidentifiable

**Example:**
- 8 parameters, PR = 7.8 → excellent, nearly full rank
- 8 parameters, PR = 2.1 → poor, only ~2 independent directions

### 4. Effective Dimension

**Definition:**

Number of eigenvalues λ_i such that:

```
Σ(i=1 to k) λ_i / Σ(all) λ_i ≥ 0.99
```

**Interpretation:**
- How many parameter dimensions are actually informed by data
- Effective dim = d → full dimensional information
- Effective dim << d → low-dimensional manifold, redundancy

### 5. Hessian Condition Number

**Definition:**

From loss Hessian H:

```
κ(H) = |H_max| / |H_min|
```

**Interpretation:**
- Measures curvature of loss landscape
- High condition number → flat directions exist
- Complements Fisher analysis

## API Reference

### Main Functions

#### `compute_all_enhanced_metrics()`

Compute all metrics in one call (recommended):

```python
metrics = compute_all_enhanced_metrics(
    model_fn=model_fn,        # (x, theta) -> p(y=1|x,theta)
    loss_fn=loss_fn,          # (theta) -> scalar loss
    theta=best_theta,         # Optimized parameters
    X=training_data,          # Input data
    y=training_labels,        # Binary labels
    hessian_eps=1e-5,         # Finite difference epsilon
    batch_size=100            # Batch size for Fisher computation
)
```

**Returns:**
```python
{
    # Fisher Information metrics
    'fisher_trace': 12.456,
    'fisher_condition_number': 1523.4,
    'fisher_effective_rank': 6.2,
    'fisher_effective_dimension': 7,
    'fisher_participation_ratio': 0.775,
    'fisher_spectral_gap': 1523.4,
    'fisher_eigenvalue_max': 5.2,
    'fisher_eigenvalue_min': 0.0034,
    
    # Hessian metrics
    'hessian_condition_number': 892.1,
    'hessian_max': 2.1,
    'hessian_min': 0.0024,
    
    # Legacy
    'identifiability_proxy': 0.0011,
    
    # Assessment
    'is_well_conditioned': False,
    'is_identifiable': True,
    'information_geometry_quality': 'fair'
}
```

#### `compute_fisher_information_matrix()`

Compute full FIM (slower, more accurate):

```python
fisher = compute_fisher_information_matrix(
    model_fn=model_fn,
    theta=theta,
    X=X,
    y=y,
    eps=1e-7  # Numerical stability constant
)
# Returns (d, d) matrix
```

#### `compute_empirical_fisher()`

Efficient batch-based FIM computation:

```python
fisher = compute_empirical_fisher(
    model_fn=model_fn,
    theta=theta,
    X=X,
    y=y,
    batch_size=100  # Use subset for efficiency
)
```

#### `analyze_identifiability_geometry()`

Analyze Fisher and Hessian together:

```python
analysis = analyze_identifiability_geometry(
    fisher_matrix=fisher,
    hessian_diag=hessian_diag
)
```

#### `format_metrics_report()`

Generate human-readable report:

```python
report = format_metrics_report(metrics)
print(report)
```

**Output:**
```
============================================================
IDENTIFIABILITY ANALYSIS: Information Geometry
============================================================

Fisher Information Matrix:
  Trace (information mass):    1.2456e+01
  Condition number:            1.5234e+03
  Effective rank:              6.20
  Effective dimension:         7
  Participation ratio:         0.7750
  Spectral gap:                1.5234e+03

Loss Hessian:
  Condition number:            8.9210e+02
  Max curvature:               2.1000e+00
  Min curvature:               2.4000e-03

Assessment:
  Well-conditioned:            False
  Identifiable:                True
  Geometry quality:            fair

============================================================
INTERPRETATION:
⚠  The information geometry is MARGINALLY conditioned.
   Some parameter directions may be weakly identifiable.
============================================================
```

### Utility Functions

#### `compute_condition_number()`

```python
cond = compute_condition_number(
    matrix=fisher,
    method='svd'  # or 'eigenvalue'
)
```

#### `compute_effective_rank()`

```python
eff_rank = compute_effective_rank(
    matrix=fisher,
    threshold=1e-10
)
```

#### `compute_effective_dimension()`

```python
eff_dim = compute_effective_dimension(
    matrix=fisher,
    threshold=0.99  # Capture 99% of spectral mass
)
```

#### `compute_gradient_stability()`

Track optimization stability:

```python
gradients = []  # Collect during training
# ... training loop ...

stability = compute_gradient_stability(
    gradients=np.array(gradients),
    window_size=10
)
# Returns:
# {
#     'mean_grad_norm': 0.123,
#     'grad_norm_std': 0.045,
#     'grad_cosine_similarity': 0.89,
#     'grad_angle_variance': 12.3
# }
```

## Mathematical Theory

### Why Fisher Information?

The Fisher Information Matrix provides a **natural Riemannian metric** on the parameter space. It tells us:

1. **How much information** data provides about each parameter
2. **Which parameter directions** are well-determined vs. flat
3. **The local curvature** of the information geometry

### Connection to Identifiability

A parameter θ is **identifiable** if different values of θ produce distinguishably different data distributions:

```
θ₁ ≠ θ₂  ⟹  p(·|θ₁) ≠ p(·|θ₂)
```

**Mathematically:**

- If F is full rank (det(F) > 0), parameters are locally identifiable
- If F is singular or nearly singular (κ(F) → ∞), identifiability collapses
- Effective rank measures the "true" number of identifiable directions

### Why Condition Number Matters

The **Cramér-Rao bound** states:

```
Var(θ̂) ≥ F⁻¹
```

**Implications:**

- If κ(F) is large, F⁻¹ has very large eigenvalues
- This means parameter estimates have huge variance
- Small data perturbations → large parameter changes
- **Practical non-identifiability**

### Noise-Induced Collapse

Under noise:

1. **Information Loss**: Fisher eigenvalues shrink
2. **Flat Directions Emerge**: Some λ_i → 0
3. **Condition Number Explodes**: κ(F) → ∞
4. **Effective Rank Drops**: Many directions become unidentifiable

**This is exactly what we observe in experiments!**

## Usage Examples

### Example 1: Basic Analysis

```python
import numpy as np
from src.enhanced_metrics import compute_all_enhanced_metrics

# Your model
def model_fn(x, theta):
    phi = x / (np.linalg.norm(x) + 1e-12)
    t = theta / (np.linalg.norm(theta) + 1e-12)
    score = np.dot(phi, t)
    return 1.0 / (1.0 + np.exp(-score))

# Your loss
def loss_fn(theta):
    total_loss = 0.0
    for i in range(len(X)):
        p = model_fn(X[i], theta)
        total_loss += -(y[i]*np.log(p+1e-12) + (1-y[i])*np.log(1-p+1e-12))
    return total_loss / len(X)

# Compute metrics
metrics = compute_all_enhanced_metrics(
    model_fn=model_fn,
    loss_fn=loss_fn,
    theta=optimized_theta,
    X=data,
    y=labels
)

print(f"Fisher condition number: {metrics['fisher_condition_number']:.2e}")
print(f"Effective rank: {metrics['fisher_effective_rank']:.2f}")
print(f"Quality: {metrics['information_geometry_quality']}")
```

### Example 2: Noise Sweep Analysis

```python
from src.enhanced_metrics import compute_all_enhanced_metrics

results = []

for noise_level in [0.0, 0.05, 0.10, 0.15, 0.20]:
    # Train model with noise
    theta = train_with_noise(noise_level)
    
    # Compute enhanced metrics
    metrics = compute_all_enhanced_metrics(
        model_fn=model_fn,
        loss_fn=lambda t: loss_with_noise(t, noise_level),
        theta=theta,
        X=X,
        y=y
    )
    
    results.append({
        'noise': noise_level,
        'fisher_cond': metrics['fisher_condition_number'],
        'eff_rank': metrics['fisher_effective_rank'],
        'quality': metrics['information_geometry_quality']
    })

# Visualize collapse
import matplotlib.pyplot as plt

noises = [r['noise'] for r in results]
conds = [r['fisher_cond'] for r in results]

plt.semilogy(noises, conds, 'o-')
plt.xlabel('Noise Level')
plt.ylabel('Fisher Condition Number')
plt.title('Information Geometry Degrades with Noise')
plt.show()
```

### Example 3: Gradient Stability Tracking

```python
from src.enhanced_metrics import compute_gradient_stability

# During training, collect gradients
gradient_history = []

for epoch in range(num_epochs):
    grad = compute_gradient(theta)
    gradient_history.append(grad)
    theta = theta - learning_rate * grad

# Analyze stability
stability = compute_gradient_stability(
    gradients=np.array(gradient_history),
    window_size=10
)

print(f"Mean gradient norm: {stability['mean_grad_norm']:.4f}")
print(f"Gradient cosine similarity: {stability['grad_cosine_similarity']:.4f}")
print(f"Angle variance: {stability['grad_angle_variance']:.2f}°")
```

## Interpretation Guide

### Fisher Condition Number

| Range | Quality | Identifiability |
|-------|---------|-----------------|
| < 100 | Excellent | Strongly identifiable |
| 100-1,000 | Good | Identifiable |
| 1,000-10,000 | Fair | Marginally identifiable |
| > 10,000 | Poor | Non-identifiable |

### Participation Ratio

| Value | Meaning |
|-------|---------|
| > 0.9 | Nearly full rank, excellent |
| 0.7-0.9 | Good participation |
| 0.5-0.7 | Moderate, some redundancy |
| < 0.5 | Poor, significant redundancy |

### Effective Dimension

Compare to actual dimension d:

| Ratio | Interpretation |
|-------|----------------|
| eff_dim / d > 0.9 | All parameters informed |
| eff_dim / d = 0.5-0.9 | Some redundancy |
| eff_dim / d < 0.5 | Severe dimensional collapse |

## Best Practices

1. **Always compute Fisher metrics** for rigorous identifiability analysis
2. **Report condition numbers** alongside accuracy
3. **Track effective rank** over noise sweeps
4. **Use batch computation** for large datasets (faster)
5. **Compare with Hessian** for cross-validation
6. **Visualize eigenvalue spectra** to see collapse directly

## Performance Considerations

### Computational Complexity

- **Full FIM**: O(n × d²) where n = data size, d = parameter dimension
- **Batch FIM**: O(b × d²) where b = batch size (<< n)
- **Hessian diagonal**: O(d) (much faster)

### Recommendations

- For d ≤ 16: Use full FIM
- For d > 16: Use batch FIM with batch_size=100
- For d > 64: Consider Hessian-based approximations only

### Memory Usage

- FIM requires O(d²) memory
- For very large d, use sparse or low-rank approximations

## Comparison with Legacy Metrics

| Metric | Old | Enhanced |
|--------|-----|----------|
| Identifiability | Hessian diagonal ratio | Fisher condition number |
| Dimensionality | Full d assumed | Effective rank/dimension |
| Conditioning | Not measured | Full spectral analysis |
| Rigor | Ad-hoc | Information geometry |

**The enhanced metrics provide rigorous mathematical foundation.**

## Citation

When using these metrics in publications, cite both the empirical Fisher approach and participation ratio:

```
“We assess identifiability using the Fisher Information Matrix condition number
and effective rank (participation ratio), which rigorously quantify the information
geometry of the parameter space. Identifiability collapses when the Fisher matrix
becomes ill-conditioned (κ(F) > 10⁴), indicating that the information geometry has
degenerated and parameters are no longer uniquely recoverable from data.”
```

## Troubleshooting

### Fisher matrix is singular

**Symptom**: Condition number = inf

**Solutions**:
- Use regularization: F + εI
- Check for redundant parameters
- Increase batch size
- Use SVD-based condition number

### Computation is slow

**Solutions**:
- Reduce batch_size
- Use Hessian-only metrics
- Parallelize Fisher computation
- Use analytical gradients if available

### Numerical instability

**Solutions**:
- Increase eps parameter
- Use double precision (float64)
- Add regularization to Fisher matrix
- Clip extreme values

## Future Enhancements

- Analytical gradient computation (faster)
- Parallel/GPU Fisher computation
- Full Hessian (not just diagonal)
- Natural gradient visualization
- Bayesian Fisher (with priors)

---

**License:** MIT (see `LICENSE`) · **Contact:** x@christopheraltman.com  
Back to: `README.md`