import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
import json


def plot_loss_landscape_2d(
    loss_fn: Callable,
    theta_opt: np.ndarray,
    output_path: Path,
    n_points: int = 50,
    scale: float = 1.0,
    direction_indices: Optional[Tuple[int, int]] = None
):
    """
    Plot 2D slice of loss landscape around optimum.
    
    Args:
        loss_fn: Loss function that takes theta and returns scalar loss
        theta_opt: Optimal parameters (center of visualization)
        output_path: Path to save figure
        n_points: Number of grid points in each direction
        scale: Scale factor for exploration range
        direction_indices: Tuple of (i, j) indices to visualize. If None, uses first two dimensions.
    """
    d = len(theta_opt)
    
    # Select dimensions to visualize
    if direction_indices is None:
        i, j = 0, min(1, d-1)
    else:
        i, j = direction_indices
    
    # Create grid
    delta = scale * np.linalg.norm(theta_opt) / 5.0
    if delta < 1e-6:
        delta = 0.1
    
    x = np.linspace(-delta, delta, n_points)
    y = np.linspace(-delta, delta, n_points)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate loss on grid
    Z = np.zeros_like(X)
    for idx_x in range(n_points):
        for idx_y in range(n_points):
            theta_probe = theta_opt.copy()
            theta_probe[i] += X[idx_x, idx_y]
            theta_probe[j] += Y[idx_x, idx_y]
            Z[idx_x, idx_y] = loss_fn(theta_probe)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Contour plot
    levels = np.linspace(Z.min(), Z.max(), 20)
    contour = ax1.contour(X, Y, Z, levels=levels, cmap='viridis')
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.plot(0, 0, 'r*', markersize=15, label='Optimum')
    ax1.set_xlabel(f'Δθ[{i}]')
    ax1.set_ylabel(f'Δθ[{j}]')
    ax1.set_title('Loss Landscape Contours')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 3D surface plot
    ax2.remove()
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax2.plot([0], [0], [loss_fn(theta_opt)], 'r*', markersize=15)
    ax2.set_xlabel(f'Δθ[{i}]')
    ax2.set_ylabel(f'Δθ[{j}]')
    ax2.set_zlabel('Loss')
    ax2.set_title('Loss Landscape Surface')
    fig.colorbar(surf, ax=ax2, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_loss_landscape_pca(
    loss_fn: Callable,
    theta_opt: np.ndarray,
    theta_history: List[np.ndarray],
    output_path: Path,
    n_points: int = 50
):
    """
    Plot loss landscape along principal components of optimization trajectory.
    
    Args:
        loss_fn: Loss function
        theta_opt: Optimal parameters
        theta_history: List of parameter vectors from optimization
        output_path: Path to save figure
        n_points: Grid resolution
    """
    if len(theta_history) < 2:
        print("Not enough history for PCA visualization")
        return
    
    # Convert history to matrix
    history_matrix = np.array(theta_history)
    
    # Center and compute PCA
    mean_theta = history_matrix.mean(axis=0)
    centered = history_matrix - mean_theta
    
    # Get top 2 principal components
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    pc1 = eigenvectors[:, 0]
    pc2 = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.random.randn(len(pc1))
    
    # Create grid along principal components
    scale1 = 2.0 * np.sqrt(eigenvalues[0]) if eigenvalues[0] > 0 else 1.0
    scale2 = 2.0 * np.sqrt(eigenvalues[1]) if len(eigenvalues) > 1 and eigenvalues[1] > 0 else 1.0
    
    x = np.linspace(-scale1, scale1, n_points)
    y = np.linspace(-scale2, scale2, n_points)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate loss
    Z = np.zeros_like(X)
    for idx_x in range(n_points):
        for idx_y in range(n_points):
            theta_probe = theta_opt + X[idx_x, idx_y] * pc1 + Y[idx_x, idx_y] * pc2
            Z[idx_x, idx_y] = loss_fn(theta_probe)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax.plot(0, 0, 'r*', markersize=15, label='Optimum')
    
    # Project history onto principal components
    history_proj = []
    for theta in theta_history:
        delta = theta - theta_opt
        proj1 = np.dot(delta, pc1)
        proj2 = np.dot(delta, pc2)
        history_proj.append([proj1, proj2])
    
    history_proj = np.array(history_proj)
    ax.plot(history_proj[:, 0], history_proj[:, 1], 'w.-', alpha=0.6, linewidth=2, label='Trajectory')
    ax.plot(history_proj[0, 0], history_proj[0, 1], 'go', markersize=10, label='Start')
    
    ax.set_xlabel(f'PC1 (λ={eigenvalues[0]:.2e})')
    ax.set_ylabel(f'PC2 (λ={eigenvalues[1] if len(eigenvalues) > 1 else 0:.2e})')
    ax.set_title('Loss Landscape along Principal Components')
    ax.legend()
    plt.colorbar(contour, ax=ax, label='Loss')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_parameter_trajectory(
    theta_history: List[np.ndarray],
    theta_true: np.ndarray,
    loss_history: List[float],
    output_path: Path
):
    """
    Plot parameter evolution during optimization.
    
    Args:
        theta_history: List of parameter vectors
        theta_true: Ground truth parameters
        loss_history: Loss at each iteration
        output_path: Path to save figure
    """
    theta_history = np.array(theta_history)
    iterations = np.arange(len(theta_history))
    d = theta_history.shape[1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Parameter components over time
    ax = axes[0, 0]
    for i in range(min(d, 8)):  # Plot first 8 dimensions
        ax.plot(iterations, theta_history[:, i], label=f'θ[{i}]', alpha=0.7)
        ax.axhline(theta_true[i], color=f'C{i}', linestyle='--', alpha=0.3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Parameter Evolution')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Distance to optimum
    ax = axes[0, 1]
    distances = [np.linalg.norm(theta - theta_true) for theta in theta_history]
    ax.semilogy(iterations, distances, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||θ - θ*||₂')
    ax.set_title('Distance to Ground Truth')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss evolution
    ax = axes[1, 0]
    ax.semilogy(iterations, loss_history, 'r-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Evolution')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Parameter space trajectory (2D projection)
    ax = axes[1, 1]
    if d >= 2:
        ax.plot(theta_history[:, 0], theta_history[:, 1], 'b.-', alpha=0.6, label='Trajectory')
        ax.plot(theta_history[0, 0], theta_history[0, 1], 'go', markersize=10, label='Start')
        ax.plot(theta_history[-1, 0], theta_history[-1, 1], 'ro', markersize=10, label='End')
        ax.plot(theta_true[0], theta_true[1], 'k*', markersize=15, label='True')
        ax.set_xlabel('θ[0]')
        ax.set_ylabel('θ[1]')
        ax.set_title('Trajectory in Parameter Space')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_noise_heatmap(
    results: List[Dict[str, Any]],
    output_path: Path,
    metric: str = 'ident_proxy'
):
    """
    Create heatmap of metrics across noise parameter grid.
    
    Args:
        results: List of experiment results
        output_path: Path to save figure
        metric: Metric to visualize ('acc', 'param_l2', 'ident_proxy')
    """
    # Extract unique noise values
    p_deps = sorted(set(r['p_dep'] for r in results))
    sigmas = sorted(set(r['sigma_phase'] for r in results))
    
    # Create grid
    grid = np.full((len(sigmas), len(p_deps)), np.nan)
    
    for r in results:
        i = sigmas.index(r['sigma_phase'])
        j = p_deps.index(r['p_dep'])
        grid[i, j] = r[metric]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metric_labels = {
        'acc': 'Accuracy',
        'param_l2': 'Parameter L2 Error',
        'ident_proxy': 'Identifiability Proxy'
    }
    
    im = ax.imshow(grid, aspect='auto', cmap='viridis', origin='lower')
    
    # Set ticks
    ax.set_xticks(range(len(p_deps)))
    ax.set_yticks(range(len(sigmas)))
    ax.set_xticklabels([f'{p:.2f}' for p in p_deps])
    ax.set_yticklabels([f'{s:.2f}' for s in sigmas])
    
    ax.set_xlabel('Depolarizing Probability (p)', fontsize=12)
    ax.set_ylabel('Phase Noise Sigma (σ)', fontsize=12)
    ax.set_title(f'{metric_labels.get(metric, metric)} Heatmap', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_labels.get(metric, metric), fontsize=12)
    
    # Annotate cells with values
    for i in range(len(sigmas)):
        for j in range(len(p_deps)):
            if not np.isnan(grid[i, j]):
                text = ax.text(j, i, f'{grid[i, j]:.3f}',
                             ha="center", va="center", color="w", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_curvature_spectrum(
    hessian_diag: np.ndarray,
    output_path: Path,
    title: str = 'Hessian Diagonal Spectrum'
):
    """
    Visualize Hessian diagonal curvature spectrum.
    
    Args:
        hessian_diag: Diagonal elements of Hessian
        output_path: Path to save figure
        title: Plot title
    """
    d = len(hessian_diag)
    indices = np.arange(d)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    colors = ['red' if h < 0 else 'blue' for h in hessian_diag]
    ax1.bar(indices, np.abs(hessian_diag), color=colors, alpha=0.7)
    ax1.set_xlabel('Parameter Index')
    ax1.set_ylabel('|Hessian Diagonal|')
    ax1.set_title(f'{title} (Magnitude)')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Sorted view
    sorted_h = np.sort(np.abs(hessian_diag))[::-1]
    ax2.semilogy(range(d), sorted_h, 'bo-', markersize=8, linewidth=2)
    ax2.set_xlabel('Sorted Index')
    ax2.set_ylabel('|Hessian Diagonal|')
    ax2.set_title('Sorted Curvature Spectrum')
    ax2.grid(True, alpha=0.3)
    
    # Add condition number info
    if sorted_h[-1] > 1e-12:
        condition = sorted_h[0] / sorted_h[-1]
        ax2.text(0.5, 0.95, f'Condition: {condition:.2e}',
                transform=ax2.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_combined_metrics_grid(
    results: List[Dict[str, Any]],
    output_path: Path
):
    """
    Create comprehensive grid of all key metrics.
    
    Args:
        results: List of experiment results
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Extract data
    p_deps = np.array([r['p_dep'] for r in results])
    sigmas = np.array([r['sigma_phase'] for r in results])
    noise_index = p_deps + sigmas
    accs = np.array([r['acc'] for r in results])
    param_errs = np.array([r['param_l2'] for r in results])
    idents = np.array([r['ident_proxy'] for r in results])
    
    # Plot 1: Accuracy vs Noise
    ax = axes[0, 0]
    scatter = ax.scatter(noise_index, accs, c=idents, s=100, cmap='viridis', 
                        edgecolors='black', linewidth=1)
    ax.set_xlabel('Noise Index (p + σ)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Noise Level')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Identifiability')
    
    # Plot 2: Parameter Error vs Noise
    ax = axes[0, 1]
    scatter = ax.scatter(noise_index, param_errs, c=idents, s=100, cmap='viridis',
                        edgecolors='black', linewidth=1)
    ax.set_xlabel('Noise Index (p + σ)')
    ax.set_ylabel('Parameter L2 Error')
    ax.set_title('Parameter Recovery vs Noise')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Identifiability')
    
    # Plot 3: Identifiability vs Noise
    ax = axes[1, 0]
    ax.semilogy(noise_index, idents, 'ro-', markersize=8, linewidth=2)
    ax.set_xlabel('Noise Index (p + σ)')
    ax.set_ylabel('Identifiability Proxy')
    ax.set_title('Identifiability Collapse')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy vs Identifiability (main result)
    ax = axes[1, 1]
    scatter = ax.scatter(accs, idents, c=noise_index, s=100, cmap='plasma',
                        edgecolors='black', linewidth=1)
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Identifiability Proxy')
    ax.set_yscale('log')
    ax.set_title('High Accuracy ≠ High Identifiability')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Noise Level')
    
    for i, r in enumerate(results):
        axes[1, 1].annotate(f"p={r['p_dep']:.2f},σ={r['sigma_phase']:.2f}",
                           (accs[i], idents[i]), fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_visualization_metadata(
    output_dir: Path,
    metadata: Dict[str, Any]
):
    """Save metadata about generated visualizations."""
    metadata_path = output_dir / 'visualization_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
