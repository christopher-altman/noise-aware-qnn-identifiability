import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional

def plot_results(results, output_dir=None, extended_viz: bool = False, interactive: bool = False):
    """
    Generate visualizations for experiment results.

    Args:
        results: List of experiment results
        output_dir: Output directory for plots
        extended_viz: Generate extended visualizations (heatmaps, combined metrics)
        interactive: Generate interactive Plotly visualizations
    """
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = np.array([r["p_dep"] for r in results])
    s = np.array([r["sigma_phase"] for r in results])
    acc = np.array([r["acc"] for r in results])
    perr = np.array([r["param_l2"] for r in results])
    ident = np.array([r["ident_proxy"] for r in results])

    # Annotation policy for dense sweeps
    ANNOTATE_ALL_THRESHOLD = 15
    n_results = len(results)

    def get_indices_to_annotate():
        """Get indices of points to annotate based on sweep density."""
        if n_results <= ANNOTATE_ALL_THRESHOLD:
            # Small sweep: annotate all points
            return set(range(n_results))
        else:
            # Dense sweep: annotate only extremes
            indices = set()

            # Lowest identifiability (most collapsed)
            indices.add(int(np.argmin(ident)))

            # Highest accuracy
            indices.add(int(np.argmax(acc)))

            # Highest Fisher condition number (if available)
            if results and 'fisher_condition_number' in results[0]:
                fisher_cond = np.array([r.get('fisher_condition_number', np.nan) for r in results])
                if not np.all(np.isnan(fisher_cond)):
                    indices.add(int(np.nanargmax(fisher_cond)))

            return indices

    indices_to_annotate = get_indices_to_annotate()

    # Accuracy vs identifiability proxy (the money plot)
    plt.figure()
    plt.scatter(acc, ident)
    for i in indices_to_annotate:
        plt.annotate(f"p={p[i]:.2f},σ={s[i]:.2f}", (acc[i], ident[i]))
    plt.xlabel("Accuracy")
    plt.ylabel("Identifiability proxy (min|H|/max|H|)")
    plt.yscale('log')
    plt.title("High accuracy can coexist with identifiability collapse")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_accuracy_vs_identifiability.png", dpi=200)
    plt.close()

    # Param error vs noise
    plt.figure()
    plt.scatter(p + s, perr)
    for i in indices_to_annotate:
        plt.annotate(f"p={p[i]:.2f},σ={s[i]:.2f}", (p[i]+s[i], perr[i]))
    plt.xlabel("Noise index (p + σ)")
    plt.ylabel("Parameter L2 error")
    plt.title("Parameter recovery degrades with noise")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_param_error_vs_noise.png", dpi=200)
    plt.close()
    
    # Fisher condition number vs noise (if enhanced metrics are available)
    if results and 'fisher_condition_number' in results[0]:
        fisher_cond = np.array([r.get('fisher_condition_number', np.nan) for r in results])
        
        # Fisher condition number vs noise level
        plt.figure(figsize=(10, 6))
        noise_level = p + s
        plt.semilogy(noise_level, fisher_cond, 'o-', markersize=8, linewidth=2)

        # Add quality thresholds
        plt.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Excellent (κ<100)')
        plt.axhline(y=1000, color='orange', linestyle='--', alpha=0.5, label='Good (κ<1000)')
        plt.axhline(y=10000, color='r', linestyle='--', alpha=0.5, label='Fair (κ<10000)')

        for i in indices_to_annotate:
            plt.annotate(f"p={p[i]:.2f},σ={s[i]:.2f}",
                        (noise_level[i], fisher_cond[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.xlabel("Noise Level (p + σ)", fontsize=12)
        plt.ylabel("Fisher Condition Number κ(F)", fontsize=12)
        plt.title("Information Geometry Degrades with Noise\nIdentifiability collapses as κ(F) → ∞", fontsize=13)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "fig_fisher_condition_vs_noise.png", dpi=200)
        plt.close()
        
        # Fisher condition vs identifiability proxy (correlation check)
        plt.figure(figsize=(10, 6))
        plt.loglog(ident, fisher_cond, 'o', markersize=10)
        for i in indices_to_annotate:
            plt.annotate(f"p={p[i]:.2f},σ={s[i]:.2f}",
                        (ident[i], fisher_cond[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.xlabel("Identifiability Proxy (min|H|/max|H|)", fontsize=12)
        plt.ylabel("Fisher Condition Number κ(F)", fontsize=12)
        plt.title("Fisher κ(F) Correlates with Identifiability Collapse\nBoth metrics detect ill-conditioning", fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "fig_fisher_vs_identifiability.png", dpi=200)
        plt.close()
        
        # Effective rank vs noise
        if 'fisher_effective_rank' in results[0]:
            eff_rank = np.array([r.get('fisher_effective_rank', np.nan) for r in results])
            param_dim = results[0].get('fisher_effective_dimension', 8)  # fallback
            
            plt.figure(figsize=(10, 6))
            plt.plot(noise_level, eff_rank, 'o-', markersize=8, linewidth=2, label='Effective Rank')
            plt.axhline(y=param_dim, color='g', linestyle='--', alpha=0.5, label=f'Full Rank (d={param_dim})')

            for i in indices_to_annotate:
                plt.annotate(f"p={p[i]:.2f},σ={s[i]:.2f}",
                            (noise_level[i], eff_rank[i]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

            plt.xlabel("Noise Level (p + σ)", fontsize=12)
            plt.ylabel("Effective Rank (Participation Ratio)", fontsize=12)
            plt.title("Dimensional Collapse Under Noise\nEffective rank drops as noise increases", fontsize=13)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "fig_effective_rank_vs_noise.png", dpi=200)
            plt.close()
    
    # Extended visualizations
    if extended_viz:
        from .visualizations import (
            plot_noise_heatmap,
            plot_combined_metrics_grid
        )
        
        # Generate heatmaps for each metric
        for metric in ['acc', 'param_l2', 'ident_proxy']:
            plot_noise_heatmap(
                results,
                output_dir / f"heatmap_{metric}.png",
                metric=metric
            )
        
        # Generate combined metrics grid
        plot_combined_metrics_grid(
            results,
            output_dir / "combined_metrics.png"
        )
    
    # Interactive visualizations
    if interactive:
        try:
            from .interactive_viz import (
                create_interactive_noise_heatmap,
                create_interactive_metrics_dashboard
            )
            
            create_interactive_noise_heatmap(
                results,
                output_dir / "interactive_heatmaps.html"
            )
            
            create_interactive_metrics_dashboard(
                results,
                output_dir / "interactive_dashboard.html"
            )
        except ImportError:
            print("Warning: plotly not available for interactive visualizations")
