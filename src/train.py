import numpy as np
import time
import sys
from .circuits import feature_map, qnn_forward
from .noise import apply_depolarizing, apply_phase_noise, apply_correlated_noise
from .metrics import accuracy, l2_param_error, identifiability_proxy
from .plots import plot_results

# Optional tqdm import (terminal-safe)
try:
    from tqdm import tqdm  # terminal-safe, not tqdm.auto
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

def make_dataset(n: int, d: int, theta_true: np.ndarray, rng: np.random.Generator):
    X = rng.normal(size=(n, d))
    y = []
    for i in range(n):
        phi = feature_map(X[i])
        score = qnn_forward(phi, theta_true)
        y.append(1 if score >= 0 else 0)
    return X, np.array(y, dtype=int)

def finite_diff_hessian_diag(loss_fn, theta: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Very small, very explicit Hessian diagonal estimate (O(d)) for identifiability proxy.
    """
    d = theta.size
    hdiag = np.zeros(d)
    base = loss_fn(theta)
    for j in range(d):
        e = np.zeros(d); e[j] = 1.0
        l1 = loss_fn(theta + eps*e)
        l2 = loss_fn(theta - eps*e)
        hdiag[j] = (l1 - 2*base + l2) / (eps**2)
    return hdiag

def run_experiment(
    seed: int = 0,
    n: int = 512,
    d: int = 8,
    noise_grid: list = None,
    optimizer_iterations: int = 2000,
    hessian_eps: float = 1e-3,
    output_dir = None,
    generate_plots: bool = True,
    extended_viz: bool = False,
    interactive: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    enhanced_metrics: bool = False,
    fisher_batch_size: int = None,
):
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Generating dataset with n={n}, d={d}...")
    
    theta_true = rng.normal(size=d)
    theta_true = theta_true / (np.linalg.norm(theta_true) + 1e-12)

    X, y = make_dataset(n, d, theta_true, rng)

    # "Training": we do a tiny random search to keep deps minimal (no torch)
    # Later you can swap this with a real optimizer.
    def predict(theta, p_dep, sigma_phase, gamma_corr=0.0):
        preds = []
        for i in range(n):
            phi = feature_map(X[i])
            # inject proxy noise into the embedding vector
            phi_n = apply_depolarizing(p_dep, phi, rng)
            phi_n = apply_phase_noise(sigma_phase, phi_n, rng)
            phi_n = apply_correlated_noise(gamma_corr, phi_n, rng)
            score = qnn_forward(phi_n, theta)
            preds.append(1 if score >= 0 else 0)
        return np.array(preds, dtype=int)

    def loss(theta, p_dep, sigma_phase, gamma_corr=0.0):
        yhat = predict(theta, p_dep, sigma_phase, gamma_corr)
        # simple 0-1 loss proxy
        return 1.0 - accuracy(y, yhat)

    if noise_grid is None:
        noise_grid = [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.10, 0.0, 0.0), (0.10, 0.10, 0.0), (0.20, 0.20, 0.0)]

    results = []

    # Determine progress display mode
    show_progress = not quiet and TQDM_AVAILABLE
    fallback_progress = not quiet and not TQDM_AVAILABLE

    # Wrap noise_grid with tqdm if available and not quiet
    if show_progress:
        pbar = tqdm(
            noise_grid,
            desc="Noise settings",
            unit="setting",
            dynamic_ncols=True,
            leave=True,
            file=sys.stderr,  # plays nicer in many terminals
        )
    else:
        pbar = noise_grid

    # Time heartbeat tracking
    last_beat = time.time()

    for idx, noise_params in enumerate(pbar, 1):
        # Heartbeat: periodic time update (every 30s)
        now = time.time()
        if not quiet and (now - last_beat) > 30:
            print(f"[heartbeat] still working… ({idx}/{len(noise_grid)})", flush=True)
            last_beat = now
        # Support both 2-param (backward compat) and 3-param noise models
        if len(noise_params) == 2:
            p_dep, sigma_phase = noise_params
            gamma_corr = 0.0
        else:
            p_dep, sigma_phase, gamma_corr = noise_params
        
        # Heartbeat: noise setting boundary
        if fallback_progress:
            if gamma_corr > 0:
                print(f"[{idx}/{len(noise_grid)}] p={p_dep:.2f}, σ={sigma_phase:.2f}, γ={gamma_corr:.2f}")
            else:
                print(f"[{idx}/{len(noise_grid)}] p={p_dep:.2f}, σ={sigma_phase:.2f}")

        if verbose:
            if gamma_corr > 0:
                print(f"\nNoise setting {idx}/{len(noise_grid)}: p={p_dep:.2f}, σ={sigma_phase:.2f}, γ={gamma_corr:.2f}")
            else:
                print(f"\nNoise setting {idx}/{len(noise_grid)}: p={p_dep:.2f}, σ={sigma_phase:.2f}")

        best_theta = None
        best_loss = 1e9

        for i in range(optimizer_iterations):
            if verbose and (i + 1) % 500 == 0:
                print(f"  Iteration {i + 1}/{optimizer_iterations}, best loss: {best_loss:.4f}")
            cand = rng.normal(size=d)
            cand = cand / (np.linalg.norm(cand) + 1e-12)
            l = loss(cand, p_dep, sigma_phase, gamma_corr)
            if l < best_loss:
                best_loss = l
                best_theta = cand

        acc = 1.0 - best_loss
        perr = l2_param_error(best_theta, theta_true)

        def loss_fn(t):
            return loss(t, p_dep, sigma_phase, gamma_corr)

        # Heartbeat: Hessian computation
        t0_hess = time.time()
        if verbose:
            print(f"  Computing Hessian diagonal with eps={hessian_eps}...")

        hdiag = finite_diff_hessian_diag(loss_fn, best_theta, eps=hessian_eps)
        ident = identifiability_proxy(hdiag)

        t_hess = time.time() - t0_hess
        if verbose:
            print(f"  Hessian computed in {t_hess:.2f}s")
        
        # Compute enhanced metrics if requested
        enhanced_result = {}
        if enhanced_metrics:
            # Heartbeat: Fisher computation start
            t0_fisher = time.time()
            if verbose:
                print(f"  Computing Fisher Information Matrix...")

            # Create model and loss function wrappers for Fisher computation
            def model_fn(x, theta):
                """Model function: returns p(y=1|x,theta) with current noise."""
                phi = feature_map(x)
                phi_n = apply_depolarizing(p_dep, phi, rng)
                phi_n = apply_phase_noise(sigma_phase, phi_n, rng)
                phi_n = apply_correlated_noise(gamma_corr, phi_n, rng)
                score = qnn_forward(phi_n, theta)
                # Convert score to probability using logistic function
                return 1.0 / (1.0 + np.exp(-score))
            
            def loss_fn_fisher(theta_param):
                """Loss function for Fisher computation."""
                return loss(theta_param, p_dep, sigma_phase, gamma_corr)
            
            try:
                from .enhanced_metrics import compute_all_enhanced_metrics
                
                enhanced_result = compute_all_enhanced_metrics(
                    model_fn=model_fn,
                    loss_fn=loss_fn_fisher,
                    theta=best_theta,
                    X=X,
                    y=y,
                    hessian_eps=hessian_eps,
                    batch_size=fisher_batch_size
                )

                t_fisher = time.time() - t0_fisher
                if verbose:
                    print(f"  Fisher computed in {t_fisher:.2f}s")
                    print(f"  Fisher κ(F): {enhanced_result['fisher_condition_number']:.2e}")
                    print(f"  Effective rank: {enhanced_result['fisher_effective_rank']:.2f}")
                    print(f"  Quality: {enhanced_result['information_geometry_quality']}")
            
            except ImportError:
                if verbose:
                    print("  Warning: Enhanced metrics module not available")
                enhanced_result = {}
            except Exception as e:
                if verbose:
                    print(f"  Warning: Fisher computation failed: {e}")
                enhanced_result = {}
        
        if verbose:
            print(f"  Results: acc={acc:.4f}, param_err={perr:.4f}, ident={ident:.2e}")

        # Update tqdm postfix with metrics
        if show_progress:
            postfix_dict = {
                'acc': f'{acc:.3f}',
                'ident': f'{ident:.1e}'
            }
            if enhanced_result and 'fisher_condition_number' in enhanced_result:
                postfix_dict['κ(F)'] = f'{enhanced_result["fisher_condition_number"]:.1e}'
            try:
                pbar.set_postfix(postfix_dict)
            except:
                pass

        result_dict = {
            "p_dep": p_dep,
            "sigma_phase": sigma_phase,
            "gamma_corr": gamma_corr,
            "acc": acc,
            "param_l2": perr,
            "ident_proxy": ident,
        }
        
        # Add enhanced metrics to results if computed
        if enhanced_result:
            result_dict.update(enhanced_result)
        
        results.append(result_dict)

    if generate_plots:
        # Heartbeat: plot generation
        t0_plots = time.time()
        if verbose:
            print("\nGenerating plots...")
        from pathlib import Path
        # Determine base output directory
        base_output_dir = Path(output_dir) if output_dir is not None else Path('artifacts/latest')
        # All figures go into figs/ subdirectory
        plot_output_dir = base_output_dir / 'figs'
        plot_output_dir.mkdir(parents=True, exist_ok=True)
        plot_results(results, output_dir=plot_output_dir, extended_viz=extended_viz, interactive=interactive)
        t_plots = time.time() - t0_plots
        if verbose:
            print(f"Plots generated in {t_plots:.2f}s")
        if not quiet:
            print(f"\nPlots saved to: {plot_output_dir}/")
            print(f"  - {plot_output_dir}/fig_accuracy_vs_identifiability.png")
            print(f"  - {plot_output_dir}/fig_param_error_vs_noise.png")
            if enhanced_metrics:
                print(f"  - fig_fisher_condition_vs_noise.png")
                print(f"  - fig_fisher_vs_identifiability.png")
                print(f"  - fig_effective_rank_vs_noise.png")
            if extended_viz:
                print(f"  - heatmap_*.png (3 files)")
                print(f"  - combined_metrics.png")
            if interactive:
                print(f"  - interactive_heatmaps.html")
                print(f"  - interactive_dashboard.html")
    
    return results
