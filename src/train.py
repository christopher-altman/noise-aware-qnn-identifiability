import numpy as np
from .circuits import feature_map, qnn_forward
from .noise import apply_depolarizing, apply_phase_noise
from .metrics import accuracy, l2_param_error, identifiability_proxy
from .plots import plot_results

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
    verbose: bool = False,
    quiet: bool = False,
):
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Generating dataset with n={n}, d={d}...")
    
    theta_true = rng.normal(size=d)
    theta_true = theta_true / (np.linalg.norm(theta_true) + 1e-12)

    X, y = make_dataset(n, d, theta_true, rng)

    # "Training": we do a tiny random search to keep deps minimal (no torch)
    # Later you can swap this with a real optimizer.
    def predict(theta, p_dep, sigma_phase):
        preds = []
        for i in range(n):
            phi = feature_map(X[i])
            # inject proxy noise into the embedding vector
            phi_n = apply_depolarizing(p_dep, phi, rng)
            phi_n = apply_phase_noise(sigma_phase, phi_n, rng)
            score = qnn_forward(phi_n, theta)
            preds.append(1 if score >= 0 else 0)
        return np.array(preds, dtype=int)

    def loss(theta, p_dep, sigma_phase):
        yhat = predict(theta, p_dep, sigma_phase)
        # simple 0-1 loss proxy
        return 1.0 - accuracy(y, yhat)

    if noise_grid is None:
        noise_grid = [(0.0, 0.0), (0.05, 0.0), (0.10, 0.0), (0.10, 0.10), (0.20, 0.20)]
    
    results = []

    for idx, (p_dep, sigma_phase) in enumerate(noise_grid, 1):
        if verbose:
            print(f"\nNoise setting {idx}/{len(noise_grid)}: p={p_dep:.2f}, σ={sigma_phase:.2f}")
        
        best_theta = None
        best_loss = 1e9

        for i in range(optimizer_iterations):
            if verbose and (i + 1) % 500 == 0:
                print(f"  Iteration {i + 1}/{optimizer_iterations}, best loss: {best_loss:.4f}")
            cand = rng.normal(size=d)
            cand = cand / (np.linalg.norm(cand) + 1e-12)
            l = loss(cand, p_dep, sigma_phase)
            if l < best_loss:
                best_loss = l
                best_theta = cand

        acc = 1.0 - best_loss
        perr = l2_param_error(best_theta, theta_true)

        def loss_fn(t):
            return loss(t, p_dep, sigma_phase)

        if verbose:
            print(f"  Computing Hessian diagonal with eps={hessian_eps}...")
        
        hdiag = finite_diff_hessian_diag(loss_fn, best_theta, eps=hessian_eps)
        ident = identifiability_proxy(hdiag)
        
        if verbose:
            print(f"  Results: acc={acc:.4f}, param_err={perr:.4f}, ident={ident:.2e}")

        results.append({
            "p_dep": p_dep,
            "sigma_phase": sigma_phase,
            "acc": acc,
            "param_l2": perr,
            "ident_proxy": ident,
        })

    if generate_plots:
        if verbose:
            print("\nGenerating plots...")
        from pathlib import Path
        plot_output_dir = Path(output_dir) if output_dir is not None else Path('.')
        plot_results(results, output_dir=plot_output_dir)
        if not quiet:
            print(f"\nPlots saved to: {plot_output_dir}/")
            print(f"  - fig_accuracy_vs_identifiability.png")
            print(f"  - fig_param_error_vs_noise.png")
    
    return results
