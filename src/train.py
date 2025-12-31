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

def run_experiment(seed: int = 0):
    rng = np.random.default_rng(seed)

    # Problem size deliberately small for fast runs
    n, d = 512, 8
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

    noise_grid = [(0.0, 0.0), (0.05, 0.0), (0.10, 0.0), (0.10, 0.10), (0.20, 0.20)]
    results = []

    for (p_dep, sigma_phase) in noise_grid:
        best_theta = None
        best_loss = 1e9

        for _ in range(2000):
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

        hdiag = finite_diff_hessian_diag(loss_fn, best_theta, eps=1e-3)
        ident = identifiability_proxy(hdiag)

        results.append({
            "p_dep": p_dep,
            "sigma_phase": sigma_phase,
            "acc": acc,
            "param_l2": perr,
            "ident_proxy": ident,
        })

    plot_results(results)
    print("Wrote: fig_accuracy_vs_identifiability.png, fig_param_error_vs_noise.png")
    for r in results:
        print(r)
    return results
