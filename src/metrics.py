import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float((y_true == y_pred).mean())

def l2_param_error(theta_hat: np.ndarray, theta_true: np.ndarray) -> float:
    theta_hat = np.asarray(theta_hat, dtype=float)
    theta_true = np.asarray(theta_true, dtype=float)
    return float(np.linalg.norm(theta_hat - theta_true))

def identifiability_proxy(hessian_diag: np.ndarray, eps: float = 1e-12) -> float:
    """
    Proxy: inverse condition-like measure using diagonal Hessian magnitude.
    Low values ~ flat directions ~ poor identifiability.
    """
    h = np.asarray(hessian_diag, dtype=float)
    return float(np.min(np.abs(h)) / (np.max(np.abs(h)) + eps))
