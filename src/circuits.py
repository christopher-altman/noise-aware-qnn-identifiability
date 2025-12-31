import numpy as np

def feature_map(x: np.ndarray) -> np.ndarray:
    """
    Minimal feature map: normalize and return.
    Placeholder for a real quantum embedding.
    """
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x) + 1e-12
    return x / n

def qnn_forward(phi: np.ndarray, theta: np.ndarray) -> float:
    """
    Minimal 'QNN-like' forward function:
    score = dot(phi, normalized(theta)) in [-1,1].
    """
    t = theta / (np.linalg.norm(theta) + 1e-12)
    return float(np.dot(phi, t))
