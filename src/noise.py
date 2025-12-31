import numpy as np

def apply_depolarizing(p: float, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simple proxy noise channel: with prob p, replace state with random unit vector.
    This is a deliberately minimal placeholder to keep dependencies minimal.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")
    if rng.random() < p:
        v = rng.normal(size=state.shape)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v
    return state

def apply_phase_noise(sigma: float, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Another lightweight proxy: multiply by a random phase-like perturbation.
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    return state + rng.normal(scale=sigma, size=state.shape)
