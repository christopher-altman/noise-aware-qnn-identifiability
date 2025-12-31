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


def apply_correlated_noise(gamma: float, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Correlated amplitude damping-like noise.
    
    Models systematic errors where all feature dimensions are affected by
    a common environmental factor (e.g., temperature fluctuations, power supply noise).
    
    This demonstrates generality beyond independent noise models.
    
    Args:
        gamma: Correlation strength in [0, 1]
               - 0: no damping (identity)
               - 1: complete damping (collapse toward origin)
        state: Input state vector
        rng: Random number generator
    
    Returns:
        Damped state with correlated perturbation
    
    Mathematical model:
        state_out = sqrt(1 - gamma) * state + sqrt(gamma) * v
        where v is a random direction sampled once and applied to all dimensions
    
    Physical interpretation:
        - Amplitude damping toward a common attractor
        - Correlated decoherence across feature space
        - Systematic drift in embedding space
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1].")
    
    if gamma == 0.0:
        return state
    
    # Sample a single random direction (correlated across all dimensions)
    d = len(state)
    common_direction = rng.normal(size=d)
    common_direction = common_direction / (np.linalg.norm(common_direction) + 1e-12)
    
    # Amplitude damping with correlation
    damped = np.sqrt(1.0 - gamma) * state + np.sqrt(gamma) * common_direction
    
    return damped
