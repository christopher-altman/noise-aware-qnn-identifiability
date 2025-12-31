import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional
from scipy.linalg import svd, eigh
import warnings


def compute_fisher_information_matrix(
    model_fn: Callable,
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Compute Fisher Information Matrix (FIM) using empirical approximation.
    
    For a binary classification model with outputs p(y=1|x,θ), the FIM is:
    F = E[(∇_θ log p(y|x,θ))(∇_θ log p(y|x,θ))^T]
    
    We approximate this empirically over the dataset.
    
    Args:
        model_fn: Function that takes (x, theta) and returns probability p(y=1|x,θ)
        theta: Current parameter vector
        X: Input data (n, d)
        y: Binary labels (n,)
        eps: Small constant for numerical stability
        
    Returns:
        Fisher Information Matrix (d, d)
    """
    n, d = X.shape[0], len(theta)
    
    # Compute score vectors (gradient of log-likelihood) for each data point
    scores = []
    
    for i in range(n):
        x_i = X[i]
        y_i = y[i]
        
        # Compute gradient of log p(y_i|x_i, theta)
        # For binary classification: log p(y|x) = y*log(p) + (1-y)*log(1-p)
        # ∇_θ log p = y * ∇p/p - (1-y) * ∇p/(1-p)
        
        # Compute gradient of model output w.r.t. theta using finite differences
        grad_p = np.zeros(d)
        p = model_fn(x_i, theta)
        p = np.clip(p, eps, 1 - eps)  # Numerical stability
        
        for j in range(d):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            p_plus = model_fn(x_i, theta_plus)
            p_plus = np.clip(p_plus, eps, 1 - eps)
            
            grad_p[j] = (p_plus - p) / eps
        
        # Compute score (gradient of log-likelihood)
        if y_i == 1:
            score = grad_p / p
        else:
            score = -grad_p / (1 - p)
        
        scores.append(score)
    
    scores = np.array(scores)
    
    # Empirical FIM: average of outer products of scores
    fim = np.zeros((d, d))
    for score in scores:
        fim += np.outer(score, score)
    fim /= n
    
    return fim


def compute_empirical_fisher(
    model_fn: Callable,
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: Optional[int] = None
) -> np.ndarray:
    """
    Compute empirical Fisher Information Matrix (more efficient for large datasets).
    
    Uses the identity: F ≈ H (Hessian of negative log-likelihood at optimum)
    This is the observed Fisher information.
    
    Args:
        model_fn: Model function
        theta: Parameters
        X: Data
        y: Labels
        batch_size: Optional batch size for computation
        
    Returns:
        Empirical Fisher matrix
    """
    n = len(X)
    if batch_size is None:
        batch_size = min(n, 100)
    
    # Use subset for efficiency
    indices = np.random.choice(n, min(batch_size, n), replace=False)
    X_batch = X[indices]
    y_batch = y[indices]
    
    return compute_fisher_information_matrix(model_fn, theta, X_batch, y_batch)


def compute_condition_number(matrix: np.ndarray, method: str = 'svd') -> float:
    """
    Compute condition number of a matrix.
    
    Args:
        matrix: Square matrix
        method: 'svd' or 'eigenvalue'
        
    Returns:
        Condition number (ratio of largest to smallest singular/eigenvalue)
    """
    if method == 'svd':
        try:
            singular_values = svd(matrix, compute_uv=False)
            singular_values = singular_values[singular_values > 1e-12]
            
            if len(singular_values) == 0:
                return np.inf
            if len(singular_values) == 1:
                return singular_values[0] / 1e-12  # Single non-zero singular value
            
            return singular_values[0] / singular_values[-1]
        except (ValueError, IndexError) as e:
            warnings.warn(f"SVD-based condition number failed: {e}")
            return np.inf
    
    elif method == 'eigenvalue':
        try:
            eigenvalues = np.abs(eigh(matrix)[0])
            eigenvalues = eigenvalues[eigenvalues > 1e-12]
            
            if len(eigenvalues) == 0:
                return np.inf
            if len(eigenvalues) == 1:
                return eigenvalues[0] / 1e-12  # Single non-zero eigenvalue
            
            return eigenvalues[-1] / eigenvalues[0]
        except (ValueError, IndexError) as e:
            warnings.warn(f"Eigenvalue-based condition number failed: {e}")
            return np.inf
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_effective_rank(matrix: np.ndarray, threshold: float = 1e-10) -> float:
    """
    Compute effective rank using participation ratio.
    
    The participation ratio is defined as:
    PR = (Σ λ_i)^2 / (Σ λ_i^2)
    
    where λ_i are the eigenvalues. This measures how many dimensions
    actually carry significant information.
    
    Args:
        matrix: Square matrix
        threshold: Threshold for considering eigenvalues as non-zero
        
    Returns:
        Effective rank (participation ratio)
    """
    eigenvalues = eigh(matrix)[0]
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
    eigenvalues = eigenvalues[eigenvalues > threshold]
    
    if len(eigenvalues) == 0:
        return 0.0
    
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues ** 2)
    
    if sum_eig_sq < 1e-12:
        return 0.0
    
    return (sum_eig ** 2) / sum_eig_sq


def compute_effective_dimension(matrix: np.ndarray, threshold: float = 0.99) -> int:
    """
    Compute effective dimension based on eigenvalue spectrum.
    
    Returns the number of eigenvalues needed to capture `threshold`
    fraction of the total spectral mass.
    
    Args:
        matrix: Square matrix
        threshold: Fraction of spectral mass to capture (default: 0.99)
        
    Returns:
        Effective dimension
    """
    eigenvalues = eigh(matrix)[0]
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    
    total_mass = np.sum(eigenvalues)
    
    if total_mass < 1e-12:
        return 0
    
    cumsum = np.cumsum(eigenvalues)
    fraction = cumsum / total_mass
    
    effective_dim = np.searchsorted(fraction, threshold) + 1
    
    return min(effective_dim, len(eigenvalues))


def compute_fisher_trace(fisher_matrix: np.ndarray) -> float:
    """
    Compute trace of Fisher Information Matrix (total information mass).
    
    Args:
        fisher_matrix: Fisher Information Matrix
        
    Returns:
        Trace (sum of eigenvalues)
    """
    return np.trace(fisher_matrix)


def compute_gradient_stability(
    gradients: np.ndarray,
    window_size: int = 10
) -> Dict[str, float]:
    """
    Compute stability metrics for gradient trajectory.
    
    Args:
        gradients: Array of gradients over time (T, d)
        window_size: Window for computing stability
        
    Returns:
        Dictionary with stability metrics
    """
    if len(gradients) < window_size:
        return {
            'mean_grad_norm': np.mean(np.linalg.norm(gradients, axis=1)),
            'grad_norm_std': np.std(np.linalg.norm(gradients, axis=1)),
            'grad_cosine_similarity': 0.0,
            'grad_angle_variance': 0.0
        }
    
    grad_norms = np.linalg.norm(gradients, axis=1)
    
    # Cosine similarity between consecutive gradients
    cosine_sims = []
    for i in range(len(gradients) - 1):
        g1 = gradients[i]
        g2 = gradients[i + 1]
        
        norm1 = np.linalg.norm(g1)
        norm2 = np.linalg.norm(g2)
        
        if norm1 > 1e-12 and norm2 > 1e-12:
            cos_sim = np.dot(g1, g2) / (norm1 * norm2)
            cosine_sims.append(cos_sim)
    
    # Angle variance (in degrees)
    angles = np.arccos(np.clip(cosine_sims, -1.0, 1.0)) * 180 / np.pi
    
    return {
        'mean_grad_norm': float(np.mean(grad_norms)),
        'grad_norm_std': float(np.std(grad_norms)),
        'grad_cosine_similarity': float(np.mean(cosine_sims)) if cosine_sims else 0.0,
        'grad_angle_variance': float(np.var(angles)) if len(angles) > 0 else 0.0
    }


def compute_loss_hessian_diagonal(
    loss_fn: Callable,
    theta: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Compute diagonal of Hessian using finite differences.
    
    Args:
        loss_fn: Loss function
        theta: Parameters
        eps: Finite difference step
        
    Returns:
        Diagonal of Hessian
    """
    d = len(theta)
    hess_diag = np.zeros(d)
    
    base_loss = loss_fn(theta)
    
    for i in range(d):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        
        theta_minus = theta.copy()
        theta_minus[i] -= eps
        
        loss_plus = loss_fn(theta_plus)
        loss_minus = loss_fn(theta_minus)
        
        hess_diag[i] = (loss_plus - 2 * base_loss + loss_minus) / (eps ** 2)
    
    return hess_diag


def analyze_identifiability_geometry(
    fisher_matrix: np.ndarray,
    hessian_diag: np.ndarray
) -> Dict[str, Any]:
    """
    Comprehensive analysis of identifiability from information geometry.
    
    This provides the rigorous mathematical backing for identifiability collapse.
    
    Args:
        fisher_matrix: Fisher Information Matrix
        hessian_diag: Diagonal of loss Hessian
        
    Returns:
        Dictionary with comprehensive metrics
    """
    # Fisher Information metrics
    fisher_trace = compute_fisher_trace(fisher_matrix)
    fisher_condition = compute_condition_number(fisher_matrix, method='eigenvalue')
    fisher_effective_rank = compute_effective_rank(fisher_matrix)
    fisher_effective_dim = compute_effective_dimension(fisher_matrix, threshold=0.99)
    
    # Eigenvalue spectrum analysis
    eigenvalues = eigh(fisher_matrix)[0]
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    
    # Hessian metrics
    nonzero_hess = hessian_diag[np.abs(hessian_diag) > 1e-12]
    if len(nonzero_hess) > 0:
        hessian_condition = np.max(np.abs(hessian_diag)) / np.min(np.abs(nonzero_hess))
        hessian_min = float(np.min(np.abs(nonzero_hess)))
    else:
        hessian_condition = np.inf
        hessian_min = 0.0
    
    # Compute participation ratio more explicitly
    if len(eigenvalues) > 0:
        participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    else:
        participation_ratio = 0.0
    
    # Spectral gap (indicator of ill-conditioning)
    if len(eigenvalues) >= 2:
        spectral_gap = eigenvalues[0] / (eigenvalues[-1] + 1e-12)
    else:
        spectral_gap = np.inf
    
    return {
        # Fisher Information metrics
        'fisher_trace': float(fisher_trace),
        'fisher_condition_number': float(fisher_condition),
        'fisher_effective_rank': float(fisher_effective_rank),
        'fisher_effective_dimension': int(fisher_effective_dim),
        'fisher_participation_ratio': float(participation_ratio),
        'fisher_spectral_gap': float(spectral_gap),
        'fisher_eigenvalue_max': float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0,
        'fisher_eigenvalue_min': float(eigenvalues[-1]) if len(eigenvalues) > 0 else 0.0,
        
        # Hessian metrics
        'hessian_condition_number': float(hessian_condition),
        'hessian_max': float(np.max(np.abs(hessian_diag))),
        'hessian_min': hessian_min,
        
        # Identifiability proxy (legacy)
        'identifiability_proxy': float(np.min(np.abs(hessian_diag)) / (np.max(np.abs(hessian_diag)) + 1e-12)),
        
        # Summary verdict
        'is_well_conditioned': bool(fisher_condition < 1000),
        'is_identifiable': bool(fisher_effective_rank > 0.5 * len(hessian_diag)),
        'information_geometry_quality': _assess_geometry_quality(fisher_condition, participation_ratio)
    }


def _assess_geometry_quality(condition_number: float, participation_ratio: float) -> str:
    """
    Assess quality of information geometry.
    
    Args:
        condition_number: Condition number of Fisher matrix
        participation_ratio: Participation ratio
        
    Returns:
        Quality assessment string
    """
    if condition_number < 100 and participation_ratio > 0.8:
        return "excellent"
    elif condition_number < 1000 and participation_ratio > 0.5:
        return "good"
    elif condition_number < 10000 and participation_ratio > 0.3:
        return "fair"
    else:
        return "poor_ill_conditioned"


def compute_all_enhanced_metrics(
    model_fn: Callable,
    loss_fn: Callable,
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    hessian_eps: float = 1e-5,
    batch_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute all enhanced identifiability metrics.
    
    This is the main function to call for comprehensive analysis.
    
    Args:
        model_fn: Model function (x, theta) -> p(y=1|x,theta)
        loss_fn: Loss function (theta) -> loss
        theta: Current parameters
        X: Input data
        y: Labels
        hessian_eps: Epsilon for Hessian finite differences
        batch_size: Batch size for Fisher computation
        
    Returns:
        Dictionary with all metrics
    """
    try:
        # Compute Fisher Information Matrix
        fisher_matrix = compute_empirical_fisher(model_fn, theta, X, y, batch_size)
        
        # Add small regularization to avoid numerical issues
        d = len(theta)
        fisher_matrix += np.eye(d) * 1e-10
        
        # Compute Hessian diagonal
        hessian_diag = compute_loss_hessian_diagonal(loss_fn, theta, hessian_eps)
        
        # Comprehensive analysis
        metrics = analyze_identifiability_geometry(fisher_matrix, hessian_diag)
        
        return metrics
    except Exception as e:
        # Return NaN metrics on failure
        warnings.warn(f"Enhanced metrics computation failed: {e}")
        d = len(theta)
        return {
            'fisher_trace': np.nan,
            'fisher_condition_number': np.nan,
            'fisher_effective_rank': np.nan,
            'fisher_effective_dimension': 0,
            'fisher_participation_ratio': np.nan,
            'fisher_spectral_gap': np.nan,
            'fisher_eigenvalue_max': np.nan,
            'fisher_eigenvalue_min': np.nan,
            'hessian_condition_number': np.nan,
            'hessian_max': np.nan,
            'hessian_min': np.nan,
            'identifiability_proxy': np.nan,
            'is_well_conditioned': False,
            'is_identifiable': False,
            'information_geometry_quality': 'unknown'
        }


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """
    Format metrics into human-readable report.
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        Formatted string report
    """
    report = []
    report.append("="*60)
    report.append("IDENTIFIABILITY ANALYSIS: Information Geometry")
    report.append("="*60)
    report.append("")
    
    report.append("Fisher Information Matrix:")
    report.append(f"  Trace (information mass):    {metrics['fisher_trace']:.4e}")
    report.append(f"  Condition number:            {metrics['fisher_condition_number']:.4e}")
    report.append(f"  Effective rank:              {metrics['fisher_effective_rank']:.2f}")
    report.append(f"  Effective dimension:         {metrics['fisher_effective_dimension']}")
    report.append(f"  Participation ratio:         {metrics['fisher_participation_ratio']:.4f}")
    report.append(f"  Spectral gap:                {metrics['fisher_spectral_gap']:.4e}")
    report.append("")
    
    report.append("Loss Hessian:")
    report.append(f"  Condition number:            {metrics['hessian_condition_number']:.4e}")
    report.append(f"  Max curvature:               {metrics['hessian_max']:.4e}")
    report.append(f"  Min curvature:               {metrics['hessian_min']:.4e}")
    report.append("")
    
    report.append("Assessment:")
    report.append(f"  Well-conditioned:            {metrics['is_well_conditioned']}")
    report.append(f"  Identifiable:                {metrics['is_identifiable']}")
    report.append(f"  Geometry quality:            {metrics['information_geometry_quality']}")
    report.append("")
    
    report.append("="*60)
    report.append("INTERPRETATION:")
    
    if metrics['information_geometry_quality'] == 'poor_ill_conditioned':
        report.append("⚠️  The information geometry is ILL-CONDITIONED.")
        report.append("    Identifiability has collapsed due to flat loss directions.")
        report.append("    Parameters are not uniquely recoverable from data.")
    elif metrics['information_geometry_quality'] in ['good', 'excellent']:
        report.append("✓  The information geometry is WELL-CONDITIONED.")
        report.append("   Parameters are identifiable from the data.")
    else:
        report.append("⚠  The information geometry is MARGINALLY conditioned.")
        report.append("   Some parameter directions may be weakly identifiable.")
    
    report.append("="*60)
    
    return "\n".join(report)
