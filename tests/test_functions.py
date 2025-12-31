import numpy as np
import pytest
from src.circuits import feature_map, qnn_forward
from src.noise import apply_depolarizing, apply_phase_noise, apply_correlated_noise
from src.train import finite_diff_hessian_diag


class TestFeatureMap:
    """Tests for feature_map normalization."""
    
    def test_feature_map_normalizes_simple_vector(self):
        """Test that feature_map correctly normalizes a simple input vector."""
        x = np.array([3.0, 4.0])
        result = feature_map(x)
        
        # Check that the result is normalized (norm = 1)
        assert np.isclose(np.linalg.norm(result), 1.0)
        
        # Check that the direction is preserved
        expected = np.array([3.0, 4.0]) / 5.0
        assert np.allclose(result, expected)
    
    def test_feature_map_normalizes_zero_vector(self):
        """Test that feature_map handles zero vector without division by zero."""
        x = np.array([0.0, 0.0, 0.0])
        result = feature_map(x)
        
        # Should not raise an error and should return zero vector
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_feature_map_normalizes_arbitrary_vector(self):
        """Test that feature_map normalizes an arbitrary vector."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = feature_map(x)
        
        # Check normalization
        assert np.isclose(np.linalg.norm(result), 1.0)
        
        # Check direction is preserved
        norm = np.linalg.norm(x)
        expected = x / norm
        assert np.allclose(result, expected)
    
    def test_feature_map_with_negative_values(self):
        """Test that feature_map correctly normalizes vectors with negative values."""
        x = np.array([-1.0, 1.0, -1.0, 1.0])
        result = feature_map(x)
        
        assert np.isclose(np.linalg.norm(result), 1.0)


class TestQnnForward:
    """Tests for qnn_forward dot product computation."""
    
    def test_qnn_forward_computes_dot_product(self):
        """Test that qnn_forward computes dot product of feature and normalized parameter vector."""
        phi = np.array([1.0, 0.0, 0.0])
        theta = np.array([2.0, 0.0, 0.0])
        
        result = qnn_forward(phi, theta)
        
        # theta normalized is [1, 0, 0], dot with phi is 1.0
        assert np.isclose(result, 1.0)
    
    def test_qnn_forward_with_orthogonal_vectors(self):
        """Test qnn_forward with orthogonal vectors returns 0."""
        phi = np.array([1.0, 0.0])
        theta = np.array([0.0, 1.0])
        
        result = qnn_forward(phi, theta)
        
        # Orthogonal vectors should give dot product of 0
        assert np.isclose(result, 0.0)
    
    def test_qnn_forward_with_opposite_vectors(self):
        """Test qnn_forward with opposite direction vectors returns -1."""
        phi = np.array([1.0, 0.0, 0.0])
        theta = np.array([-3.0, 0.0, 0.0])
        
        result = qnn_forward(phi, theta)
        
        # Opposite direction should give -1
        assert np.isclose(result, -1.0)
    
    def test_qnn_forward_normalizes_theta(self):
        """Test that qnn_forward normalizes theta before computing dot product."""
        phi = np.array([0.6, 0.8])  # Already normalized
        theta = np.array([3.0, 4.0])  # Not normalized, norm = 5
        
        result = qnn_forward(phi, theta)
        
        # theta normalized is [3/5, 4/5] = [0.6, 0.8]
        # dot product with phi should be 1.0
        expected = np.dot(phi, theta / np.linalg.norm(theta))
        assert np.isclose(result, expected)
        assert np.isclose(result, 1.0)
    
    def test_qnn_forward_output_in_valid_range(self):
        """Test that qnn_forward output is in [-1, 1] range."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            phi = rng.normal(size=5)
            phi = phi / np.linalg.norm(phi)
            theta = rng.normal(size=5)
            
            result = qnn_forward(phi, theta)
            
            assert -1.0 <= result <= 1.0


class TestApplyDepolarizing:
    """Tests for apply_depolarizing noise channel."""
    
    def test_apply_depolarizing_with_zero_probability(self):
        """Test that apply_depolarizing returns original state when p=0."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 2.0, 3.0, 4.0])
        
        result = apply_depolarizing(0.0, state, rng)
        
        # With p=0, state should be unchanged
        assert np.allclose(result, state)
    
    def test_apply_depolarizing_multiple_calls_with_zero_probability(self):
        """Test multiple calls with p=0 always return original state."""
        rng = np.random.default_rng(123)
        state = np.array([0.5, -0.5, 1.0])
        
        for _ in range(100):
            result = apply_depolarizing(0.0, state, rng)
            assert np.allclose(result, state)
    
    def test_apply_depolarizing_with_probability_one(self):
        """Test that apply_depolarizing always replaces state when p=1."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 0.0, 0.0])
        
        result = apply_depolarizing(1.0, state, rng)
        
        # With p=1, state should be replaced (very unlikely to be the same)
        assert not np.allclose(result, state)
        # Result should be normalized
        assert np.isclose(np.linalg.norm(result), 1.0)
    
    def test_apply_depolarizing_validates_probability(self):
        """Test that apply_depolarizing raises error for invalid probability."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 0.0])
        
        with pytest.raises(ValueError, match="p must be in"):
            apply_depolarizing(-0.1, state, rng)
        
        with pytest.raises(ValueError, match="p must be in"):
            apply_depolarizing(1.5, state, rng)
    
    def test_apply_depolarizing_returns_normalized_vector(self):
        """Test that apply_depolarizing returns a normalized vector when noise is applied."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 2.0, 3.0])
        
        # Run multiple times to ensure we hit the noise case
        for _ in range(100):
            result = apply_depolarizing(1.0, state, rng)
            assert np.isclose(np.linalg.norm(result), 1.0)


class TestApplyPhaseNoise:
    """Tests for apply_phase_noise Gaussian perturbation."""
    
    def test_apply_phase_noise_with_zero_sigma(self):
        """Test that apply_phase_noise returns original state when sigma=0."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 2.0, 3.0, 4.0])
        
        result = apply_phase_noise(0.0, state, rng)
        
        # With sigma=0, no noise should be added
        assert np.allclose(result, state)
    
    def test_apply_phase_noise_adds_gaussian_perturbation(self):
        """Test that apply_phase_noise correctly adds Gaussian perturbation."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 0.0, 0.0])
        sigma = 0.1
        
        result = apply_phase_noise(sigma, state, rng)
        
        # Result should be different from original
        assert not np.allclose(result, state)
        
        # The perturbation should be small for small sigma
        difference = result - state
        assert np.linalg.norm(difference) < 1.0  # Reasonable bound
    
    def test_apply_phase_noise_perturbation_scale(self):
        """Test that larger sigma produces larger perturbations on average."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 1.0, 1.0, 1.0])
        
        # Small sigma
        small_sigma = 0.01
        small_differences = []
        for i in range(100):
            rng_temp = np.random.default_rng(42 + i)
            result = apply_phase_noise(small_sigma, state, rng_temp)
            small_differences.append(np.linalg.norm(result - state))
        
        # Large sigma
        large_sigma = 0.5
        large_differences = []
        for i in range(100):
            rng_temp = np.random.default_rng(42 + i)
            result = apply_phase_noise(large_sigma, state, rng_temp)
            large_differences.append(np.linalg.norm(result - state))
        
        # Average perturbation should be larger for larger sigma
        assert np.mean(large_differences) > np.mean(small_differences)
    
    def test_apply_phase_noise_validates_sigma(self):
        """Test that apply_phase_noise raises error for negative sigma."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 0.0])
        
        with pytest.raises(ValueError, match="sigma must be"):
            apply_phase_noise(-0.1, state, rng)
    
    def test_apply_phase_noise_preserves_shape(self):
        """Test that apply_phase_noise preserves the shape of the state."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = apply_phase_noise(0.1, state, rng)
        
        assert result.shape == state.shape


class TestApplyCorrelatedNoise:
    """Tests for apply_correlated_noise amplitude damping."""
    
    def test_apply_correlated_noise_with_zero_gamma(self):
        """Test that apply_correlated_noise returns original state when gamma=0."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 2.0, 3.0, 4.0])
        
        result = apply_correlated_noise(0.0, state, rng)
        
        # With gamma=0, state should be unchanged
        assert np.allclose(result, state)
    
    def test_apply_correlated_noise_with_gamma_one(self):
        """Test that apply_correlated_noise produces damped state when gamma=1."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 0.0, 0.0])
        
        result = apply_correlated_noise(1.0, state, rng)
        
        # With gamma=1, should collapse to random direction
        assert not np.allclose(result, state)
        # Result should still be normalized (sqrt(1-gamma)*state + sqrt(gamma)*v)
        assert np.isclose(np.linalg.norm(result), 1.0, rtol=1e-3)
    
    def test_apply_correlated_noise_intermediate_gamma(self):
        """Test correlated noise with intermediate gamma value."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 1.0, 1.0, 1.0]) / 2.0  # Normalized
        gamma = 0.3
        
        result = apply_correlated_noise(gamma, state, rng)
        
        # Should be different from original
        assert not np.allclose(result, state)
        
        # Should preserve approximate scale
        assert 0.5 < np.linalg.norm(result) < 1.5
    
    def test_apply_correlated_noise_validates_gamma(self):
        """Test that apply_correlated_noise raises error for invalid gamma."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 0.0])
        
        with pytest.raises(ValueError, match="gamma must be in"):
            apply_correlated_noise(-0.1, state, rng)
        
        with pytest.raises(ValueError, match="gamma must be in"):
            apply_correlated_noise(1.5, state, rng)
    
    def test_apply_correlated_noise_correlation_structure(self):
        """Test that correlated noise affects all dimensions in a correlated way."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 0.0, 0.0, 0.0])
        gamma = 0.5
        
        # Apply correlated noise multiple times and observe correlation
        results = []
        for i in range(10):
            rng_temp = np.random.default_rng(100 + i)
            result = apply_correlated_noise(gamma, state, rng_temp)
            results.append(result)
        
        results = np.array(results)
        
        # Correlation structure: dimensions should co-vary
        # (not independent as in phase noise)
        # Check that results are not all identical (stochastic)
        variances = np.var(results, axis=0)
        assert np.all(variances > 1e-6)  # All dimensions vary
    
    def test_apply_correlated_noise_preserves_shape(self):
        """Test that apply_correlated_noise preserves the shape of the state."""
        rng = np.random.default_rng(42)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = apply_correlated_noise(0.3, state, rng)
        
        assert result.shape == state.shape
    
    def test_apply_correlated_noise_reduces_to_random_at_gamma_one(self):
        """Test that gamma=1 produces a random direction independent of input."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)  # Same seed
        
        state1 = np.array([1.0, 0.0, 0.0])
        state2 = np.array([0.0, 1.0, 0.0])  # Different input
        
        result1 = apply_correlated_noise(1.0, state1, rng1)
        result2 = apply_correlated_noise(1.0, state2, rng2)
        
        # With same RNG seed and gamma=1, results should be identical
        # (independent of input state)
        assert np.allclose(result1, result2)


class TestFiniteDiffHessianDiag:
    """Tests for finite_diff_hessian_diag Hessian diagonal calculation."""
    
    def test_finite_diff_hessian_diag_simple_quadratic(self):
        """Test Hessian diagonal calculation for a simple quadratic function."""
        # f(x) = x[0]^2 + 2*x[1]^2 + 3*x[2]^2
        # Hessian diagonal: [2, 4, 6]
        def quadratic_loss(theta):
            return theta[0]**2 + 2*theta[1]**2 + 3*theta[2]**2
        
        theta = np.array([1.0, 1.0, 1.0])
        hdiag = finite_diff_hessian_diag(quadratic_loss, theta, eps=1e-5)
        
        expected = np.array([2.0, 4.0, 6.0])
        assert np.allclose(hdiag, expected, rtol=1e-3)
    
    def test_finite_diff_hessian_diag_at_different_point(self):
        """Test that Hessian diagonal is consistent at different points for quadratic."""
        def quadratic_loss(theta):
            return theta[0]**2 + 2*theta[1]**2
        
        theta1 = np.array([0.0, 0.0])
        theta2 = np.array([5.0, -3.0])
        
        hdiag1 = finite_diff_hessian_diag(quadratic_loss, theta1, eps=1e-5)
        hdiag2 = finite_diff_hessian_diag(quadratic_loss, theta2, eps=1e-5)
        
        # For a quadratic, Hessian should be constant
        expected = np.array([2.0, 4.0])
        assert np.allclose(hdiag1, expected, rtol=1e-3)
        assert np.allclose(hdiag2, expected, rtol=1e-3)
    
    def test_finite_diff_hessian_diag_linear_function(self):
        """Test Hessian diagonal for linear function (should be zero)."""
        # f(x) = 2*x[0] + 3*x[1]
        # Hessian diagonal: [0, 0]
        def linear_loss(theta):
            return 2*theta[0] + 3*theta[1]
        
        theta = np.array([1.0, 1.0])
        hdiag = finite_diff_hessian_diag(linear_loss, theta, eps=1e-5)
        
        expected = np.array([0.0, 0.0])
        assert np.allclose(hdiag, expected, atol=1e-5)
    
    def test_finite_diff_hessian_diag_quartic_function(self):
        """Test Hessian diagonal for a quartic function."""
        # f(x) = x[0]^4
        # f''(x) = 12*x[0]^2
        def quartic_loss(theta):
            return theta[0]**4
        
        theta = np.array([2.0])
        hdiag = finite_diff_hessian_diag(quartic_loss, theta, eps=1e-4)
        
        # At x=2, f''(2) = 12*4 = 48
        expected = np.array([48.0])
        assert np.allclose(hdiag, expected, rtol=1e-2)
    
    def test_finite_diff_hessian_diag_output_shape(self):
        """Test that output shape matches input parameter dimension."""
        def loss(theta):
            return np.sum(theta**2)
        
        for d in [1, 3, 5, 10]:
            theta = np.ones(d)
            hdiag = finite_diff_hessian_diag(loss, theta)
            assert hdiag.shape == (d,)
    
    def test_finite_diff_hessian_diag_custom_eps(self):
        """Test that custom epsilon value works correctly."""
        def quadratic_loss(theta):
            return theta[0]**2 + theta[1]**2
        
        theta = np.array([1.0, 1.0])
        
        # Different epsilon values should give similar results for smooth functions
        hdiag1 = finite_diff_hessian_diag(quadratic_loss, theta, eps=1e-3)
        hdiag2 = finite_diff_hessian_diag(quadratic_loss, theta, eps=1e-5)
        
        expected = np.array([2.0, 2.0])
        assert np.allclose(hdiag1, expected, rtol=1e-2)
        assert np.allclose(hdiag2, expected, rtol=1e-3)
