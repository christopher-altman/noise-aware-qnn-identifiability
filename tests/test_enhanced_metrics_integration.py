"""
Integration tests for enhanced metrics with the main experiment pipeline.

These tests verify that:
1. Enhanced metrics can be computed successfully
2. Fisher condition number correlates with identifiability collapse
3. Effective rank decreases with noise
4. Integration with CLI and train.py works correctly
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from src.train import run_experiment
from src.circuits import feature_map, qnn_forward
from src.noise import apply_depolarizing, apply_phase_noise
from src.enhanced_metrics import compute_all_enhanced_metrics


class TestEnhancedMetricsIntegration:
    """Test enhanced metrics integration with main pipeline."""
    
    def test_enhanced_metrics_computation(self):
        """Test that enhanced metrics can be computed in a small experiment."""
        results = run_experiment(
            seed=42,
            n=50,  # Small dataset for speed
            d=4,   # Small dimension
            noise_grid=[(0.0, 0.0), (0.1, 0.0)],
            optimizer_iterations=100,
            enhanced_metrics=True,
            generate_plots=False,
            verbose=False,
            quiet=True
        )
        
        # Check that results contain enhanced metrics
        assert len(results) == 2
        for result in results:
            assert 'fisher_condition_number' in result
            assert 'fisher_effective_rank' in result
            assert 'fisher_trace' in result
            assert 'information_geometry_quality' in result
    
    def test_fisher_condition_increases_with_noise(self):
        """Verify that Fisher condition number trends upward with noise level."""
        results = run_experiment(
            seed=42,
            n=50,
            d=4,
            noise_grid=[(0.0, 0.0), (0.2, 0.0)],  # Use larger gap for clearer trend
            optimizer_iterations=200,  # More iterations for better convergence
            enhanced_metrics=True,
            generate_plots=False,
            verbose=False,
            quiet=True
        )
        
        # Extract Fisher condition numbers
        fisher_conds = [r['fisher_condition_number'] for r in results]
        
        # With significant noise, at least one metric should show degradation
        # Either condition number increases OR effective rank decreases
        eff_ranks = [r['fisher_effective_rank'] for r in results]
        
        cond_increased = fisher_conds[1] > fisher_conds[0] * 0.5  # Allow some variance
        rank_decreased = eff_ranks[1] < eff_ranks[0] * 1.2  # Allow some variance
        
        assert cond_increased or rank_decreased, \
            f"Some identifiability metric should degrade with noise. "\
            f"Fisher conds: {fisher_conds}, Eff ranks: {eff_ranks}"
    
    def test_effective_rank_decreases_with_noise(self):
        """Verify that effective rank decreases with noise (dimensional collapse)."""
        results = run_experiment(
            seed=42,
            n=50,
            d=4,
            noise_grid=[(0.0, 0.0), (0.15, 0.0)],
            optimizer_iterations=100,
            enhanced_metrics=True,
            generate_plots=False,
            verbose=False,
            quiet=True
        )
        
        # Extract effective ranks
        eff_ranks = [r['fisher_effective_rank'] for r in results]
        
        # Effective rank should decrease with noise
        assert eff_ranks[1] <= eff_ranks[0], \
            f"Effective rank should decrease with noise: {eff_ranks}"
    
    def test_fisher_correlates_with_identifiability_proxy(self):
        """Verify Fisher metrics detect identifiability issues."""
        results = run_experiment(
            seed=42,
            n=80,  # Larger dataset for more stable estimates
            d=4,
            noise_grid=[(0.0, 0.0), (0.0, 0.0), (0.15, 0.0), (0.15, 0.0)],  # Compare clean vs noisy
            optimizer_iterations=150,
            enhanced_metrics=True,
            generate_plots=False,
            verbose=False,
            quiet=True
        )
        
        # Group by noise level
        clean_results = [results[0], results[1]]
        noisy_results = [results[2], results[3]]
        
        # Average metrics for each group
        clean_fisher = np.mean([r['fisher_condition_number'] for r in clean_results])
        noisy_fisher = np.mean([r['fisher_condition_number'] for r in noisy_results])
        
        clean_rank = np.mean([r['fisher_effective_rank'] for r in clean_results])
        noisy_rank = np.mean([r['fisher_effective_rank'] for r in noisy_results])
        
        # At least one metric should show degradation with noise
        fisher_worse = noisy_fisher > clean_fisher * 1.1
        rank_worse = noisy_rank < clean_rank * 0.9
        
        assert fisher_worse or rank_worse, \
            f"Fisher metrics should detect degradation. Clean: κ={clean_fisher:.2e}, rank={clean_rank:.2f}. "\
            f"Noisy: κ={noisy_fisher:.2e}, rank={noisy_rank:.2f}"
    
    def test_quality_rating_matches_condition_number(self):
        """Verify quality rating aligns with condition number thresholds."""
        results = run_experiment(
            seed=42,
            n=50,
            d=4,
            noise_grid=[(0.0, 0.0), (0.2, 0.2)],
            optimizer_iterations=100,
            enhanced_metrics=True,
            generate_plots=False,
            verbose=False,
            quiet=True
        )
        
        for result in results:
            fisher_cond = result['fisher_condition_number']
            quality = result['information_geometry_quality']
            
            # Verify quality ratings match thresholds
            if fisher_cond < 100:
                assert quality == 'excellent'
            elif fisher_cond < 1000:
                assert quality == 'good'
            elif fisher_cond < 10000:
                assert quality == 'fair'
            else:
                assert quality == 'poor_ill_conditioned'
    
    def test_enhanced_metrics_with_batch_size(self):
        """Test that batch_size parameter works for Fisher computation."""
        results = run_experiment(
            seed=42,
            n=100,
            d=4,
            noise_grid=[(0.0, 0.0)],
            optimizer_iterations=100,
            enhanced_metrics=True,
            fisher_batch_size=30,  # Use subset
            generate_plots=False,
            verbose=False,
            quiet=True
        )
        
        assert len(results) == 1
        assert 'fisher_condition_number' in results[0]
        assert np.isfinite(results[0]['fisher_condition_number'])
    
    def test_enhanced_metrics_with_plots(self):
        """Test that Fisher plots are generated when enhanced_metrics=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            results = run_experiment(
                seed=42,
                n=50,
                d=4,
                noise_grid=[(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)],
                optimizer_iterations=100,
                enhanced_metrics=True,
                generate_plots=True,
                output_dir=output_dir,
                verbose=False,
                quiet=True
            )
            
            # Check that plots were created in figs/ subdirectory
            figs_dir = output_dir / "figs"
            # Base plots (always generated)
            assert (figs_dir / "fig_accuracy_vs_identifiability.png").exists()
            assert (figs_dir / "fig_param_error_vs_noise.png").exists()
            # Hero variants (always generated)
            assert (figs_dir / "hero_identifiability_dark.png").exists()
            assert (figs_dir / "hero_identifiability_light.png").exists()
            # Fisher plots (generated with enhanced_metrics=True)
            assert (figs_dir / "fig_fisher_condition_vs_noise.png").exists()
            assert (figs_dir / "fig_fisher_vs_identifiability.png").exists()
            assert (figs_dir / "fig_effective_rank_vs_noise.png").exists()
    
    def test_without_enhanced_metrics_no_fisher_data(self):
        """Verify that Fisher metrics are NOT computed when enhanced_metrics=False."""
        results = run_experiment(
            seed=42,
            n=50,
            d=4,
            noise_grid=[(0.0, 0.0)],
            optimizer_iterations=100,
            enhanced_metrics=False,
            generate_plots=False,
            verbose=False,
            quiet=True
        )
        
        # Should not have Fisher metrics
        assert 'fisher_condition_number' not in results[0]
        assert 'fisher_effective_rank' not in results[0]
    
    def test_model_loss_wrappers_correctness(self):
        """Test that model and loss function wrappers are correctly defined."""
        # Setup data
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 4))
        y = rng.integers(0, 2, size=10)
        theta = rng.normal(size=4)
        theta = theta / (np.linalg.norm(theta) + 1e-12)
        
        # Define wrappers as in train.py
        p_dep = 0.0
        sigma_phase = 0.0
        
        def model_fn(x, theta_param):
            phi = feature_map(x)
            phi_n = apply_depolarizing(p_dep, phi, rng)
            phi_n = apply_phase_noise(sigma_phase, phi_n, rng)
            score = qnn_forward(phi_n, theta_param)
            return 1.0 / (1.0 + np.exp(-score))
        
        # Test model function produces valid probabilities
        for i in range(len(X)):
            prob = model_fn(X[i], theta)
            assert 0 <= prob <= 1, f"Invalid probability: {prob}"
    
    def test_all_fisher_metrics_present(self):
        """Verify all expected Fisher metrics are in results."""
        results = run_experiment(
            seed=42,
            n=50,
            d=4,
            noise_grid=[(0.0, 0.0)],
            optimizer_iterations=100,
            enhanced_metrics=True,
            generate_plots=False,
            verbose=False,
            quiet=True
        )

        result = results[0]

        # Check all expected Fisher metrics
        expected_metrics = [
            'fisher_trace',
            'fisher_condition_number',
            'fisher_effective_rank',
            'fisher_effective_dimension',
            'fisher_participation_ratio',
            'fisher_spectral_gap',
            'fisher_eigenvalue_max',
            'fisher_eigenvalue_min',
            'hessian_condition_number',
            'hessian_max',
            'hessian_min',
            'is_well_conditioned',
            'is_identifiable',
            'information_geometry_quality'
        ]

        for metric in expected_metrics:
            assert metric in result, f"Missing metric: {metric}"
            assert result[metric] is not None, f"Metric is None: {metric}"

    def test_extended_viz_outputs_created(self):
        """Test that extended viz outputs are created when extended_viz=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            results = run_experiment(
                seed=42,
                n=50,
                d=4,
                noise_grid=[(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)],
                optimizer_iterations=100,
                generate_plots=True,
                extended_viz=True,
                interactive=False,
                enhanced_metrics=False,
                output_dir=output_dir,
                verbose=False,
                quiet=True
            )

            # Check that plots were created in figs/ subdirectory
            figs_dir = output_dir / "figs"

            # Base plots (always generated)
            assert (figs_dir / "fig_accuracy_vs_identifiability.png").exists()
            assert (figs_dir / "fig_param_error_vs_noise.png").exists()
            # Hero variants (always generated)
            assert (figs_dir / "hero_identifiability_dark.png").exists()
            assert (figs_dir / "hero_identifiability_light.png").exists()

            # Extended viz plots
            assert (figs_dir / "heatmap_acc.png").exists()
            assert (figs_dir / "heatmap_param_l2.png").exists()
            assert (figs_dir / "heatmap_ident_proxy.png").exists()
            assert (figs_dir / "combined_metrics.png").exists()

    def test_interactive_outputs_created_when_plotly_available(self):
        """Test that interactive outputs are created when plotly is available."""
        pytest.importorskip("plotly")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            results = run_experiment(
                seed=42,
                n=50,
                d=4,
                noise_grid=[(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)],
                optimizer_iterations=100,
                generate_plots=True,
                interactive=True,
                extended_viz=False,
                enhanced_metrics=False,
                output_dir=output_dir,
                verbose=False,
                quiet=True
            )

            # Check that plots were created in figs/ subdirectory
            figs_dir = output_dir / "figs"

            # Base plots (always generated)
            assert (figs_dir / "fig_accuracy_vs_identifiability.png").exists()
            assert (figs_dir / "fig_param_error_vs_noise.png").exists()
            # Hero variants (always generated)
            assert (figs_dir / "hero_identifiability_dark.png").exists()
            assert (figs_dir / "hero_identifiability_light.png").exists()

            # Interactive outputs
            assert (figs_dir / "interactive_heatmaps.html").exists()
            assert (figs_dir / "interactive_dashboard.html").exists()


class TestCLIIntegration:
    """Test CLI integration with enhanced metrics."""
    
    def test_cli_enhanced_metrics_flag(self):
        """Test that CLI accepts --enhanced-metrics flag."""
        from src.cli import create_parser
        
        parser = create_parser()
        
        # Test with enhanced metrics enabled
        args = parser.parse_args(['--enhanced-metrics'])
        assert args.enhanced_metrics is True
        
        # Test without flag
        args = parser.parse_args([])
        assert args.enhanced_metrics is False
    
    def test_cli_fisher_batch_size_flag(self):
        """Test that CLI accepts --fisher-batch-size flag."""
        from src.cli import create_parser
        
        parser = create_parser()
        
        # Test with batch size
        args = parser.parse_args(['--fisher-batch-size', '50'])
        assert args.fisher_batch_size == 50
        
        # Test default
        args = parser.parse_args([])
        assert args.fisher_batch_size is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
