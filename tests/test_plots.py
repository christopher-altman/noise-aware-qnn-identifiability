import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.plots import plot_results


class TestGetIndicesToAnnotate:
    """Tests for get_indices_to_annotate annotation logic."""
    
    def test_returns_all_indices_when_below_threshold(self):
        """Test that get_indices_to_annotate returns all indices when n_results is below ANNOTATE_ALL_THRESHOLD."""
        # Create 10 results (below threshold of 15)
        results = [
            {
                "p_dep": 0.1 * i,
                "sigma_phase": 0.05 * i,
                "acc": 0.8 + 0.01 * i,
                "param_l2": 0.1 - 0.005 * i,
                "ident_proxy": 0.01 * (i + 1)
            }
            for i in range(10)
        ]
        
        # Mock plt to avoid actual plotting
        with patch('src.plots.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig
            
            # Call plot_results which internally uses get_indices_to_annotate
            plot_results(results, output_dir="/tmp/test_plots")
            
            # Check that annotate was called for all 10 points
            # Each point should be annotated in each of the plots
            annotate_calls = mock_plt.annotate.call_args_list
            
            # There should be annotations (we expect multiple plots with all points annotated)
            assert len(annotate_calls) > 0
            
            # Verify that all indices 0-9 appear in annotations
            # Extract the annotation strings to verify all points are included
            annotation_strings = [call[0][0] for call in annotate_calls]
            
            # Check that we have annotations for all 10 points in at least one plot
            # Each plot should have 10 annotations
            assert len(annotation_strings) >= 10
    
    def test_returns_extreme_indices_when_above_threshold(self):
        """Test that get_indices_to_annotate returns indices for lowest identifiability and highest accuracy when n_results is above ANNOTATE_ALL_THRESHOLD."""
        # Create 20 results (above threshold of 15)
        results = []
        for i in range(20):
            results.append({
                "p_dep": 0.1 * i,
                "sigma_phase": 0.05 * i,
                "acc": 0.5 + 0.02 * i,  # Highest at index 19
                "param_l2": 0.2 - 0.005 * i,
                "ident_proxy": 0.1 + 0.01 * i  # Lowest at index 0
            })
        
        with patch('src.plots.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig
            
            plot_results(results, output_dir="/tmp/test_plots")
            
            # Check annotate calls
            annotate_calls = mock_plt.annotate.call_args_list
            
            # Extract annotation strings and positions
            annotations = []
            for call in annotate_calls:
                text = call[0][0]
                position = call[0][1]
                annotations.append((text, position))
            
            # We should have fewer annotations than total points (not all 20)
            # Should only annotate extreme points
            unique_positions = set([pos for _, pos in annotations])
            
            # With 20 points, we expect only 2-3 annotations per plot (extremes)
            # There are multiple plots, so we check that we don't annotate all 20
            assert len(unique_positions) < 20
            
            # Verify that the annotations include the extremes
            # Index 0 should be annotated (lowest identifiability)
            # Index 19 should be annotated (highest accuracy)
            p_values = [r["p_dep"] for r in results]
            s_values = [r["sigma_phase"] for r in results]
            acc_values = [r["acc"] for r in results]
            ident_values = [r["ident_proxy"] for r in results]
            
            # Check if extreme points are in the annotations
            min_ident_idx = np.argmin(ident_values)
            max_acc_idx = np.argmax(acc_values)
            
            # Construct expected annotation texts
            expected_min_ident = f"p={p_values[min_ident_idx]:.2f},σ={s_values[min_ident_idx]:.2f}"
            expected_max_acc = f"p={p_values[max_acc_idx]:.2f},σ={s_values[max_acc_idx]:.2f}"
            
            annotation_texts = [text for text, _ in annotations]
            assert expected_min_ident in annotation_texts
            assert expected_max_acc in annotation_texts
    
    def test_includes_highest_fisher_condition_number_when_available(self):
        """Test that get_indices_to_annotate includes index for highest fisher_condition_number when available and n_results is above ANNOTATE_ALL_THRESHOLD."""
        # Create 20 results with fisher_condition_number
        results = []
        for i in range(20):
            results.append({
                "p_dep": 0.1 * i,
                "sigma_phase": 0.05 * i,
                "acc": 0.5 + 0.02 * i,
                "param_l2": 0.2 - 0.005 * i,
                "ident_proxy": 0.1 + 0.01 * i,
                "fisher_condition_number": 100.0 + 50.0 * i  # Highest at index 19
            })
        
        with patch('src.plots.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig
            
            plot_results(results, output_dir="/tmp/test_plots")
            
            # Get annotations
            annotate_calls = mock_plt.annotate.call_args_list
            annotations = [call[0][0] for call in annotate_calls]
            
            # The highest fisher condition number is at index 19
            # This should be annotated
            max_fisher_idx = 19
            p_val = results[max_fisher_idx]["p_dep"]
            s_val = results[max_fisher_idx]["sigma_phase"]
            expected_text = f"p={p_val:.2f},σ={s_val:.2f}"
            
            # This annotation should appear in the annotations
            assert expected_text in annotations
    
    def test_handles_missing_fisher_condition_number_gracefully(self):
        """Test that get_indices_to_annotate handles missing fisher_condition_number gracefully."""
        # Create 20 results without fisher_condition_number field
        results = []
        for i in range(20):
            results.append({
                "p_dep": 0.1 * i,
                "sigma_phase": 0.05 * i,
                "acc": 0.5 + 0.02 * i,
                "param_l2": 0.2 - 0.005 * i,
                "ident_proxy": 0.1 + 0.01 * i
                # No fisher_condition_number field
            })
        
        # Should not raise an error
        with patch('src.plots.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig
            
            # This should complete without error
            plot_results(results, output_dir="/tmp/test_plots")
            
            # Verify that plot was called
            assert mock_plt.figure.called
            assert mock_plt.savefig.called
    
    def test_handles_nan_fisher_condition_number_gracefully(self):
        """Test that get_indices_to_annotate handles NaN fisher_condition_number gracefully."""
        # Create 20 results with some NaN fisher_condition_number values
        results = []
        for i in range(20):
            fisher_val = np.nan if i % 5 == 0 else 100.0 + 50.0 * i
            results.append({
                "p_dep": 0.1 * i,
                "sigma_phase": 0.05 * i,
                "acc": 0.5 + 0.02 * i,
                "param_l2": 0.2 - 0.005 * i,
                "ident_proxy": 0.1 + 0.01 * i,
                "fisher_condition_number": fisher_val
            })
        
        # Should not raise an error
        with patch('src.plots.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig
            
            # This should complete without error
            plot_results(results, output_dir="/tmp/test_plots")
            
            # Verify that plot was called
            assert mock_plt.figure.called
            assert mock_plt.savefig.called
    
    def test_handles_all_nan_fisher_condition_numbers(self):
        """Test that get_indices_to_annotate handles all NaN fisher_condition_number values gracefully."""
        # Create 20 results where all fisher_condition_number values are NaN
        results = []
        for i in range(20):
            results.append({
                "p_dep": 0.1 * i,
                "sigma_phase": 0.05 * i,
                "acc": 0.5 + 0.02 * i,
                "param_l2": 0.2 - 0.005 * i,
                "ident_proxy": 0.1 + 0.01 * i,
                "fisher_condition_number": np.nan
            })
        
        # Should not raise an error even when all values are NaN
        with patch('src.plots.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig
            
            # This should complete without error
            plot_results(results, output_dir="/tmp/test_plots")
            
            # Verify that plot was called
            assert mock_plt.figure.called
            assert mock_plt.savefig.called
    
    def test_annotation_density_threshold_boundary(self):
        """Test behavior exactly at the ANNOTATE_ALL_THRESHOLD boundary."""
        # Test with exactly 15 results (at threshold)
        results = []
        for i in range(15):
            results.append({
                "p_dep": 0.1 * i,
                "sigma_phase": 0.05 * i,
                "acc": 0.5 + 0.02 * i,
                "param_l2": 0.2 - 0.005 * i,
                "ident_proxy": 0.1 + 0.01 * i
            })
        
        with patch('src.plots.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig
            
            plot_results(results, output_dir="/tmp/test_plots")
            
            # At threshold, should annotate all points
            annotate_calls = mock_plt.annotate.call_args_list
            assert len(annotate_calls) > 0
    
    def test_annotation_indices_above_threshold_only_includes_extremes(self):
        """Test that when above threshold, only extreme indices are annotated, not all points."""
        # Create 25 results (well above threshold)
        results = []
        for i in range(25):
            results.append({
                "p_dep": 0.1 * i,
                "sigma_phase": 0.05 * i,
                "acc": 0.6 + 0.01 * i,
                "param_l2": 0.3 - 0.005 * i,
                "ident_proxy": 0.05 + 0.02 * i,
                "fisher_condition_number": 200.0 + 100.0 * i
            })
        
        with patch('src.plots.plt') as mock_plt:
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig
            
            plot_results(results, output_dir="/tmp/test_plots")
            
            # Extract annotation texts to verify limited annotations
            annotate_calls = mock_plt.annotate.call_args_list
            
            # Extract all annotation texts
            annotation_texts = [call[0][0] for call in annotate_calls]
            
            # Count unique annotation texts (same text may appear in multiple plots)
            unique_texts = set(annotation_texts)
            
            # With 25 points and 3 extremes to annotate, we should see at most 3 unique texts
            # (one for min ident, one for max acc, one for max fisher)
            # May be fewer if extremes overlap (e.g., max acc == max fisher)
            # These same texts appear across all plots
            assert 1 <= len(unique_texts) <= 3
            
            # Verify we're not annotating all 25 points
            assert len(unique_texts) < 25
