import pytest
import argparse
from src.cli import parse_noise_grid, get_noise_preset, create_parser


class TestParseNoiseGrid:
    """Tests for noise grid parsing."""
    
    def test_parse_simple_noise_grid(self):
        """Test parsing a simple noise grid string."""
        result = parse_noise_grid("0.0,0.0;0.1,0.0")
        expected = [(0.0, 0.0), (0.1, 0.0)]
        assert result == expected
    
    def test_parse_complex_noise_grid(self):
        """Test parsing a complex noise grid with multiple settings."""
        result = parse_noise_grid("0.0,0.0;0.05,0.1;0.1,0.2;0.15,0.3")
        expected = [(0.0, 0.0), (0.05, 0.1), (0.1, 0.2), (0.15, 0.3)]
        assert result == expected
    
    def test_parse_noise_grid_with_floats(self):
        """Test parsing noise grid with various float formats."""
        result = parse_noise_grid("0.123,0.456;0.789,0.012")
        expected = [(0.123, 0.456), (0.789, 0.012)]
        assert result == expected
    
    def test_parse_empty_noise_grid(self):
        """Test parsing empty noise grid returns None."""
        result = parse_noise_grid("")
        assert result is None
    
    def test_parse_invalid_noise_grid_format(self):
        """Test that invalid format raises appropriate error."""
        with pytest.raises(argparse.ArgumentTypeError):
            parse_noise_grid("0.0,0.0,0.1")  # Too many values
        
        with pytest.raises(argparse.ArgumentTypeError):
            parse_noise_grid("0.0;0.1")  # Missing comma
        
        with pytest.raises(argparse.ArgumentTypeError):
            parse_noise_grid("invalid,format")  # Non-numeric values


class TestGetNoisePreset:
    """Tests for noise preset retrieval."""
    
    def test_get_minimal_preset(self):
        """Test minimal preset has correct format."""
        result = get_noise_preset('minimal')
        assert len(result) == 2
        assert result[0] == (0.0, 0.0)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in result)
    
    def test_get_default_preset(self):
        """Test default preset has correct format."""
        result = get_noise_preset('default')
        assert len(result) == 5
        assert result[0] == (0.0, 0.0)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in result)
    
    def test_get_extensive_preset(self):
        """Test extensive preset has correct format."""
        result = get_noise_preset('extensive')
        assert len(result) == 9
        assert result[0] == (0.0, 0.0)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in result)
    
    def test_get_high_noise_preset(self):
        """Test high-noise preset has correct format."""
        result = get_noise_preset('high-noise')
        assert len(result) == 5
        assert result[0] == (0.0, 0.0)
        # Verify it actually has higher noise values
        assert any(p[0] >= 0.2 or p[1] >= 0.2 for p in result)
    
    def test_all_presets_start_with_baseline(self):
        """Test that all presets start with (0.0, 0.0) baseline."""
        presets = ['minimal', 'default', 'extensive', 'high-noise']
        for preset in presets:
            result = get_noise_preset(preset)
            assert result[0] == (0.0, 0.0), f"Preset '{preset}' should start with (0.0, 0.0)"
    
    def test_preset_noise_values_in_valid_range(self):
        """Test that all preset values are in valid ranges."""
        presets = ['minimal', 'default', 'extensive', 'high-noise']
        for preset in presets:
            result = get_noise_preset(preset)
            for p_dep, sigma_phase in result:
                assert 0.0 <= p_dep <= 1.0, f"p_dep should be in [0,1]: {p_dep}"
                assert sigma_phase >= 0.0, f"sigma_phase should be >= 0: {sigma_phase}"


class TestCreateParser:
    """Tests for argument parser creation."""
    
    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_parser_default_values(self):
        """Test parser default values."""
        parser = create_parser()
        args = parser.parse_args([])
        
        assert args.samples == 512
        assert args.dimension == 8
        assert args.seed == 0
        assert args.optimizer_iterations == 2000
        assert args.hessian_eps == 1e-3
        assert args.noise_presets == 'default'
        assert args.verbose is False
        assert args.quiet is False
    
    def test_parser_samples_argument(self):
        """Test parsing samples argument."""
        parser = create_parser()
        args = parser.parse_args(['--samples', '1000'])
        assert args.samples == 1000
        
        args = parser.parse_args(['-n', '256'])
        assert args.samples == 256
    
    def test_parser_dimension_argument(self):
        """Test parsing dimension argument."""
        parser = create_parser()
        args = parser.parse_args(['--dimension', '16'])
        assert args.dimension == 16
        
        args = parser.parse_args(['-d', '4'])
        assert args.dimension == 4
    
    def test_parser_seed_argument(self):
        """Test parsing seed argument."""
        parser = create_parser()
        args = parser.parse_args(['--seed', '42'])
        assert args.seed == 42
        
        args = parser.parse_args(['-s', '123'])
        assert args.seed == 123
    
    def test_parser_noise_grid_argument(self):
        """Test parsing custom noise grid."""
        parser = create_parser()
        args = parser.parse_args(['--noise-grid', '0.0,0.0;0.1,0.1'])
        assert args.noise_grid == [(0.0, 0.0), (0.1, 0.1)]
    
    def test_parser_noise_presets_argument(self):
        """Test parsing noise presets."""
        parser = create_parser()
        
        for preset in ['minimal', 'default', 'extensive', 'high-noise']:
            args = parser.parse_args(['--noise-presets', preset])
            assert args.noise_presets == preset
    
    def test_parser_optimizer_iterations_argument(self):
        """Test parsing optimizer iterations."""
        parser = create_parser()
        args = parser.parse_args(['--optimizer-iterations', '5000'])
        assert args.optimizer_iterations == 5000
    
    def test_parser_hessian_eps_argument(self):
        """Test parsing Hessian epsilon."""
        parser = create_parser()
        args = parser.parse_args(['--hessian-eps', '1e-4'])
        assert args.hessian_eps == 1e-4
    
    def test_parser_output_dir_argument(self):
        """Test parsing output directory."""
        parser = create_parser()
        args = parser.parse_args(['--output-dir', './results'])
        assert str(args.output_dir) == 'results'
        
        args = parser.parse_args(['-o', './test'])
        assert str(args.output_dir) == 'test'
    
    def test_parser_no_plots_flag(self):
        """Test no-plots flag."""
        parser = create_parser()
        args = parser.parse_args(['--no-plots'])
        assert args.no_plots is True
        
        args = parser.parse_args([])
        assert args.no_plots is False
    
    def test_parser_save_results_argument(self):
        """Test save-results argument."""
        parser = create_parser()
        args = parser.parse_args(['--save-results', 'results.json'])
        assert str(args.save_results) == 'results.json'
    
    def test_parser_verbose_flag(self):
        """Test verbose flag."""
        parser = create_parser()
        args = parser.parse_args(['--verbose'])
        assert args.verbose is True
        
        args = parser.parse_args(['-v'])
        assert args.verbose is True
    
    def test_parser_quiet_flag(self):
        """Test quiet flag."""
        parser = create_parser()
        args = parser.parse_args(['--quiet'])
        assert args.quiet is True
        
        args = parser.parse_args(['-q'])
        assert args.quiet is True
    
    def test_parser_verbose_quiet_mutual_exclusion(self):
        """Test that verbose and quiet are mutually exclusive."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['--verbose', '--quiet'])
    
    def test_parser_complex_combination(self):
        """Test parsing a complex combination of arguments."""
        parser = create_parser()
        args = parser.parse_args([
            '--samples', '1000',
            '--dimension', '16',
            '--seed', '42',
            '--noise-grid', '0.0,0.0;0.1,0.1;0.2,0.2',
            '--optimizer-iterations', '3000',
            '--output-dir', './results',
            '--save-results', 'exp.json',
            '--verbose'
        ])
        
        assert args.samples == 1000
        assert args.dimension == 16
        assert args.seed == 42
        assert args.noise_grid == [(0.0, 0.0), (0.1, 0.1), (0.2, 0.2)]
        assert args.optimizer_iterations == 3000
        assert str(args.output_dir) == 'results'
        assert str(args.save_results) == 'exp.json'
        assert args.verbose is True
