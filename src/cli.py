import argparse
import sys
from pathlib import Path
from .train import run_experiment
from .config import load_config, ExperimentConfig, BatchConfig
from .batch_runner import run_batch, save_batch_summary


def parse_noise_grid(noise_str: str) -> list:
    """
    Parse noise grid from string format.
    
    Examples:
        "0.0,0.0;0.05,0.0;0.1,0.1" -> [(0.0, 0.0), (0.05, 0.0), (0.1, 0.1)]
    """
    if not noise_str:
        return None
    
    try:
        # Support both semicolon and space separation
        if ';' in noise_str:
            pairs = noise_str.split(';')
        else:
            pairs = noise_str.split()
            
        result = []
        for pair in pairs:
            if not pair.strip():
                continue
            p_dep, sigma_phase = pair.split(',')
            result.append((float(p_dep), float(sigma_phase)))
        return result
    except (ValueError, IndexError) as e:
        raise argparse.ArgumentTypeError(
            f"Invalid noise grid format. Expected 'p1,s1;p2,s2;...' but got: {noise_str}"
        ) from e


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog='noise-aware-qnn-identifiability',
        description='Study identifiability collapse in noisy quantum neural networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python -m src.cli
  
  # Custom problem size and seed
  python -m src.cli --samples 1000 --dimension 16 --seed 42
  
  # Custom noise grid
  python -m src.cli --noise-grid "0.0,0.0;0.1,0.0;0.2,0.1"
  
  # Specify output directory
  python -m src.cli --output-dir ./results/experiment_01
  
  # Increase optimizer iterations
  python -m src.cli --optimizer-iterations 5000
  
  # Quiet mode (minimal output)
  python -m src.cli --quiet
  
  # Verbose mode (detailed progress)
  python -m src.cli --verbose
        """
    )
    
    # Configuration file
    config_group = parser.add_argument_group('Configuration File')
    config_group.add_argument(
        '-c', '--config',
        type=Path,
        default=None,
        metavar='FILE',
        help='Load experiment configuration from YAML or JSON file'
    )
    
    # Problem configuration
    problem_group = parser.add_argument_group('Problem Configuration')
    problem_group.add_argument(
        '-n', '--samples',
        type=int,
        default=512,
        metavar='N',
        help='Number of training samples (default: 512)'
    )
    problem_group.add_argument(
        '-d', '--dimension',
        type=int,
        default=8,
        metavar='D',
        help='Parameter dimension (default: 8)'
    )
    problem_group.add_argument(
        '-s', '--seed',
        type=int,
        default=0,
        metavar='SEED',
        help='Random seed for reproducibility (default: 0)'
    )
    
    # Noise configuration
    noise_group = parser.add_argument_group('Noise Configuration')
    noise_group.add_argument(
        '--noise-grid',
        type=parse_noise_grid,
        default=None,
        metavar='GRID',
        help='Custom noise grid as "p1,s1;p2,s2;..." where p=depolarizing prob, s=phase sigma. '
             'Example: "0.0,0.0;0.05,0.0;0.1,0.1"'
    )
    noise_group.add_argument(
        '--noise-presets',
        choices=['minimal', 'default', 'extensive', 'high-noise'],
        default='default',
        help='Use predefined noise grid preset (default: default)'
    )
    
    # Optimization configuration
    opt_group = parser.add_argument_group('Optimization Configuration')
    opt_group.add_argument(
        '--optimizer-iterations',
        type=int,
        default=2000,
        metavar='ITERS',
        help='Number of random search iterations per noise setting (default: 2000)'
    )
    opt_group.add_argument(
        '--hessian-eps',
        type=float,
        default=1e-3,
        metavar='EPS',
        help='Finite difference epsilon for Hessian computation (default: 1e-3)'
    )
    
    # Enhanced metrics configuration
    metrics_group = parser.add_argument_group('Enhanced Metrics')
    metrics_group.add_argument(
        '--enhanced-metrics',
        action='store_true',
        help='Compute Fisher Information Matrix and advanced identifiability metrics '
             '(more rigorous but computationally expensive)'
    )
    metrics_group.add_argument(
        '--fisher-batch-size',
        type=int,
        default=None,
        metavar='SIZE',
        help='Batch size for Fisher computation (default: use all samples). '
             'Smaller values are faster but less accurate'
    )
    
    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=None,
        metavar='DIR',
        help='Output directory for figures and results (default: artifacts/latest)'
    )
    output_group.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    output_group.add_argument(
        '--extended-viz',
        action='store_true',
        help='Generate extended visualizations (heatmaps, combined metrics)'
    )
    output_group.add_argument(
        '--interactive',
        action='store_true',
        help='Generate interactive Plotly visualizations (HTML)'
    )
    output_group.add_argument(
        '--save-results',
        type=Path,
        default=None,
        metavar='FILE',
        help='Save results to JSON file (e.g., results.json)'
    )
    
    # Verbosity
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output with detailed progress'
    )
    verbosity_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress all non-essential output'
    )
    
    return parser


def get_noise_preset(preset_name: str) -> list:
    """Return predefined noise grid based on preset name."""
    presets = {
        'minimal': [
            (0.0, 0.0),
            (0.1, 0.0),
        ],
        'default': [
            (0.0, 0.0),
            (0.05, 0.0),
            (0.10, 0.0),
            (0.10, 0.10),
            (0.20, 0.20),
        ],
        'extensive': [
            (0.0, 0.0),
            (0.05, 0.0),
            (0.10, 0.0),
            (0.15, 0.0),
            (0.10, 0.05),
            (0.10, 0.10),
            (0.15, 0.15),
            (0.20, 0.20),
            (0.25, 0.25),
        ],
        'high-noise': [
            (0.0, 0.0),
            (0.20, 0.0),
            (0.30, 0.0),
            (0.30, 0.30),
            (0.40, 0.40),
        ],
    }
    return presets[preset_name]


def run_from_config_file(args):
    """Run experiment(s) from configuration file."""
    try:
        config = load_config(args.config)
        
        if isinstance(config, BatchConfig):
            # Run batch of experiments
            if not args.quiet:
                print(f"Loading batch configuration from: {args.config}")
                print(f"Batch name: {config.name}")
                print(f"Total experiments (including sweeps): {len(config.get_all_experiments())}")
            
            summary = run_batch(config)
            
            # Save batch summary
            summary_path = Path(config.experiments[0].output_dir if config.experiments else "./results") / "batch_summary.json"
            save_batch_summary(summary, summary_path)
            
            return 0
            
        elif isinstance(config, ExperimentConfig):
            # Run single experiment
            if not args.quiet:
                print(f"Loading experiment configuration from: {args.config}")
                print(f"Experiment name: {config.name}")
            
            from .batch_runner import run_single_experiment
            result = run_single_experiment(config)
            
            if result['status'] == 'success':
                if not args.quiet:
                    print(f"\n✓ Experiment completed successfully")
                return 0
            else:
                print(f"\n✗ Experiment failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
                return 1
        
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Check if config file is provided
    if args.config:
        return run_from_config_file(args)
    
    # Create output directory if needed
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine noise grid
    if args.noise_grid is not None:
        noise_grid = args.noise_grid
        if not args.quiet:
            print(f"Using custom noise grid: {noise_grid}")
    else:
        noise_grid = get_noise_preset(args.noise_presets)
        if not args.quiet:
            print(f"Using noise preset '{args.noise_presets}': {noise_grid}")
    
    # Print configuration
    if args.verbose:
        print("\n=== Experiment Configuration ===")
        print(f"Samples (n):              {args.samples}")
        print(f"Dimension (d):            {args.dimension}")
        print(f"Random seed:              {args.seed}")
        print(f"Optimizer iterations:     {args.optimizer_iterations}")
        print(f"Hessian epsilon:          {args.hessian_eps}")
        print(f"Output directory:         {args.output_dir}")
        print(f"Generate plots:           {not args.no_plots}")
        if args.save_results:
            print(f"Save results to:          {args.save_results}")
        print("=" * 33 + "\n")
    
    # Run experiment
    try:
        results = run_experiment(
            seed=args.seed,
            n=args.samples,
            d=args.dimension,
            noise_grid=noise_grid,
            optimizer_iterations=args.optimizer_iterations,
            hessian_eps=args.hessian_eps,
            output_dir=args.output_dir,
            generate_plots=not args.no_plots,
            extended_viz=args.extended_viz,
            interactive=args.interactive,
            verbose=args.verbose,
            quiet=args.quiet,
            enhanced_metrics=args.enhanced_metrics,
            fisher_batch_size=args.fisher_batch_size,
        )
        
        # Save results if requested
        if args.save_results:
            import json
            save_path = args.output_dir / args.save_results
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            if not args.quiet:
                print(f"\nResults saved to: {save_path}")
        
        # Print summary
        if not args.quiet:
            print("\n=== Experiment Summary ===")
            for r in results:
                print(f"p={r['p_dep']:.2f}, σ={r['sigma_phase']:.2f} | "
                      f"acc={r['acc']:.4f}, param_err={r['param_l2']:.4f}, "
                      f"ident={r['ident_proxy']:.2e}")
        
        if args.verbose:
            print("\n✓ Experiment completed successfully")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
