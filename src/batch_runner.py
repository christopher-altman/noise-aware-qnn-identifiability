import json
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from .config import ExperimentConfig, BatchConfig
from .train import run_experiment


def run_single_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with experiment metadata and results
    """
    start_time = time.time()
    
    if not config.verbose:
        print(f"Starting experiment: {config.name}")
    
    # Create output directory
    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run experiment
        results = run_experiment(
            seed=config.seed,
            n=config.samples,
            d=config.dimension,
            noise_grid=config.get_noise_tuples(),
            optimizer_iterations=config.optimizer_iterations,
            hessian_eps=config.hessian_eps,
            output_dir=output_dir,
            generate_plots=config.generate_plots,
            verbose=config.verbose,
            quiet=not config.verbose,
        )
        
        # Save results
        if config.save_results:
            results_path = output_dir / "results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        elapsed_time = time.time() - start_time
        
        return {
            'name': config.name,
            'status': 'success',
            'elapsed_time': elapsed_time,
            'output_dir': str(output_dir),
            'results': results,
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error in experiment {config.name}: {e}")
        
        return {
            'name': config.name,
            'status': 'failed',
            'elapsed_time': elapsed_time,
            'error': str(e),
        }


def run_batch_sequential(configs: List[ExperimentConfig]) -> List[Dict[str, Any]]:
    """
    Run experiments sequentially.
    
    Args:
        configs: List of experiment configurations
        
    Returns:
        List of results for each experiment
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running {len(configs)} experiments sequentially")
    print(f"{'='*60}\n")
    
    for idx, config in enumerate(configs, 1):
        print(f"\n[{idx}/{len(configs)}] Running: {config.name}")
        result = run_single_experiment(config)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✓ Completed in {result['elapsed_time']:.2f}s")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")
    
    return results


def run_batch_parallel(configs: List[ExperimentConfig], 
                       max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Run experiments in parallel.
    
    Args:
        configs: List of experiment configurations
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of results for each experiment
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running {len(configs)} experiments in parallel")
    print(f"Max workers: {max_workers}")
    print(f"{'='*60}\n")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_single_experiment, config): config 
            for config in configs
        }
        
        # Process completed jobs
        completed = 0
        for future in as_completed(future_to_config):
            completed += 1
            config = future_to_config[future]
            
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"[{completed}/{len(configs)}] ✓ {result['name']} "
                          f"({result['elapsed_time']:.2f}s)")
                else:
                    print(f"[{completed}/{len(configs)}] ✗ {result['name']} "
                          f"- {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"[{completed}/{len(configs)}] ✗ {config.name} - {e}")
                results.append({
                    'name': config.name,
                    'status': 'failed',
                    'error': str(e),
                })
    
    return results


def run_batch(batch_config: BatchConfig) -> Dict[str, Any]:
    """
    Run a batch of experiments.
    
    Args:
        batch_config: Batch configuration
        
    Returns:
        Summary of batch execution
    """
    start_time = time.time()
    
    # Get all experiments
    experiments = batch_config.get_all_experiments()
    
    if not experiments:
        print("No experiments to run!")
        return {
            'batch_name': batch_config.name,
            'total_experiments': 0,
            'results': [],
        }
    
    # Run experiments
    if batch_config.parallel:
        results = run_batch_parallel(experiments, batch_config.max_workers)
    else:
        results = run_batch_sequential(experiments)
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    summary = {
        'batch_name': batch_config.name,
        'total_experiments': len(experiments),
        'successful': successful,
        'failed': failed,
        'total_time': total_time,
        'parallel': batch_config.parallel,
        'results': results,
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Batch Execution Summary: {batch_config.name}")
    print(f"{'='*60}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {total_time/len(experiments):.2f}s per experiment")
    print(f"{'='*60}\n")
    
    return summary


def save_batch_summary(summary: Dict[str, Any], output_path: Path):
    """Save batch execution summary to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Batch summary saved to: {output_path}")
