import json
import csv
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def export_results_json(
    results: List[Dict[str, Any]],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
    pretty: bool = True
):
    """
    Export results to JSON format.
    
    Args:
        results: List of experiment results
        output_path: Path to save JSON file
        metadata: Optional metadata to include
        pretty: Use pretty printing (indent)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'num_experiments': len(results),
        'metadata': metadata or {},
        'results': results
    }
    
    with open(output_path, 'w') as f:
        if pretty:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        else:
            json.dump(data, f, cls=NumpyEncoder)


def export_results_csv(
    results: List[Dict[str, Any]],
    output_path: Path,
    flatten_nested: bool = True
):
    """
    Export results to CSV format.
    
    Args:
        results: List of experiment results
        output_path: Path to save CSV file
        flatten_nested: Flatten nested dictionaries
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    # Determine all keys
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    # Sort keys for consistent column order
    fieldnames = sorted(all_keys)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Convert numpy types to Python types
            row = {}
            for key, value in result.items():
                if isinstance(value, (np.ndarray, list)):
                    row[key] = str(value)  # Convert arrays to string
                elif isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                               np.int16, np.int32, np.int64, np.uint8,
                               np.uint16, np.uint32, np.uint64)):
                    row[key] = int(value)
                elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                    row[key] = float(value)
                elif isinstance(value, np.bool_):
                    row[key] = bool(value)
                else:
                    row[key] = value
            
            writer.writerow(row)


def export_results_pickle(
    results: List[Dict[str, Any]],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Export results to pickle format (preserves exact types).
    
    Args:
        results: List of experiment results
        output_path: Path to save pickle file
        metadata: Optional metadata to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'num_experiments': len(results),
        'metadata': metadata or {},
        'results': results
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_results_json(input_path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)


def load_results_pickle(input_path: Path) -> Dict[str, Any]:
    """Load results from pickle file."""
    with open(input_path, 'rb') as f:
        return pickle.load(f)


def export_summary_statistics(
    results: List[Dict[str, Any]],
    output_path: Path
):
    """
    Export summary statistics of results.
    
    Args:
        results: List of experiment results
        output_path: Path to save summary
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics for numeric fields
    numeric_fields = ['acc', 'param_l2', 'ident_proxy', 'p_dep', 'sigma_phase']
    
    summary = {
        'num_experiments': len(results),
        'timestamp': datetime.now().isoformat(),
        'statistics': {}
    }
    
    for field in numeric_fields:
        values = [r[field] for r in results if field in r]
        if values:
            summary['statistics'][field] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def export_all_formats(
    results: List[Dict[str, Any]],
    output_dir: Path,
    base_name: str = 'results',
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Export results in all available formats.
    
    Args:
        results: List of experiment results
        output_dir: Output directory
        base_name: Base name for output files
        metadata: Optional metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export JSON
    export_results_json(
        results,
        output_dir / f"{base_name}.json",
        metadata=metadata
    )
    
    # Export CSV
    export_results_csv(
        results,
        output_dir / f"{base_name}.csv"
    )
    
    # Export pickle
    export_results_pickle(
        results,
        output_dir / f"{base_name}.pkl",
        metadata=metadata
    )
    
    # Export summary
    export_summary_statistics(
        results,
        output_dir / f"{base_name}_summary.json"
    )


def create_results_dataframe(results: List[Dict[str, Any]]):
    """
    Convert results to pandas DataFrame (if pandas available).
    
    Args:
        results: List of experiment results
        
    Returns:
        DataFrame or None if pandas not available
    """
    try:
        import pandas as pd
        return pd.DataFrame(results)
    except ImportError:
        print("Warning: pandas not available for DataFrame export")
        return None
