# Data Export, Logging & Checkpointing

Comprehensive system for exporting results, tracking experiments, and checkpointing long runs.

## Quick Start

### Data Export

```python
from src.data_export import export_all_formats

# Export results in all formats
export_all_formats(
    results=experiment_results,
    output_dir=Path("./results"),
    base_name="my_experiment",
    metadata={'description': 'Test run'}
)

# Creates:
# - my_experiment.json (JSON with metadata)
# - my_experiment.csv (CSV for spreadsheets)
# - my_experiment.pkl (Pickle for exact types)
# - my_experiment_summary.json (Statistics)
```

### Experiment Logging

```python
from src.experiment_logger import create_experiment_logger

# Create logger
logger = create_experiment_logger(
    experiment_name="exp001",
    output_dir=Path("./logs"),
    verbose=True
)

# Log throughout experiment
logger.log_config({'samples': 512, 'dimension': 8})
logger.log_progress(iteration=100, total=1000, metrics={'loss': 0.5})
logger.log_result({'accuracy': 0.95})
logger.finalize(final_status='completed')

# Creates:
# - exp001.log (Human-readable log)
# - exp001_structured.jsonl (Machine-readable JSONL)
```

### Checkpointing

```python
from src.checkpointing import create_checkpoint_manager

# Create manager
ckpt_mgr = create_checkpoint_manager(
    experiment_name="long_run",
    checkpoint_dir=Path("./checkpoints"),
    keep_last_n=3,
    save_frequency=100  # Save every 100 iterations
)

# Save checkpoint
state = {'iteration': 500, 'theta': best_theta, 'loss': current_loss}
ckpt_mgr.save_checkpoint(state, iteration=500)

# Resume from checkpoint
checkpoint = ckpt_mgr.load_checkpoint()
theta = checkpoint['state']['theta']
start_iter = checkpoint['state']['iteration']
```

## Data Export Module

### Features

- **Multiple Formats**: JSON, CSV, Pickle
- **Numpy Support**: Automatic handling of numpy types
- **Metadata**: Attach timestamps and custom metadata
- **Summary Statistics**: Auto-calculate mean, std, min, max, median
- **DataFrame Support**: Optional pandas integration

### Export Functions

#### `export_results_json()`

Export to JSON format with metadata and timestamps.

```python
from src.data_export import export_results_json

export_results_json(
    results=[{'acc': 0.95, 'loss': 0.1}],
    output_path=Path("results.json"),
    metadata={'experiment': 'test', 'version': '1.0'},
    pretty=True  # Indented output
)
```

**Output:**
```json
{
  "timestamp": "2025-12-31T05:30:00",
  "num_experiments": 1,
  "metadata": {"experiment": "test", "version": "1.0"},
  "results": [{"acc": 0.95, "loss": 0.1}]
}
```

#### `export_results_csv()`

Export to CSV for spreadsheet analysis.

```python
from src.data_export import export_results_csv

export_results_csv(
    results=[{'p_dep': 0.0, 'acc': 0.95}, {'p_dep': 0.1, 'acc': 0.90}],
    output_path=Path("results.csv")
)
```

**Output:**
```csv
acc,p_dep
0.95,0.0
0.90,0.1
```

#### `export_results_pickle()`

Export to pickle for exact type preservation.

```python
from src.data_export import export_results_pickle

export_results_pickle(
    results=results,
    output_path=Path("results.pkl"),
    metadata={'numpy_arrays': True}
)
```

**Use case**: Preserves numpy arrays, custom objects

#### `export_summary_statistics()`

Auto-calculate statistics for numeric fields.

```python
from src.data_export import export_summary_statistics

export_summary_statistics(
    results=[{'acc': 0.95}, {'acc': 0.90}, {'acc': 0.92}],
    output_path=Path("summary.json")
)
```

**Output:**
```json
{
  "num_experiments": 3,
  "timestamp": "2025-12-31T05:30:00",
  "statistics": {
    "acc": {
      "mean": 0.9233,
      "std": 0.0208,
      "min": 0.90,
      "max": 0.95,
      "median": 0.92
    }
  }
}
```

#### `export_all_formats()`

Export in all formats at once.

```python
from src.data_export import export_all_formats

export_all_formats(
    results=results,
    output_dir=Path("./output"),
    base_name="experiment_001"
)

# Creates 4 files:
# - experiment_001.json
# - experiment_001.csv
# - experiment_001.pkl
# - experiment_001_summary.json
```

### Loading Functions

```python
from src.data_export import load_results_json, load_results_pickle

# Load JSON
data = load_results_json(Path("results.json"))
results = data['results']
metadata = data['metadata']

# Load Pickle
data = load_results_pickle(Path("results.pkl"))
```

## Experiment Logging Module

### Features

- **Dual Logging**: File + console
- **Structured Logs**: JSONL format for machine parsing
- **Event Types**: Config, progress, metrics, results, errors, checkpoints
- **Timestamps**: Automatic timestamping
- **Log Levels**: INFO, WARNING, ERROR, DEBUG

### ExperimentLogger Class

```python
from src.experiment_logger import ExperimentLogger

logger = ExperimentLogger(
    experiment_name="my_exp",
    output_dir=Path("./logs"),
    log_to_file=True,
    log_to_console=True,
    level=logging.INFO
)
```

### Logging Methods

#### Configuration

```python
logger.log_config({
    'samples': 512,
    'dimension': 8,
    'seed': 42
})
```

#### Progress

```python
for i in range(1000):
    # Your code here
    if i % 100 == 0:
        logger.log_progress(
            iteration=i,
            total=1000,
            metrics={'loss': current_loss}
        )
```

#### Metrics

```python
logger.log_metrics(
    metrics={'accuracy': 0.95, 'f1': 0.93},
    step=100
)
```

#### Results

```python
logger.log_result({
    'final_accuracy': 0.95,
    'param_error': 0.05,
    'identifiability': 0.001
})
```

#### Errors

```python
try:
    risky_operation()
except Exception as e:
    logger.log_error(e, context={'step': current_step})
    raise
```

#### Finalization

```python
logger.finalize(final_status='completed')  # or 'failed', 'interrupted'
```

### Log Formats

#### Human-Readable Log (`experiment.log`)

```
2025-12-31 05:30:00 - experiment.my_exp - INFO - Experiment configuration:
2025-12-31 05:30:00 - experiment.my_exp - INFO -   samples: 512
2025-12-31 05:30:01 - experiment.my_exp - INFO - Progress: 100/1000 (10.0%) | loss=0.5000
```

#### Structured Log (`experiment_structured.jsonl`)

```json
{"timestamp": "2025-12-31T05:30:00", "event_type": "config", "data": {"samples": 512}}
{"timestamp": "2025-12-31T05:30:01", "event_type": "progress", "data": {"iteration": 100, "total": 1000, "metrics": {"loss": 0.5}}}
```

### Log Analysis

```python
from src.experiment_logger import load_structured_logs, analyze_logs

# Load logs
events = load_structured_logs(Path("logs/my_exp_structured.jsonl"))

# Analyze
analysis = analyze_logs(Path("logs/my_exp_structured.jsonl"))
print(f"Total events: {analysis['total_events']}")
print(f"Event counts: {analysis['event_counts']}")
print(f"Status: {analysis['final_status']}")
```

## Checkpointing Module

### Features

- **Automatic Saving**: Based on iteration frequency
- **Rolling Checkpoints**: Keep last N checkpoints
- **Resume Capability**: Load and continue from checkpoint
- **Metadata**: Store context with checkpoints
- **Index Tracking**: JSON index of all checkpoints
- **Auto-backup**: Save on errors

### CheckpointManager Class

```python
from src.checkpointing import CheckpointManager

ckpt_mgr = CheckpointManager(
    checkpoint_dir=Path("./checkpoints"),
    experiment_name="long_exp",
    keep_last_n=3,  # Keep only 3 most recent
    save_frequency=100  # Auto-save every 100 iterations
)
```

### Saving Checkpoints

#### Manual Save

```python
state = {
    'iteration': 500,
    'theta': best_theta,
    'loss': current_loss,
    'rng_state': rng.bit_generator.state
}

checkpoint_path = ckpt_mgr.save_checkpoint(
    state=state,
    iteration=500,
    metadata={'note': 'Good convergence point'}
)
```

#### Automatic Save

```python
for iteration in range(1000):
    # Your training code
    
    if ckpt_mgr.should_save_checkpoint(iteration):
        state = get_current_state()
        ckpt_mgr.save_checkpoint(state, iteration=iteration)
```

### Loading Checkpoints

#### Load Latest

```python
checkpoint = ckpt_mgr.load_checkpoint()  # Load most recent

# Extract state
iteration = checkpoint['iteration']
theta = checkpoint['state']['theta']
timestamp = checkpoint['timestamp']
```

#### Load Specific

```python
checkpoints = ckpt_mgr.list_checkpoints()
checkpoint = ckpt_mgr.load_checkpoint(checkpoints[0])
```

### Resuming Experiments

```python
from src.checkpointing import resume_from_checkpoint

# Resume
checkpoint = resume_from_checkpoint(Path("checkpoints/exp_iter_005000.ckpt"))

# Continue from where you left off
start_iteration = checkpoint['state']['iteration'] + 1
theta = checkpoint['state']['theta']

for iteration in range(start_iteration, total_iterations):
    # Continue training
    pass
```

### Auto-Checkpoint Context Manager

Automatically save checkpoint on errors:

```python
from src.checkpointing import AutoCheckpoint

def get_state():
    return {'theta': current_theta, 'iteration': current_iter}

with AutoCheckpoint(ckpt_mgr, get_state):
    # Your code - checkpoint saved automatically on exception
    run_long_experiment()
```

## Complete Example

```python
from pathlib import Path
from src.experiment_logger import create_experiment_logger
from src.checkpointing import create_checkpoint_manager
from src.data_export import export_all_formats
import numpy as np

def run_logged_experiment(experiment_name="exp001"):
    # Setup
    output_dir = Path(f"./results/{experiment_name}")
    
    # Create logger
    logger = create_experiment_logger(
        experiment_name=experiment_name,
        output_dir=output_dir / "logs",
        verbose=True
    )
    
    # Create checkpoint manager
    ckpt_mgr = create_checkpoint_manager(
        experiment_name=experiment_name,
        checkpoint_dir=output_dir / "checkpoints",
        keep_last_n=3,
        save_frequency=100
    )
    
    # Log configuration
    config = {
        'samples': 512,
        'dimension': 8,
        'seed': 42,
        'iterations': 1000
    }
    logger.log_config(config)
    
    # Check for existing checkpoint
    try:
        checkpoint = ckpt_mgr.load_checkpoint()
        start_iter = checkpoint['state']['iteration'] + 1
        logger.info(f"Resuming from iteration {start_iter}")
    except FileNotFoundError:
        start_iter = 0
        logger.info("Starting new experiment")
    
    # Run experiment
    results = []
    try:
        for iteration in range(start_iter, 1000):
            # Your training code
            loss = np.random.rand()  # Placeholder
            
            # Log progress
            if iteration % 100 == 0:
                logger.log_progress(
                    iteration=iteration,
                    total=1000,
                    metrics={'loss': loss}
                )
            
            # Save checkpoint
            if ckpt_mgr.should_save_checkpoint(iteration):
                state = {'iteration': iteration, 'loss': loss}
                ckpt_mgr.save_checkpoint(state, iteration=iteration)
                logger.log_checkpoint({'iteration': iteration})
            
            # Collect results
            if iteration % 200 == 0:
                result = {'iteration': iteration, 'loss': loss}
                results.append(result)
        
        # Export results
        export_all_formats(
            results=results,
            output_dir=output_dir,
            base_name=experiment_name,
            metadata=config
        )
        
        logger.info("Results exported")
        logger.finalize(final_status='completed')
        
    except Exception as e:
        logger.log_error(e, context={'iteration': iteration})
        logger.finalize(final_status='failed')
        raise
```

## CLI Integration

Data export is automatically integrated into the CLI:

```bash
# Results are automatically saved
python -m src --samples 512 --save-results results.json
```

For logging and checkpointing in custom scripts, use the programmatic API.

## Best Practices

### Data Export

1. **Always export summaries**: Include statistics for quick reference
2. **Use JSON for sharing**: Most portable and readable
3. **Use pickle for exact replication**: Preserves numpy arrays exactly
4. **CSV for analysis**: Easy to load into Excel/R/Julia

### Logging

1. **Log early and often**: Start with config, end with finalize
2. **Use structured logs**: Machine-readable for automated analysis
3. **Include context in errors**: Helps debugging
4. **Set appropriate log levels**: DEBUG for development, INFO for production

### Checkpointing

1. **Save frequently for long runs**: Every 100-1000 iterations
2. **Keep multiple checkpoints**: Protects against corruption
3. **Include RNG state**: For exact reproducibility
4. **Test resume logic**: Verify it works before long runs
5. **Use auto-checkpoint for safety**: Context manager saves on errors

## File Organization

Recommended structure:

```
project/
└── results/
    └── experiment_001/
        ├── logs/
        │   ├── experiment_001.log
        │   └── experiment_001_structured.jsonl
        ├── checkpoints/
        │   ├── experiment_001_iter_000100.ckpt
        │   ├── experiment_001_iter_000200.ckpt
        │   ├── experiment_001_iter_000300.ckpt
        │   └── experiment_001_index.json
        ├── experiment_001.json
        ├── experiment_001.csv
        ├── experiment_001.pkl
        ├── experiment_001_summary.json
        └── visualizations/
            └── *.png
```

## Troubleshooting

### Large checkpoint files

**Solution**: Only save essential state, exclude large intermediate arrays

### Disk space issues

**Solution**: Reduce `keep_last_n`, increase `save_frequency`

### Log files too large

**Solution**: Use log rotation (implement with Python's `RotatingFileHandler`)

### Cannot resume from checkpoint

**Solution**: Verify pickle protocol compatibility, check file permissions

## Advanced Usage

### Custom NumpyEncoder

Already handles numpy types automatically. For custom types:

```python
from src.data_export import NumpyEncoder
import json

class CustomEncoder(NumpyEncoder):
    def default(self, obj):
        if isinstance(obj, MyCustomType):
            return {'__custom__': True, 'data': obj.to_dict()}
        return super().default(obj)

with open('output.json', 'w') as f:
    json.dump(data, f, cls=CustomEncoder)
```

### Checkpoint Backup

```python
from src.checkpointing import backup_checkpoint

# Backup important checkpoints
checkpoint_path = Path("checkpoints/exp_iter_001000.ckpt")
backup_checkpoint(checkpoint_path, Path("./backups"))
```

### Log Filtering

```python
from src.experiment_logger import load_structured_logs

events = load_structured_logs(Path("logs/exp_structured.jsonl"))

# Filter by event type
progress_events = [e for e in events if e['event_type'] == 'progress']
error_events = [e for e in events if e['event_type'] == 'error']
```

## Future Enhancements

- Database integration (SQLite/PostgreSQL)
- Cloud storage support (S3/GCS)
- Real-time monitoring dashboard
- Automatic experiment comparison
- Distributed checkpointing

---

**License:** MIT (see `LICENSE`) · **Contact:** x@christopheraltman.com  
Back to: `README.md`
