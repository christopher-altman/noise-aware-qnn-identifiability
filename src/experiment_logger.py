import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import sys


class ExperimentLogger:
    """
    Structured logger for experiment tracking.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: Path,
        log_to_file: bool = True,
        log_to_console: bool = True,
        level: int = logging.INFO
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory for log files
            log_to_file: Enable file logging
            log_to_console: Enable console logging
            level: Logging level
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(f"experiment.{experiment_name}")
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Clear any existing handlers
        
        # File handler
        if log_to_file:
            log_file = self.output_dir / f"{experiment_name}.log"
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(
                '%(levelname)s: %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Structured log storage
        self.structured_log_path = self.output_dir / f"{experiment_name}_structured.jsonl"
        
        # Start experiment
        self.log_event('experiment_start', {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat()
        })
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log structured event to JSONL file.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        
        with open(self.structured_log_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.logger.info("Experiment configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.log_event('config', config)
    
    def log_progress(self, iteration: int, total: int, metrics: Optional[Dict[str, Any]] = None):
        """
        Log progress update.
        
        Args:
            iteration: Current iteration
            total: Total iterations
            metrics: Optional metrics dict
        """
        progress_pct = (iteration / total) * 100 if total > 0 else 0
        msg = f"Progress: {iteration}/{total} ({progress_pct:.1f}%)"
        
        if metrics:
            metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                  for k, v in metrics.items())
            msg += f" | {metric_str}"
        
        self.logger.info(msg)
        self.log_event('progress', {
            'iteration': iteration,
            'total': total,
            'progress_pct': progress_pct,
            'metrics': metrics or {}
        })
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step/iteration number
        """
        metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in metrics.items())
        
        if step is not None:
            self.logger.info(f"Step {step} - Metrics: {metric_str}")
        else:
            self.logger.info(f"Metrics: {metric_str}")
        
        self.log_event('metrics', {
            'step': step,
            'metrics': metrics
        })
    
    def log_result(self, result: Dict[str, Any]):
        """Log experiment result."""
        self.logger.info("Experiment result:")
        for key, value in result.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")
        self.log_event('result', result)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Log error with context.
        
        Args:
            error: Exception object
            context: Optional context information
        """
        self.logger.error(f"Error occurred: {str(error)}")
        if context:
            self.logger.error(f"Context: {context}")
        
        self.log_event('error', {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        })
    
    def log_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved")
        self.log_event('checkpoint', checkpoint_data)
    
    def finalize(self, final_status: str = 'completed'):
        """
        Finalize experiment logging.
        
        Args:
            final_status: Final status (completed, failed, interrupted)
        """
        self.log_event('experiment_end', {
            'experiment_name': self.experiment_name,
            'end_time': datetime.now().isoformat(),
            'status': final_status
        })
        self.logger.info(f"Experiment {final_status}: {self.experiment_name}")
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)


def create_experiment_logger(
    experiment_name: str,
    output_dir: Path,
    verbose: bool = False
) -> ExperimentLogger:
    """
    Convenience function to create experiment logger.
    
    Args:
        experiment_name: Name of experiment
        output_dir: Output directory
        verbose: Enable verbose (DEBUG) logging
        
    Returns:
        ExperimentLogger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    return ExperimentLogger(
        experiment_name=experiment_name,
        output_dir=output_dir,
        log_to_file=True,
        log_to_console=True,
        level=level
    )


def load_structured_logs(log_path: Path) -> list:
    """
    Load structured logs from JSONL file.
    
    Args:
        log_path: Path to structured log file
        
    Returns:
        List of log events
    """
    events = []
    with open(log_path, 'r') as f:
        for line in f:
            events.append(json.loads(line.strip()))
    return events


def analyze_logs(log_path: Path) -> Dict[str, Any]:
    """
    Analyze structured logs and extract summary statistics.
    
    Args:
        log_path: Path to structured log file
        
    Returns:
        Dictionary with log analysis
    """
    events = load_structured_logs(log_path)
    
    # Count event types
    event_counts = {}
    for event in events:
        event_type = event['event_type']
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    
    # Extract timing information
    start_time = None
    end_time = None
    for event in events:
        if event['event_type'] == 'experiment_start':
            start_time = event['timestamp']
        elif event['event_type'] == 'experiment_end':
            end_time = event['timestamp']
    
    # Extract final status
    final_status = None
    for event in reversed(events):
        if event['event_type'] == 'experiment_end':
            final_status = event['data'].get('status')
            break
    
    analysis = {
        'total_events': len(events),
        'event_counts': event_counts,
        'start_time': start_time,
        'end_time': end_time,
        'final_status': final_status
    }
    
    return analysis
