import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import shutil


class CheckpointManager:
    """
    Manages checkpoints for long-running experiments.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        experiment_name: str,
        keep_last_n: int = 3,
        save_frequency: Optional[int] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            experiment_name: Name of experiment
            keep_last_n: Number of recent checkpoints to keep (0 = keep all)
            save_frequency: Save every N iterations (None = manual control)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.keep_last_n = keep_last_n
        self.save_frequency = save_frequency
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints
        self.checkpoint_counter = 0
        self.checkpoint_history = []
    
    def save_checkpoint(
        self,
        state: Dict[str, Any],
        iteration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            state: State dictionary to save
            iteration: Current iteration number
            metadata: Optional metadata
            
        Returns:
            Path to saved checkpoint
        """
        self.checkpoint_counter += 1
        
        # Create checkpoint name
        if iteration is not None:
            checkpoint_name = f"{self.experiment_name}_iter_{iteration:06d}.ckpt"
        else:
            checkpoint_name = f"{self.experiment_name}_ckpt_{self.checkpoint_counter:04d}.ckpt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'checkpoint_id': self.checkpoint_counter,
            'metadata': metadata or {},
            'state': state
        }
        
        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Track checkpoint
        self.checkpoint_history.append(checkpoint_path)
        
        # Clean old checkpoints if needed
        if self.keep_last_n > 0 and len(self.checkpoint_history) > self.keep_last_n:
            old_checkpoint = self.checkpoint_history.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        # Save checkpoint index
        self._save_checkpoint_index()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (None = load latest)
            
        Returns:
            Checkpoint data dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoints found")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        return checkpoint_data
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob(f"{self.experiment_name}*.ckpt"))
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob(f"{self.experiment_name}*.ckpt"))
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        return checkpoints
    
    def should_save_checkpoint(self, iteration: int) -> bool:
        """
        Check if checkpoint should be saved at this iteration.
        
        Args:
            iteration: Current iteration
            
        Returns:
            True if checkpoint should be saved
        """
        if self.save_frequency is None:
            return False
        
        return iteration % self.save_frequency == 0
    
    def _save_checkpoint_index(self):
        """Save index of all checkpoints."""
        index_path = self.checkpoint_dir / f"{self.experiment_name}_index.json"
        
        index_data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_count': self.checkpoint_counter,
            'checkpoints': [str(p) for p in self.checkpoint_history]
        }
        
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def delete_all_checkpoints(self):
        """Delete all checkpoints for this experiment."""
        for ckpt in self.list_checkpoints():
            ckpt.unlink()
        
        # Delete index
        index_path = self.checkpoint_dir / f"{self.experiment_name}_index.json"
        if index_path.exists():
            index_path.unlink()
        
        self.checkpoint_history.clear()
        self.checkpoint_counter = 0


def create_checkpoint_manager(
    experiment_name: str,
    checkpoint_dir: Path,
    keep_last_n: int = 3,
    save_frequency: Optional[int] = None
) -> CheckpointManager:
    """
    Convenience function to create checkpoint manager.
    
    Args:
        experiment_name: Name of experiment
        checkpoint_dir: Directory for checkpoints
        keep_last_n: Number of recent checkpoints to keep
        save_frequency: Save every N iterations
        
    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        keep_last_n=keep_last_n,
        save_frequency=save_frequency
    )


def resume_from_checkpoint(
    checkpoint_path: Path
) -> Dict[str, Any]:
    """
    Resume experiment from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint data
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    return checkpoint_data


def backup_checkpoint(checkpoint_path: Path, backup_dir: Path):
    """
    Create backup of checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        backup_dir: Backup directory
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    backup_path = backup_dir / checkpoint_path.name
    shutil.copy2(checkpoint_path, backup_path)


class AutoCheckpoint:
    """Context manager for automatic checkpointing."""
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        state_getter: callable
    ):
        """
        Initialize auto-checkpoint context.
        
        Args:
            checkpoint_manager: CheckpointManager instance
            state_getter: Function that returns current state dict
        """
        self.checkpoint_manager = checkpoint_manager
        self.state_getter = state_getter
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Save checkpoint on exit if exception occurred
        if exc_type is not None:
            try:
                state = self.state_getter()
                self.checkpoint_manager.save_checkpoint(
                    state=state,
                    metadata={'error': str(exc_val), 'interrupted': True}
                )
            except Exception as e:
                print(f"Failed to save checkpoint on error: {e}")
        
        return False  # Don't suppress exceptions
