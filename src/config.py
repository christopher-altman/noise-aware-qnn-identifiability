import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Union
from dataclasses import dataclass, field, asdict
import itertools


@dataclass
class NoiseConfig:
    """Configuration for noise parameters."""
    p_dep: float = 0.0
    sigma_phase: float = 0.0
    
    def __post_init__(self):
        if not (0.0 <= self.p_dep <= 1.0):
            raise ValueError(f"p_dep must be in [0,1], got {self.p_dep}")
        if self.sigma_phase < 0.0:
            raise ValueError(f"sigma_phase must be >= 0, got {self.sigma_phase}")
    
    def to_tuple(self):
        return (self.p_dep, self.sigma_phase)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    samples: int = 512
    dimension: int = 8
    seed: int = 0
    noise_grid: List[NoiseConfig] = field(default_factory=list)
    optimizer_iterations: int = 2000
    hessian_eps: float = 1e-3
    output_dir: str = "./results"
    generate_plots: bool = True
    save_results: bool = True
    verbose: bool = False
    enhanced_metrics: bool = False
    extended_viz: bool = False
    
    def __post_init__(self):
        # Convert noise_grid dicts to NoiseConfig objects if needed
        if self.noise_grid and isinstance(self.noise_grid[0], dict):
            self.noise_grid = [NoiseConfig(**n) for n in self.noise_grid]
        
        # Validate parameters
        if self.samples <= 0:
            raise ValueError(f"samples must be > 0, got {self.samples}")
        if self.dimension <= 0:
            raise ValueError(f"dimension must be > 0, got {self.dimension}")
        if self.optimizer_iterations <= 0:
            raise ValueError(f"optimizer_iterations must be > 0, got {self.optimizer_iterations}")
        if self.hessian_eps <= 0:
            raise ValueError(f"hessian_eps must be > 0, got {self.hessian_eps}")
    
    def get_noise_tuples(self):
        """Get noise grid as list of tuples."""
        return [n.to_tuple() for n in self.noise_grid]
    
    def to_dict(self):
        """Convert to dictionary."""
        d = asdict(self)
        d['noise_grid'] = [{'p_dep': n.p_dep, 'sigma_phase': n.sigma_phase} 
                           for n in self.noise_grid]
        return d


@dataclass
class SweepConfig:
    """Configuration for hyperparameter sweeps."""
    name: str
    base_config: ExperimentConfig
    sweep_params: Dict[str, List[Any]]
    
    def generate_experiments(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations from the sweep."""
        experiments = []
        
        # Get parameter names and values
        param_names = list(self.sweep_params.keys())
        param_values = list(self.sweep_params.values())
        
        # Generate all combinations
        for idx, combination in enumerate(itertools.product(*param_values)):
            # Create a copy of base config
            config_dict = self.base_config.to_dict()
            
            # Update with sweep parameters
            for param_name, param_value in zip(param_names, combination):
                config_dict[param_name] = param_value
            
            # Create unique name
            suffix = "_".join(f"{k}={v}" for k, v in zip(param_names, combination))
            config_dict['name'] = f"{self.name}_{suffix}"
            
            # Create ExperimentConfig
            experiments.append(ExperimentConfig(**config_dict))
        
        return experiments


@dataclass
class BatchConfig:
    """Configuration for batch experiment runs."""
    name: str
    experiments: List[ExperimentConfig] = field(default_factory=list)
    sweeps: List[SweepConfig] = field(default_factory=list)
    parallel: bool = False
    max_workers: int = 4
    
    def get_all_experiments(self) -> List[ExperimentConfig]:
        """Get all experiments including those from sweeps."""
        all_experiments = list(self.experiments)
        
        for sweep in self.sweeps:
            all_experiments.extend(sweep.generate_experiments())
        
        return all_experiments


def load_config(path: Union[str, Path]) -> Union[ExperimentConfig, BatchConfig]:
    """
    Load experiment configuration from YAML or JSON file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        ExperimentConfig or BatchConfig instance
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    # Load file based on extension
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif path.suffix == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Determine config type
    if 'experiments' in data or 'sweeps' in data:
        return _parse_batch_config(data)
    else:
        return _parse_experiment_config(data)


def save_config(config: Union[ExperimentConfig, BatchConfig], 
                path: Union[str, Path],
                format: str = 'yaml'):
    """
    Save experiment configuration to file.
    
    Args:
        config: Configuration to save
        path: Output path
        format: Output format ('yaml' or 'json')
    """
    path = Path(path)
    
    # Convert to dict
    if isinstance(config, ExperimentConfig):
        data = config.to_dict()
    elif isinstance(config, BatchConfig):
        data = {
            'name': config.name,
            'parallel': config.parallel,
            'max_workers': config.max_workers,
            'experiments': [exp.to_dict() for exp in config.experiments],
        }
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")
    
    # Save file
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if format == 'yaml':
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif format == 'json':
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


def _parse_experiment_config(data: Dict) -> ExperimentConfig:
    """Parse dictionary into ExperimentConfig."""
    return ExperimentConfig(**data)


def _parse_batch_config(data: Dict) -> BatchConfig:
    """Parse dictionary into BatchConfig."""
    batch_dict = {
        'name': data.get('name', 'batch'),
        'parallel': data.get('parallel', False),
        'max_workers': data.get('max_workers', 4),
    }
    
    # Parse experiments
    if 'experiments' in data:
        batch_dict['experiments'] = [
            _parse_experiment_config(exp) for exp in data['experiments']
        ]
    
    # Parse sweeps
    if 'sweeps' in data:
        batch_dict['sweeps'] = []
        for sweep_data in data['sweeps']:
            base_config = _parse_experiment_config(sweep_data['base_config'])
            sweep = SweepConfig(
                name=sweep_data['name'],
                base_config=base_config,
                sweep_params=sweep_data['sweep_params']
            )
            batch_dict['sweeps'].append(sweep)
    
    return BatchConfig(**batch_dict)


def create_default_config(name: str = "default_experiment") -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig(
        name=name,
        samples=512,
        dimension=8,
        seed=0,
        noise_grid=[
            NoiseConfig(0.0, 0.0),
            NoiseConfig(0.05, 0.0),
            NoiseConfig(0.10, 0.0),
            NoiseConfig(0.10, 0.10),
            NoiseConfig(0.20, 0.20),
        ],
        optimizer_iterations=2000,
        hessian_eps=1e-3,
        output_dir="./results",
        generate_plots=True,
        save_results=True,
        verbose=False,
        enhanced_metrics=False,
        extended_viz=False,
    )


def create_noise_preset(preset: str) -> List[NoiseConfig]:
    """Create noise grid from preset name."""
    presets = {
        'minimal': [
            NoiseConfig(0.0, 0.0),
            NoiseConfig(0.1, 0.0),
        ],
        'default': [
            NoiseConfig(0.0, 0.0),
            NoiseConfig(0.05, 0.0),
            NoiseConfig(0.10, 0.0),
            NoiseConfig(0.10, 0.10),
            NoiseConfig(0.20, 0.20),
        ],
        'extensive': [
            NoiseConfig(0.0, 0.0),
            NoiseConfig(0.05, 0.0),
            NoiseConfig(0.10, 0.0),
            NoiseConfig(0.15, 0.0),
            NoiseConfig(0.10, 0.05),
            NoiseConfig(0.10, 0.10),
            NoiseConfig(0.15, 0.15),
            NoiseConfig(0.20, 0.20),
            NoiseConfig(0.25, 0.25),
        ],
        'high-noise': [
            NoiseConfig(0.0, 0.0),
            NoiseConfig(0.20, 0.0),
            NoiseConfig(0.30, 0.0),
            NoiseConfig(0.30, 0.30),
            NoiseConfig(0.40, 0.40),
        ],
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")
    
    return presets[preset]
