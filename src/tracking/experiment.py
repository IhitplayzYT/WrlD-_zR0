from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
from pathlib import Path
from datetime import datetime
import numpy as np


class TrackerType(Enum):
    """Types of experiment trackers"""
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
    MLFLOW = "mlflow"
    LOCAL = "local"


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    project_name: str = "rlaif_trading"
    experiment_name: str = None
    tracker_type: TrackerType = TrackerType.LOCAL
    log_dir: str = "./logs"
    track_hyperparameters: bool = True
    track_metrics: bool = True
    track_model: bool = True
    track_artifacts: bool = True
    save_frequency: int = 100


class LocalTracker:
    """
    Local file-based experiment tracker.
    Fallback when external trackers are not available.
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_log = []
        self.hyperparameters = {}
        self.artifacts = {}
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        self.hyperparameters.update(params)
        
        # Save to file
        with open(self.experiment_dir / "hyperparameters.json", 'w') as f:
            json.dump(self.hyperparameters, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a step"""
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_log.append(log_entry)
        
        # Save to file
        with open(self.experiment_dir / "metrics.jsonl", 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_artifact(self, name: str, artifact: Any):
        """Log artifact (model, file, etc.)"""
        artifact_path = self.experiment_dir / "artifacts" / name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(artifact, dict):
            with open(artifact_path, 'w') as f:
                json.dump(artifact, f, indent=2, default=str)
        else:
            # Assume it's a file path or saveable object
            try:
                import torch
                if isinstance(artifact, dict) and all(isinstance(v, torch.Tensor) for v in artifact.values()):
                    torch.save(artifact, artifact_path)
                else:
                    with open(artifact_path, 'wb') as f:
                        f.write(artifact)
            except:
                with open(artifact_path, 'w') as f:
                    f.write(str(artifact))
        
        self.artifacts[name] = str(artifact_path)
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log experiment summary"""
        with open(self.experiment_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def finish(self):
        """Finish experiment and save final state"""
        final_state = {
            'experiment_name': self.experiment_name,
            'total_steps': len(self.metrics_log),
            'hyperparameters': self.hyperparameters,
            'artifacts': self.artifacts,
            'completed_at': datetime.now().isoformat()
        }
        
        with open(self.experiment_dir / "final_state.json", 'w') as f:
            json.dump(final_state, f, indent=2)


class WandBTracker:
    """
    Weights & Biases tracker for advanced experiment tracking.
    """
    
    def __init__(self, project_name: str, experiment_name: str, config: ExperimentConfig):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.enabled = False
        
        try:
            import wandb
            self.wandb = wandb
            self.enabled = True
            
            # Initialize wandb
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config.__dict__,
                dir=config.log_dir
            )
        except ImportError:
            print("WandB not available, falling back to local tracking")
            self.local_tracker = LocalTracker(config.log_dir, experiment_name)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        if self.enabled:
            self.wandb.config.update(params)
        else:
            self.local_tracker.log_hyperparameters(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a step"""
        if self.enabled:
            self.wandb.log(metrics, step=step)
        else:
            self.local_tracker.log_metrics(metrics, step)
    
    def log_artifact(self, name: str, artifact: Any):
        """Log artifact"""
        if self.enabled:
            # Save artifact to file first
            artifact_path = Path(self.config.log_dir) / f"{name}.pt"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(artifact, dict):
                import torch
                if all(isinstance(v, torch.Tensor) for v in artifact.values()):
                    torch.save(artifact, artifact_path)
                else:
                    with open(artifact_path, 'w') as f:
                        json.dump(artifact, f, indent=2, default=str)
            
            # Log to wandb
            self.wandb.save(str(artifact_path))
        else:
            self.local_tracker.log_artifact(name, artifact)
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log experiment summary"""
        if self.enabled:
            self.wandb.summary.update(summary)
        else:
            self.local_tracker.log_summary(summary)
    
    def watch_model(self, model, log_freq: int = 100):
        """Watch model gradients and parameters"""
        if self.enabled:
            self.wandb.watch(model, log_freq=log_freq)
    
    def finish(self):
        """Finish experiment"""
        if self.enabled:
            self.wandb.finish()
        else:
            self.local_tracker.finish()


class ExperimentTracker:
    """
    Main experiment tracker that supports multiple backends.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_name = config.experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize appropriate tracker
        if config.tracker_type == TrackerType.WANDB:
            self.tracker = WandBTracker(config.project_name, self.experiment_name, config)
        elif config.tracker_type == TrackerType.LOCAL:
            self.tracker = LocalTracker(config.log_dir, self.experiment_name)
        else:
            # Default to local for other types
            self.tracker = LocalTracker(config.log_dir, self.experiment_name)
        
        # Metrics history
        self.metrics_history: Dict[str, List[float]] = {}
        self.step_history: List[int] = []
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        self.tracker.log_hyperparameters(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a step"""
        self.tracker.log_metrics(metrics, step)
        
        # Store in history
        self.step_history.append(step)
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
    
    def log_artifact(self, name: str, artifact: Any):
        """Log artifact"""
        self.tracker.log_artifact(name, artifact)
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log experiment summary"""
        self.tracker.log_summary(summary)
    
    def log_model_checkpoint(self, model, step: int, metrics: Optional[Dict] = None):
        """Log model checkpoint"""
        checkpoint_name = f"model_step_{step}"
        
        # Get model state dict
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
        else:
            state_dict = model
        
        checkpoint = {
            'step': step,
            'state_dict': state_dict,
            'metrics': metrics or {}
        }
        
        self.log_artifact(checkpoint_name, checkpoint)
    
    def log_agent_performance(self, agent_id: str, performance: Dict[str, float], step: int):
        """Log individual agent performance"""
        metrics = {f"{agent_id}/{k}": v for k, v in performance.items()}
        self.log_metrics(metrics, step)
    
    def log_market_state(self, market_state: Dict[str, Any], step: int):
        """Log market state for analysis"""
        # Extract key market metrics
        market_metrics = {
            'market/price': market_state.get('current_price', 0),
            'market/volatility': market_state.get('volatility', 0),
            'market/volume': market_state.get('volume', 0),
            'market/spread': market_state.get('spread', 0),
            'market/market_cap': market_state.get('market_cap', 0)
        }
        
        self.log_metrics(market_metrics, step)
    
    def log_training_progress(self, episode: int, reward: float, loss: float, **kwargs):
        """Log training progress"""
        metrics = {
            'training/episode': episode,
            'training/reward': reward,
            'training/loss': loss,
            **kwargs
        }
        self.log_metrics(metrics, episode)
    
    def log_evaluation_results(self, eval_results: Dict[str, float], step: int):
        """Log evaluation results"""
        metrics = {f"eval/{k}": v for k, v in eval_results.items()}
        self.log_metrics(metrics, step)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all logged metrics"""
        summary = {}
        
        for metric_name, values in self.metrics_history.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'final': values[-1]
                }
        
        return summary
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot metrics history"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            # Plot first 4 metrics
            for i, (metric_name, values) in enumerate(list(self.metrics_history.items())[:4]):
                if values:
                    axes[i].plot(self.step_history, values)
                    axes[i].set_title(metric_name)
                    axes[i].set_xlabel('Step')
                    axes[i].set_ylabel('Value')
                    axes[i].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def finish(self):
        """Finish experiment"""
        # Log final summary
        summary = self.get_metrics_summary()
        self.log_summary(summary)
        
        # Finish tracker
        self.tracker.finish()
        
        print(f"Experiment '{self.experiment_name}' completed")
        print(f"Metrics summary: {summary}")


class ExperimentManager:
    """
    Manage multiple experiments and comparisons.
    """
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentTracker] = {}
    
    def create_experiment(
        self,
        name: str,
        config: Optional[ExperimentConfig] = None
    ) -> ExperimentTracker:
        """Create a new experiment"""
        if config is None:
            config = ExperimentConfig(experiment_name=name)
        else:
            config.experiment_name = name
        
        tracker = ExperimentTracker(config)
        self.experiments[name] = tracker
        return tracker
    
    def get_experiment(self, name: str) -> Optional[ExperimentTracker]:
        """Get existing experiment"""
        return self.experiments.get(name)
    
    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare metrics across experiments"""
        comparison = {}
        
        for exp_name in experiment_names:
            tracker = self.experiments.get(exp_name)
            if tracker:
                comparison[exp_name] = tracker.get_metrics_summary()
        
        return comparison
    
    def load_experiment_logs(self, log_path: str) -> Dict[str, Any]:
        """Load experiment logs from file"""
        log_file = Path(log_path)
        
        if log_file.suffix == '.jsonl':
            # Load metrics log
            metrics = []
            with open(log_file, 'r') as f:
                for line in f:
                    metrics.append(json.loads(line))
            return {'metrics': metrics}
        elif log_file.suffix == '.json':
            # Load JSON file
            with open(log_file, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def generate_report(self, experiment_names: List[str], output_path: str):
        """Generate comparison report for experiments"""
        comparison = self.compare_experiments(experiment_names)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'experiments': comparison
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to {output_path}")


if __name__ == "__main__":
    # Test experiment tracker
    config = ExperimentConfig(
        project_name="test_project",
        experiment_name="test_experiment",
        tracker_type=TrackerType.LOCAL
    )
    
    tracker = ExperimentTracker(config)
    
    # Log hyperparameters
    tracker.log_hyperparameters({
        'learning_rate': 0.001,
        'batch_size': 32,
        'n_agents': 5
    })
    
    # Log some metrics
    for step in range(10):
        tracker.log_metrics({
            'reward': np.random.randn(),
            'loss': np.random.rand() * 0.1
        }, step)
    
    # Log summary
    summary = tracker.get_metrics_summary()
    tracker.log_summary(summary)
    
    # Finish
    tracker.finish()
    
    print("Experiment tracking test completed")
