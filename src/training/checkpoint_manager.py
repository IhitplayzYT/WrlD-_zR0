"""Deterministic JSON checkpoint system with rollback from market_ensemble_v2.
Human-readable checkpoints, atomic snapshots, rollback on interrupt.
"""
import json
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np


class CheckpointManager:
    """Manages deterministic checkpoints with atomic saves and rollback support."""
    
    def __init__(self, checkpoint_dir: str, project_name: str = "WrlD_zR0"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name
        self.schema_version = "1.0"
        self._interrupted = False
        self._original_handlers = {}
        # Signal handling belongs to the training loop, not this serializer.
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful interrupt handling."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.signal(sig, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle interrupt signals for graceful shutdown."""
        if self._interrupted:
            print("\nForce exit...")
            sys.exit(1)
        print("\nInterrupt signal received. Rolling back to last checkpoint...")
        self._interrupted = True
    
    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
    
    def save_checkpoint(self, 
                        episode: int,
                        config: Dict[str, Any],
                        agent_states: Dict[str, Any],
                        ensemble_state: Optional[Dict[str, Any]] = None,
                        preference_state: Optional[Dict[str, Any]] = None,
                        training_state: Optional[Dict[str, Any]] = None,
                        episode_summaries: Optional[list] = None) -> str:
        """Save a checkpoint atomically as human-readable JSON.
        
        Args:
            episode: Current episode number
            config: Training configuration
            agent_states: Dictionary of agent states
            ensemble_state: Optional ensemble model state
            preference_state: Optional preference model state
            training_state: Optional training-specific state
            episode_summaries: List of episode summaries
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'schema_version': self.schema_version,
            'project': self.project_name,
            'timestamp': datetime.now().isoformat(),
            'episode': episode,
            'config': config,
            'agent_states': agent_states,
            'ensemble_state': ensemble_state,
            'preference_state': preference_state,
            'training_state': training_state,
            'episode_summaries': episode_summaries or []
        }
        
        # Atomic save: write to temp file then rename
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.json"
        temp_path = self.checkpoint_dir / f"checkpoint_ep{episode}.tmp"
        
        with open(temp_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=self._json_serializer)
        
        temp_path.replace(checkpoint_path)
        
        # Also save as latest.json for easy resume
        latest_path = self.checkpoint_dir / "latest.json"
        temp_latest = self.checkpoint_dir / "latest.tmp"
        with open(temp_latest, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=self._json_serializer)
        temp_latest.replace(latest_path)
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load a checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads latest.json
        
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest.json"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        # Validate schema
        if checkpoint.get('schema_version') != self.schema_version:
            print(f"Warning: Checkpoint schema version mismatch. "
                  f"Expected {self.schema_version}, got {checkpoint.get('schema_version')}")
        
        return checkpoint
    
    def get_best_checkpoint(self, metric: str = 'forecast_mse', mode: str = 'min') -> Optional[str]:
        """Find best checkpoint based on metric.
        
        Args:
            metric: Metric name to optimize
            mode: 'min' or 'max'
        
        Returns:
            Path to best checkpoint or None
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_ep*.json"))
        if not checkpoints:
            return None
        
        best_path = None
        best_value = float('inf') if mode == 'min' else float('-inf')
        
        for cp in checkpoints:
            try:
                data = self.load_checkpoint(str(cp))
                summaries = data.get('episode_summaries', [])
                if not summaries:
                    continue
                
                # Get metric from latest summary
                value = summaries[-1].get(metric)
                if value is None:
                    continue
                
                if (mode == 'min' and value < best_value) or (mode == 'max' and value > best_value):
                    best_value = value
                    best_path = str(cp)
            except Exception as e:
                print(f"Error reading {cp}: {e}")
                continue
        
        return best_path
    
    def rollback_episode(self, current_episode: int) -> int:
        """Rollback to previous completed episode.
        
        Args:
            current_episode: Current episode number
        
        Returns:
            Episode number to resume from
        """
        # Find the last successfully completed checkpoint
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.json"), 
                           key=lambda p: int(p.stem.split('_ep')[1]))
        
        if not checkpoints:
            return 0
        
        last_checkpoint = checkpoints[-1]
        with open(last_checkpoint, 'r') as f:
            data = json.load(f)
        
        return data['episode']
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types, dataclasses, and enums."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle enums
        if hasattr(obj, '__class__') and hasattr(obj.__class__, '__members__'):
            return obj.value if hasattr(obj, 'value') else str(obj)
        # Handle dataclasses
        if hasattr(obj, '__dataclass_fields__'):
            from dataclasses import asdict
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def cleanup(self):
        """Cleanup and restore signal handlers."""
        self._restore_signal_handlers()
    
    def __del__(self):
        """Cleanup on deletion."""
        self._restore_signal_handlers()
