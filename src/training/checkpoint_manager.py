""" JSON checkpoints with rollback"""
import json
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
from dataclasses import asdict

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, project_name: str = "WrlD_zR0"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name
        self.schema_version = "1.0"
        self._interrupted = False
        self._original_handlers = {}
    
    def _setup_signal_handlers(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.signal(sig, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        if self._interrupted:
            print("\nForce exit...")
            sys.exit(1)
        print("\nInterrupt signal received. Rolling back to last checkpoint...")
        self._interrupted = True
    
    def _restore_signal_handlers(self):
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
    
    def save_checkpoint(self, episode: int,config: Dict[str, Any],agent_states: Dict[str, Any],ensemble_state: Optional[Dict[str, Any]] = None,preference_state: Optional[Dict[str, Any]] = None,training_state: Optional[Dict[str, Any]] = None,episode_summaries: Optional[list] = None) -> str:
        checkpoint = {'schema_version': self.schema_version,'project': self.project_name,'timestamp': datetime.now().isoformat(),'episode': episode,'config': config,'agent_states': agent_states,'ensemble_state': ensemble_state,'preference_state': preference_state,'training_state': training_state,'episode_summaries': episode_summaries or []}
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.json"
        temp_path = self.checkpoint_dir / f"checkpoint_ep{episode}.tmp"
        with open(temp_path, 'w') as f: 
            json.dump(checkpoint, f, indent=2, default=self._json_serializer)
        temp_path.replace(checkpoint_path)
        # latest.json for resume
        latest_path = self.checkpoint_dir / "latest.json"
        temp_latest = self.checkpoint_dir / "latest.tmp"
        with open(temp_latest, 'w') as f: json.dump(checkpoint, f, indent=2, default=self._json_serializer)
        temp_latest.replace(latest_path)
        print(f"[*] Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        checkpoint_path = self.checkpoint_dir / "latest.json" if checkpoint_path is None else  Path(checkpoint_path)
        if not checkpoint_path.exists(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        if checkpoint.get('schema_version') != self.schema_version: print(f"Warning: Checkpoint schema version mismatch. Expected {self.schema_version}, got {checkpoint.get('schema_version')}")
        return checkpoint
    
    def get_best_checkpoint(self, metric: str = 'forecast_mse', mode: str = 'min') -> Optional[str]:
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_ep*.json"))
        if not checkpoints: return None
        best_path = None
        best = float('inf') if mode == 'min' else float('-inf')
        for x in checkpoints:
            try:
                data = self.load_checkpoint(str(x))
                summaries = data.get('episode_summaries', [])
                if not summaries: continue
                # Get metric from latest summary
                value = summaries[-1].get(metric)
                if value is None: continue
                if (mode == 'min' and value < best) or (mode == 'max' and value > best):
                    best = value
                    best_path = str(x)
            except Exception as e:
                print(f"Error reading {x}: {e}")
                continue
        return best_path
    
    def rollback_episode(self, current_episode: int) -> int:
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_ep*.json"), key=lambda p: int(p.stem.split('_ep')[1]))
        if not checkpoints: return 0
        last_checkpoint = checkpoints[-1]
        with open(last_checkpoint, 'r') as f:
            data = json.load(f)
        return data['episode']
    
    def _json_serializer(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if hasattr(obj, '__class__') and hasattr(obj.__class__, '__members__'): return obj.value if hasattr(obj, 'value') else str(obj)
        if hasattr(obj, '__dataclass_fields__'): return asdict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def cleanup(self): self._restore_signal_handlers()
    
    def __del__(self): self._restore_signal_handlers()
