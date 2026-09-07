from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime
from src.utils.logging import get_logger, LogCategory

logger = get_logger()

class ExperimentTracker:
    
    def __init__(self, log_dir: str = "./logs", experiment_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log = []
        self.hyperparameters = {}
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        self.hyperparameters.update(params)
        with open(self.experiment_dir / "hyperparameters.json", 'w') as f:
            json.dump(self.hyperparameters, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        log_entry = {'step': step, 'timestamp': datetime.now().isoformat(), **metrics}
        self.metrics_log.append(log_entry)
        with open(self.experiment_dir / "metrics.jsonl", 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_summary(self, summary: Dict[str, Any]):
        with open(self.experiment_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def finish(self):
        final_state = {
            'experiment_name': self.experiment_name,
            'total_steps': len(self.metrics_log),
            'hyperparameters': self.hyperparameters,
            'completed_at': datetime.now().isoformat()
        }
        with open(self.experiment_dir / "final_state.json", 'w') as f:
            json.dump(final_state, f, indent=2)
        logger.info(LogCategory.SYSTEM, f"Experiment '{self.experiment_name}' completed", data={'experiment_dir': str(self.experiment_dir)})

