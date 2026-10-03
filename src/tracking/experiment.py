from src.utils.logging import JSONLogger, get_logger, LogCategory
from typing import Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime

class ExperimentTracker:
    def __init__(self, log_dir: str = "./logs", exp_name: Optional[str] = None, logger: Optional[JSONLogger] = None):
        self.log_dir = Path(log_dir)
        self.logger = logger or get_logger()
        self.exp_name = exp_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = self.log_dir / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log = []
        self.hyperparams = {}
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        self.hyperparams.update(params)
        with open(self.exp_dir / "hyperparameters.json", 'w') as f:
            json.dump(self.hyperparams, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        log_entry = {'step': step, 'timestamp': datetime.now().isoformat(), **metrics}
        self.metrics_log.append(log_entry)
        with open(self.exp_dir / "metrics.jsonl", 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_summary(self, summary: Dict[str, Any]):
        with open(self.exp_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def finish(self):
        with open(self.exp_dir / "final_state.json", 'w') as f:
            json.dump({'experiment_name': self.exp_name,'total_steps': len(self.metrics_log),'hyperparameters': self.hyperparams,'completed_at': datetime.now().isoformat()}, f, indent=2)
        self.logger.info(LogCategory.SYSTEM, f"Experiment '{self.exp_dir}' completed", data={'experiment_dir': str(self.exp_dir)})
