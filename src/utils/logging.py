from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
import gzip
from pathlib import Path
from datetime import datetime
import threading
from collections import deque
import re


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogCategory(Enum):
    MARKET = "market"
    AGENT = "agent"
    EVENT = "event"
    TRAINING = "training"
    EVALUATION = "evaluation"
    SYSTEM = "system"
    PERFORMANCE = "performance"


@dataclass
class LogEntry:
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    agent_id: Optional[str] = None
    episode: Optional[int] = None
    step: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {'timestamp': self.timestamp.isoformat(),'level': self.level.value,'category': self.category.value,'message': self.message,'data': self.data,'source': self.source,'agent_id': self.agent_id,'episode': self.episode,'step': self.step}

    def from_dict(self,dict: Dict[str,Any]):
        self.timestamp = dict["timestamp"]
        self.level = dict["level"]
        self.category = dict["category"]
        self.message = dict["message"]
        self.data = dict["data"]
        self.source = dict["source"]
        self.agent_id = dict["agent_id"]
        self.episode = dict["episode"]
        self.step = dict["step"]
        

class LogFilter:
    def __init__(self):
        self.level_filter: Optional[LogLevel] = None
        self.category_filter: Optional[LogCategory] = None
        self.agent_filter: Optional[str] = None
        self.time_range: Optional[tuple[datetime, datetime]] = None
        self.message_pattern: Optional[str] = None
    
    def matches(self, entry: LogEntry) -> bool:
        if self.level_filter and entry.level != self.level_filter:
            return False
        
        if self.category_filter and entry.category != self.category_filter:
            return False
        
        if self.agent_filter and entry.agent_id != self.agent_filter:
            return False
        
        if self.time_range:
            start, end = self.time_range
            if (start > end) or not (start <= entry.timestamp <= end):
                return False
        
        if self.message_pattern and not re.search(self.message_pattern, entry.message, re.IGNORECASE):
            return False
        
        return True


class LogRotator:
    
    def __init__(self, max_file_size: int = 10 * 1024 * 1024, max_files: int = 10):
        self.max_file_size = max_file_size 
        self.max_files = max_files
    
    def should_rotate(self, log_file: Path) -> bool:
        if not log_file.exists():
            return False
        return log_file.stat().st_size >= self.max_file_size
    
    def rotate(self, log_file: Path):
        if not log_file.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        compressed_file = log_file.parent / f"{log_file.stem}_{timestamp}.json.gz"
        
        with open(log_file, 'rb') as f_in:
            with gzip.open(compressed_file, 'wb') as f_out:
                f_out.writelines(f_in)
        
        log_file.write_text("")
        self._cleanup_old_files(log_file.parent)
    
    def _cleanup_old_files(self, log_dir: Path):
        compressed_files = sorted(log_dir.glob("*.json.gz"), key=lambda p: p.stat().st_mtime)
        while len(compressed_files) > self.max_files:
            compressed_files[0].unlink()
            compressed_files.pop(0)


class JSONLogger:
    """ JSON-based logging system."""
    
    def __init__(self,log_dir: str = "./logs",log_file: str = "simulation.jsonl",enable_rotation: bool = True,enable_compression: bool = True,buffer_size: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_file
        self.enable_rotation = enable_rotation
        self.rotator = LogRotator() if enable_rotation else None
        self.enable_compression = enable_compression
        self.buffer: deque = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        if not self.log_file.exists():
            self.log_file.write_text("")
    
    def log(self,level: LogLevel,category: LogCategory,message: str,data: Optional[Dict[str, Any]] = None,source: str = "",agent_id: Optional[str] = None,episode: Optional[int] = None,step: Optional[int] = None):
        entry = LogEntry(timestamp=datetime.now(),level=level,category=category,message=message,data=data or {},source=source,agent_id=agent_id,episode=episode,step=step)
        with self.lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.buffer_size:
                self.flush()
    
    def debug(self, category: LogCategory, message: str, **kwargs):
        self.log(LogLevel.DEBUG, category, message, **kwargs)
    
    def info(self, category: LogCategory, message: str, **kwargs):
        self.log(LogLevel.INFO, category, message, **kwargs)
    
    def warning(self, category: LogCategory, message: str, **kwargs):
        self.log(LogLevel.WARNING, category, message, **kwargs)
    
    def error(self, category: LogCategory, message: str, **kwargs):
        self.log(LogLevel.ERROR, category, message, **kwargs)
    
    def critical(self, category: LogCategory, message: str, **kwargs):
        self.log(LogLevel.CRITICAL, category, message, **kwargs)
    
    def flush(self):
        if not self.buffer:
            return
        if self.enable_rotation and self.rotator.should_rotate(self.log_file):
            self.rotator.rotate(self.log_file)
        with open(self.log_file, 'a') as f:
            for entry in self.buffer:
                f.write(json.dumps(entry.to_dict()) + '\n')
        self.buffer.clear()
    
    def query(self,log_filter: Optional[LogFilter] = None,limit: int = 1000) -> List[LogEntry]:
        entries = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        entry = LogEntry(timestamp=datetime.fromisoformat(data['timestamp']),level=LogLevel(data['level']),category=LogCategory(data['category']),message=data['message'],data=data.get('data', {}),source=data.get('source', ''),agent_id=data.get('agent_id'),episode=data.get('episode'),step=data.get('step'))
                        if log_filter is None or log_filter.matches(entry):
                            entries.append(entry)
                            if len(entries) >= limit:
                                break
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        if self.enable_compression and len(entries) < limit:
            for compressed_file in sorted(self.log_dir.glob("*.json.gz"), reverse=True):
                if len(entries) >= limit:
                    break
                try:
                    with gzip.open(compressed_file, 'rt') as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                entry = LogEntry(timestamp=datetime.fromisoformat(data['timestamp']),level=LogLevel(data['level']),category=LogCategory(data['category']),message=data['message'],data=data.get('data', {}),source=data.get('source', ''),agent_id=data.get('agent_id'),episode=data.get('episode'),step=data.get('step'))
                                if log_filter is None or log_filter.matches(entry):
                                    entries.append(entry)
                                    if len(entries) >= limit:
                                        break
                            except (json.JSONDecodeError, KeyError):
                                continue
                except Exception:
                    continue
        
        return entries
    
    def get_statistics(self) -> Dict[str, Any]:
        entries = self.query(limit=10000) 
        if not entries:
            return {'total_entries': 0}
        level_counts = {}
        for entry in entries:
            level_counts[entry.level.value] = level_counts.get(entry.level.value, 0) + 1
        category_counts = {}
        for entry in entries:
            category_counts[entry.category.value] = category_counts.get(entry.category.value, 0) + 1
        agent_counts = {}
        for entry in entries:
            if entry.agent_id:
                agent_counts[entry.agent_id] = agent_counts.get(entry.agent_id, 0) + 1
        return {'total_entries': len(entries),'level_counts': level_counts,'category_counts': category_counts,'agent_counts': agent_counts,'time_range': {'start': min(e.timestamp for e in entries).isoformat(),'end': max(e.timestamp for e in entries).isoformat()}}
    
    def analyze_anomalies(self, threshold: float = 3.0) -> List[Dict[str, Any]]:
        entries = self.query(limit=10000)
        if not entries:
            return []
        anomalies = []
        
        # Check for error spikes
        error_entries = [e for e in entries if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        if len(error_entries) > len(entries) * 0.1:  # More than 10% errors
            anomalies.append({'type': 'error_spike','description': f'High error rate: {len(error_entries)}/{len(entries)}','severity': 'high'})
        
        # Check for unusual patterns in data
        for entry in entries:
            if entry.data:
                for key, value in entry.data.items():
                    if isinstance(value, (int, float)):
                        all_values = [e.data.get(key) for e in entries if key in e.data and isinstance(e.data[key], (int, float))]
                        if all_values:
                            mean = sum(all_values) / len(all_values)
                            std = (sum((x - mean) ** 2 for x in all_values) / len(all_values)) ** 0.5
                            if std > 0 and abs(value - mean) > threshold * std:
                                anomalies.append({'type': 'value_outlier','description': f'Outlier in {key}: {value}','timestamp': entry.timestamp.isoformat(),'severity': 'medium'})
        return anomalies
    
    def export_logs(self,output_file: str,log_filter: Optional[LogFilter] = None,format: str = 'json'):
        entries = self.query(log_filter, limit=100000)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump([e.to_dict() for e in entries], f, indent=2)
        elif format == 'jsonl':
            with open(output_path, 'w') as f:
                for entry in entries:
                    f.write(json.dumps(entry.to_dict()) + '\n')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear_logs(self):
        self.buffer.clear()
        if self.log_file.exists():
            self.log_file.write_text("")
        for compressed_file in self.log_dir.glob("*.json.gz"):
            compressed_file.unlink()


class LogMonitor:
    """Real time log monitoring"""

    def __init__(self, logger: JSONLogger):
        self.logger = logger
        self.callbacks: List[Callable[[LogEntry], None]] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def add_callback(self, callback: Callable[[LogEntry], None]):
        """Add callback for new log entries"""
        self.callbacks.append(callback)
    
    def start_monitoring(self, interval: float = 1.0):
        if self.monitoring:
            return
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_loop(self, interval: float):
        last_count = 0
        while self.monitoring:
            entries = self.logger.query(limit=100)
            for entry in entries[last_count:]:
                for callback in self.callbacks:
                    try:
                        callback(entry)
                    except Exception:
                        pass
            last_count = len(entries)
            import time
            time.sleep(interval)
    
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)


# Global logger instance
_global_logger: Optional[JSONLogger] = None


def get_logger() -> JSONLogger:
    """Initialise and get Global_logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = JSONLogger()
    return _global_logger


def setup_logger(log_dir: str = "./logs",log_file: str = "simulation.jsonl",enable_rotation: bool = True,enable_compression: bool = True) -> JSONLogger:
    global _global_logger
    _global_logger = JSONLogger(log_dir=log_dir,log_file=log_file,enable_rotation=enable_rotation,enable_compression=enable_compression)
    return _global_logger


if __name__ == "__main__":
    # Test logging system
    logger = setup_logger(log_dir="./test_logs")
    # Log some entries
    logger.info(LogCategory.MARKET, "Market initialized", data={'price': 100.0})
    logger.debug(LogCategory.AGENT, "Agent decision made", agent_id="agent_0", step=10)
    logger.warning(LogCategory.EVENT, "Event triggered", data={'event_type': 'earnings'})
    logger.error(LogCategory.SYSTEM, "Connection error", data={'error': 'timeout'})
    logger.flush()
    
    # Query logs
    filter = LogFilter()
    filter.level_filter = LogLevel.INFO
    entries = logger.query(filter, limit=10)
    print(f"Queried {len(entries)} entries")
    for entry in entries:
        print(f"[{entry.level.value}] {entry.message}")
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"\nStatistics: {stats}")
    
    logger.clear_logs()
    import shutil
    shutil.rmtree("./test_logs")
    
    print("Test completed")
