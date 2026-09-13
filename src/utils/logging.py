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
        compressed_files = sorted(log_dir.glob("*.json.gz"), key=lambda p: p.name)
        while len(compressed_files) > self.max_files:
            compressed_files[0].unlink()
            compressed_files.pop(0)


class JSONLogger:

    def __init__(self,log_dir: str = "./logs",log_file: str = "simulation.jsonl",enable_rotation: bool = True,enable_compression: bool = True,buffer_size: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_file
        if not self.log_file.exists():
            self.log_file.write_text("")

        self.enable_rotation = enable_rotation
        self.rotator = LogRotator() if enable_rotation else None
        self.enable_compression = enable_compression
        self.buffer: deque = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.lock = threading.Lock()

    
    def log(self,level: LogLevel,category: LogCategory,message: str,data: Optional[Dict[str, Any]] = None,source: str = "",agent_id: Optional[str] = None,episode: Optional[int] = None,step: Optional[int] = None):
        ent = LogEntry(datetime.now(),level,category,message,data if data else {},source,agent_id,episode,step)
        with self.lock:
            self.buffer.append(ent)
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
        n_levels = {}
        n_cats = {}
        n_agents = {}
        for entry in entries:
            n_levels[entry.level.value] = n_levels.get(entry.level.value, 0) + 1
            n_cats[entry.category.value] = n_cats.get(entry.category.value, 0) + 1
            if entry.agent_id:
                n_agents[entry.agent_id] = n_agents.get(entry.agent_id, 0) + 1            
        return {'total_entries': len(entries),'n_levels': n_levels,'n_cats': n_cats,'n_agents': n_agents,'time_range': {'start': min(e.timestamp for e in entries).isoformat(),'end': max(e.timestamp for e in entries).isoformat()}}
    
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

