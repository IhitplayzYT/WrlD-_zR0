import time
from typing import Dict, List, Optional, Tuple,Any
from dataclasses import dataclass, field
from enum import Enum
import random
from datetime import datetime
import random
from src.utils.logging import JSONLogger, get_logger, LogCategory

class MarketSession(Enum):
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"

class TimeZone(Enum):
    EST = "est" 
    PST = "pst" 
    GMT = "gmt" 
    JST = "jst" 
    CET = "cet" 

@dataclass
class AgentTimeRestriction:
    agent_id: str
    allow_st: int  # (0-indexed)
    allow_ed: int  # (exclusive)
    is_blocked: bool = False
    rem_block: int = 0
    perf_score: float = 0.5
    hist: List[Tuple[int, int]] = field(default_factory=list)


class TimeManager:
    """
    - Time based access
    - bbox periods
    - Agent blocka and killing
    """
    
    def __init__(self, day_duration_seconds: float = 90.0, movements_per_day: int = 10, black_box_duration: int = 5, black_box_frequency: int = 20, max_agents: int = 20, enable_dynamic_adjustment: bool = True, enable_market_sessions: bool = True, timezone: TimeZone = TimeZone.EST, logger: Optional[JSONLogger] = None):
        self.logger = logger or get_logger()
        self.day_dur = day_duration_seconds
        self.mov_per_day = movements_per_day
        self.bbox_dur = black_box_duration
        self.bbox_freq = black_box_frequency
        self.mx_agents = max_agents
        self.dynamic_adj = enable_dynamic_adjustment
        self.market_sess = enable_market_sessions
        self.timezone = timezone
        self.curr_day = 0
        self.curr_movement = 0
        self.tot_steps = 0
        self.sim_st_time = None
        self.day_st_time = None
        self.is_bbox = False
        self.bbox_counter = 0
        self.bbox_t_rem = 0
        self.restrictions: Dict[str, AgentTimeRestriction] = {}
        self.curr_sess = MarketSession.CLOSED
        self.sess_movements = {MarketSession.PRE_MARKET: (0, 2), MarketSession.REGULAR: (2, 8),    MarketSession.AFTER_HOURS: (8, 10)}
        self.p_block = 0.001 
        self.p_kill = 0.0001 
        self.history: Dict[str, List[float]] = {}
        self.adj_freq = 50  # Adjust windows every 50 steps
    
    def start_simulation(self):
        self.sim_st_time = self.day_st_time = time.time()

    def advance_movement(self) -> bool:
        self.curr_movement += 1
        self.tot_steps += 1
        
        if self.market_sess:
            self.curr_sess = self._get_current_session()
        
        if self.dynamic_adj and self.tot_steps % self.adj_freq == 0:
            self._adjust_time_windows()
        
        if self.curr_movement >= self.mov_per_day:
            self.curr_movement = 0
            self.curr_day+= 1
            self.day_st_time = time.time()
            self.curr_sess = MarketSession.CLOSED
            return True
        return False
    
    def _get_current_session(self) -> MarketSession:
        for session, (start, end) in self.sess_movements.items():
            if start <= self.curr_movement < end:
                return session
        return MarketSession.CLOSED
    
    def get_current_session(self) -> MarketSession:
        return self.curr_sess
    
    def is_market_open(self) -> bool:
        return self.curr_sess in [MarketSession.PRE_MARKET, MarketSession.REGULAR, MarketSession.AFTER_HOURS]
    
    def get_day_progress(self) -> float:
        return self.curr_movement / self.mov_per_day
    
    def get_time_until_day_end(self) -> float:
        return max(0.0,  self.day_dur - (time.time() - self.day_st_time))
    
    def should_trigger_black_box(self) -> bool:
        self.bbox_counter += 1
        if not self.is_bbox and self.bbox_counter >= self.bbox_freq:
            self.is_bbox = True
            self.bbox_t_rem = self.bbox_dur
            self.bbox_counter = 0
            return True
        return False
    
    def update_black_box(self):
        if self.is_bbox:
            self.bbox_t_rem -= 1
            if self.bbox_t_rem <= 0:
                self.is_bbox = False
    
    def add_agent_restriction(self,agent_id: str,start_movement: int,end_movement: int):
        self.restrictions[agent_id] = AgentTimeRestriction(agent_id,start_movement,end_movement)
    
    def can_agent_trade(self, agent_id: str) -> bool:
        restriction = self.restrictions.get(agent_id)
        if restriction is None:
            return True
        if restriction.is_blocked:
            return False
        if self.market_sess and not self.is_market_open():
            return False
        if (restriction.allow_st <= self.curr_movement < restriction.allow_ed):
            return True
        return False
    
    def update_agent_performance(self, agent_id: str, performance: float):
        if agent_id not in self.history:
            self.history[agent_id] = []
        self.history[agent_id].append(performance)

        if len(self.history[agent_id]) > 100:
            self.history[agent_id].pop(0)

        if agent_id in self.restrictions:
            recent_perf = self.history[agent_id][-10:]
            avg_perf = sum(recent_perf) / len(recent_perf) if recent_perf else 0.5
            self.restrictions[agent_id].perf_score = avg_perf
    
    def _adjust_time_windows(self):
        if not self.dynamic_adj:
            return
        for agent_id, restriction in self.restrictions.items():
            perf_history = self.history.get(agent_id, [])
            if len(perf_history) < 10:
                continue
            recent_perf = sum(perf_history[-10:]) / 10
            current_window = restriction.allow_ed - restriction.allow_st
            if recent_perf > 0.7 and current_window < self.mov_per_day - 2:
                new_start = max(0, restriction.allow_st - 1)
                new_end = min(self.mov_per_day, restriction.allow_ed + 1)
                restriction.hist.append((new_start, new_end))
                restriction.allow_st = new_start
                restriction.allow_ed = new_end
            elif recent_perf < 0.3 and current_window > 2:
                new_start = min(self.mov_per_day - 2, restriction.allow_st + 1)
                new_end = max(2, restriction.allow_ed - 1)
                restriction.hist.append((new_start, new_end))
                restriction.allow_st = new_start
                restriction.allow_ed = new_end
    
    def block_agent(self, agent_id: str, duration: int):
        if agent_id in self.restrictions:
            self.restrictions[agent_id].is_blocked = True
            self.restrictions[agent_id].rem_block = duration
        else:
            self.agent_restrictions[agent_id] = AgentTimeRestriction(agent_id=agent_id,allow_st=0,allow_ed=self.mov_per_day,is_blocked=True,rem_block=duration)
    
    def update_agent_blocks(self):
        for agent_id, restriction in self.restrictions.items():
            if restriction.is_blocked and restriction.rem_block > 0:
                restriction.rem_block -= 1
                if restriction.rem_block <= 0:
                    restriction.is_blocked = False
    
    def trigger_random_block_event(self, agent_ids: List[str]) -> Optional[str]:
        if random.random() < self.p_block and agent_ids:
            blocked_agent = random.choice(agent_ids)
            duration = random.randint(5, 20)
            self.block_agent(blocked_agent, duration)
            return blocked_agent
        return None
    
    def trigger_random_kill_event(self, agent_ids: List[str]) -> Optional[str]:
        if random.random() < self.p_kill and agent_ids:
            killed_agent = random.choice(agent_ids)
            duration = self.mov_per_day * 9999999 
            self.block_agent(killed_agent, duration)
            return killed_agent
        return None
    
    def get_agent_status(self, agent_id: str) -> Dict:
        restriction = self.restrictions.get(agent_id)
        if restriction is None:
            return {'agent_id': agent_id,'can_trade': True,'is_blocked': False,'block_remaining': 0,'time_window': None}
        return {'agent_id': agent_id,'can_trade': self.can_agent_trade(agent_id),'is_blocked': restriction.is_blocked,'block_remaining': restriction.rem_block,'time_window': (restriction.allow_st, restriction.allow_ed)}
    
    def initialize_default_restrictions(self, agent_ids: List[str]):
        n_agents = len(agent_ids)
        e_lb,e_ub = int(self.mov_per_day * 0.1) ,int(self.mov_per_day * 0.3)
        early_count = max(1, n_agents // 3)
        for i in range(early_count):
            if i < len(agent_ids):
                self.add_agent_restriction(agent_ids[i], 0, random.randint(e_lb,e_ub))
        m_lb,m_ub = int(self.mov_per_day * 0.4) ,int(self.mov_per_day * 0.6)
        mid_count = max(1, n_agents // 3)
        for i in range(early_count, early_count + mid_count):
            if i < len(agent_ids):
                self.add_agent_restriction(agent_ids[i], random.randint(e_lb,e_ub), random.randint(m_lb,m_ub))
        r_lb,r_ub = int(self.mov_per_day * 0.7) ,int(self.mov_per_day * 0.9)
        for i in range(early_count + mid_count, n_agents):
            if i < len(agent_ids):
                self.add_agent_restriction(agent_ids[i],random.randint(r_lb,r_ub), self.mov_per_day)
    
    def get_time_diversity_metrics(self) -> Dict[str, Any]:
        if not self.restrictions:
            return {'diversity_score': 0.0}        
        windows = [(r.allow_st, r.allow_ed) for r in self.restrictions.values()]
        total_overlap = n_pairs = 0
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                start1, end1 = windows[i]
                start2, end2 = windows[j]
                overlap = max(0, min(end1, end2) - max(start1, start2))
                total_overlap += overlap
                n_pairs += 1
        avg_overlap = (total_overlap / n_pairs) if n_pairs > 0 else 0
        max_possible_overlap = self.mov_per_day
        diversity_score = 1.0 - (avg_overlap / max_possible_overlap)
        return {'diversity_score': diversity_score,'avg_overlap': avg_overlap,'num_windows': len(windows),'window_sizes': [end - start for start, end in windows]}
    
    def convert_to_real_time(self, movement: int) -> datetime:
        if self.day_st_time is None:
            return datetime.now()
        seconds_per_movement = self.day_dur / self.mov_per_day
        elapsed_seconds = movement * seconds_per_movement
        real_time = datetime.fromtimestamp(self.day_st_time + elapsed_seconds)
        return real_time
    
    def get_timezone_offset(self) -> int:
        offsets = {TimeZone.EST: -5,TimeZone.PST: -8,TimeZone.GMT: 0,TimeZone.JST: 9,TimeZone.CET: 1}
        return offsets.get(self.timezone, 0)
    
    def reset(self):
        self.curr_day = self.curr_movement = self.tot_steps = self.bbox_counter = self.bbox_t_rem = 0
        self.sim_st_time = self.day_st_time = None
        self.is_bbox = False
        self.restrictions.clear()
        self.history.clear()
        self.curr_sess = MarketSession.CLOSED
    
    def get_state(self) -> Dict:
        return {'current_day': self.curr_day,'current_movement': self.curr_movement,'total_steps': self.tot_steps,'day_progress': self.get_day_progress(),'is_black_box': self.is_bbox,'black_box_remaining': self.bbox_t_rem,'time_until_day_end': self.get_time_until_day_end(),'active_restrictions': len(self.restrictions)}