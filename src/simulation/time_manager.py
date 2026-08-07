import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class AgentTimeRestriction:
    agent_id: str
    allowed_movements_start: int  # Starting movement number (0-indexed)
    allowed_movements_end: int    # Ending movement number (exclusive)
    is_blocked: bool = False
    block_remaining: int = 0
    block_reason: Optional[str] = None


class TimeManager:
    """
    Manages time-based aspects of the simulation including:
    - Day cycles (1-2 mins per day)
    - Intra-day movements (10 movements per day)
    - Timestamp-based agent access restrictions
    - Black box periods
    - Rare agent blocking/killing events
    """
    
    def __init__(
        self,
        day_duration_seconds: float = 90.0,  # 1.5 mins per day
        movements_per_day: int = 10,
        black_box_duration: int = 5,
        black_box_frequency: int = 20,
        max_agents: int = 20
    ):
        self.day_duration_seconds = day_duration_seconds
        self.movements_per_day = movements_per_day
        self.black_box_duration = black_box_duration
        self.black_box_frequency = black_box_frequency
        self.max_agents = max_agents
        
        # Time tracking
        self.current_day = 0
        self.current_movement = 0
        self.total_steps = 0
        self.simulation_start_time = None
        self.day_start_time = None
        
        # Black box tracking
        self.is_black_box = False
        self.black_box_counter = 0
        self.black_box_remaining = 0
        
        # Agent time restrictions
        self.agent_restrictions: Dict[str, AgentTimeRestriction] = {}
        
        # Rare event probabilities
        self.block_probability = 0.001  # 0.1% chance per step
        self.kill_probability = 0.0001  # 0.01% chance per step (very rare)
    
    def start_simulation(self):
        """Start the simulation timer"""
        self.simulation_start_time = time.time()
        self.day_start_time = time.time()
    
    def advance_movement(self) -> bool:
        """
        Advance to the next movement within the current day.
        Returns True if day has ended, False otherwise.
        """
        self.current_movement += 1
        self.total_steps += 1
        
        # Check if day has ended
        if self.current_movement >= self.movements_per_day:
            self.current_movement = 0
            self.current_day += 1
            self.day_start_time = time.time()
            return True
        
        return False
    
    def get_day_progress(self) -> float:
        """Get progress of current day as a percentage (0.0 to 1.0)"""
        return self.current_movement / self.movements_per_day
    
    def get_time_until_day_end(self) -> float:
        """Get seconds remaining until current day ends"""
        elapsed = time.time() - self.day_start_time
        remaining = self.day_duration_seconds - elapsed
        return max(0.0, remaining)
    
    def should_trigger_black_box(self) -> bool:
        """Check if black box period should be triggered"""
        self.black_box_counter += 1
        
        if not self.is_black_box and self.black_box_counter >= self.black_box_frequency:
            self.is_black_box = True
            self.black_box_remaining = self.black_box_duration
            self.black_box_counter = 0
            return True
        
        return False
    
    def update_black_box(self):
        """Update black box state"""
        if self.is_black_box:
            self.black_box_remaining -= 1
            if self.black_box_remaining <= 0:
                self.is_black_box = False
    
    def add_agent_restriction(
        self,
        agent_id: str,
        start_movement: int,
        end_movement: int
    ):
        """
        Add time-based access restriction for an agent.
        Agent can only trade during movement range [start_movement, end_movement).
        """
        self.agent_restrictions[agent_id] = AgentTimeRestriction(
            agent_id=agent_id,
            allowed_movements_start=start_movement,
            allowed_movements_end=end_movement
        )
    
    def can_agent_trade(self, agent_id: str) -> bool:
        """Check if agent is allowed to trade at current time"""
        restriction = self.agent_restrictions.get(agent_id)
        
        if restriction is None:
            # No restriction means agent can always trade
            return True
        
        # Check if agent is blocked
        if restriction.is_blocked:
            return False
        
        # Check time window
        if (restriction.allowed_movements_start <= self.current_movement < 
            restriction.allowed_movements_end):
            return True
        
        return False
    
    def block_agent(self, agent_id: str, duration: int, reason: str = "Unknown"):
        """Block an agent from trading for a specified duration"""
        if agent_id in self.agent_restrictions:
            self.agent_restrictions[agent_id].is_blocked = True
            self.agent_restrictions[agent_id].block_remaining = duration
            self.agent_restrictions[agent_id].block_reason = reason
        else:
            self.agent_restrictions[agent_id] = AgentTimeRestriction(
                agent_id=agent_id,
                allowed_movements_start=0,
                allowed_movements_end=self.movements_per_day,
                is_blocked=True,
                block_remaining=duration,
                block_reason=reason
            )
    
    def update_agent_blocks(self):
        """Update block durations for all agents"""
        for agent_id, restriction in self.agent_restrictions.items():
            if restriction.is_blocked and restriction.block_remaining > 0:
                restriction.block_remaining -= 1
                if restriction.block_remaining <= 0:
                    restriction.is_blocked = False
                    restriction.block_reason = None
    
    def trigger_random_block_event(self, agent_ids: List[str]) -> Optional[str]:
        """
        Trigger a random agent blocking event.
        Returns the agent_id if blocked, None otherwise.
        """
        import random
        
        if random.random() < self.block_probability and agent_ids:
            blocked_agent = random.choice(agent_ids)
            # Block for random duration (5-20 movements)
            duration = random.randint(5, 20)
            self.block_agent(blocked_agent, duration, "Random blocking event")
            return blocked_agent
        
        return None
    
    def trigger_random_kill_event(self, agent_ids: List[str]) -> Optional[str]:
        """
        Trigger a random agent kill event (very rare).
        Returns the agent_id if killed, None otherwise.
        Note: This is a placeholder - actual killing logic depends on agent lifecycle.
        """
        import random
        
        if random.random() < self.kill_probability and agent_ids:
            killed_agent = random.choice(agent_ids)
            # Block for very long duration (simulating death)
            duration = self.movements_per_day * 10  # 10 days
            self.block_agent(killed_agent, duration, "Agent killed (rare event)")
            return killed_agent
        
        return None
    
    def get_agent_status(self, agent_id: str) -> Dict:
        """Get current status of an agent"""
        restriction = self.agent_restrictions.get(agent_id)
        
        if restriction is None:
            return {
                'agent_id': agent_id,
                'can_trade': True,
                'is_blocked': False,
                'block_remaining': 0,
                'time_window': None
            }
        
        return {
            'agent_id': agent_id,
            'can_trade': self.can_agent_trade(agent_id),
            'is_blocked': restriction.is_blocked,
            'block_remaining': restriction.block_remaining,
            'block_reason': restriction.block_reason,
            'time_window': (restriction.allowed_movements_start, 
                          restriction.allowed_movements_end)
        }
    
    def initialize_default_restrictions(self, agent_ids: List[str]):
        """
        Initialize default time restrictions for agents.
        Distribute agents across different time windows to add realism.
        """
        import random
        
        n_agents = len(agent_ids)
        
        # Strategy 1: Early traders (first 30% of day)
        early_count = max(1, n_agents // 3)
        for i in range(early_count):
            if i < len(agent_ids):
                end = random.randint(3, 5)
                self.add_agent_restriction(agent_ids[i], 0, end)
        
        # Strategy 2: Mid-day traders (30-70% of day)
        mid_count = max(1, n_agents // 3)
        for i in range(early_count, early_count + mid_count):
            if i < len(agent_ids):
                start = random.randint(3, 5)
                end = random.randint(7, 9)
                self.add_agent_restriction(agent_ids[i], start, end)
        
        # Strategy 3: Late traders (last 30% of day)
        for i in range(early_count + mid_count, n_agents):
            if i < len(agent_ids):
                start = random.randint(7, 9)
                self.add_agent_restriction(agent_ids[i], start, self.movements_per_day)
    
    def reset(self):
        """Reset time manager state"""
        self.current_day = 0
        self.current_movement = 0
        self.total_steps = 0
        self.simulation_start_time = None
        self.day_start_time = None
        self.is_black_box = False
        self.black_box_counter = 0
        self.black_box_remaining = 0
        self.agent_restrictions.clear()
    
    def get_state(self) -> Dict:
        """Get current time manager state"""
        return {
            'current_day': self.current_day,
            'current_movement': self.current_movement,
            'total_steps': self.total_steps,
            'day_progress': self.get_day_progress(),
            'is_black_box': self.is_black_box,
            'black_box_remaining': self.black_box_remaining,
            'time_until_day_end': self.get_time_until_day_end(),
            'active_restrictions': len(self.agent_restrictions)
        }
