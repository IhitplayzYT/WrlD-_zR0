from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import random

class AgentType(Enum):
    BASE = "base"
    RLAIF = "rlaif"
    ADVERSARIAL = "adversarial"

@dataclass
class AgentTraits:

    risk_tolerance: float = 0.5 
    loss_aversion: float = 0.5  
    max_position_size: float = 0.3  # Max % of portfolio in single position
    
    patience: float = 0.5        # impulsive to patient
    holding_period_preference: float = 0.5  # Short vs long term preference
    reaction_speed: float = 0.5 
    
    trend_following: float = 0.5
    mean_reversion: float = 0.5   # Tendency to bet on reversions
    momentum_sensitivity: float = 0.5
    
    information_weight: float = 0.5  # Weight given to news/events
    technical_analysis_weight: float = 0.5  # Weight given to technical indicators
    fundamental_analysis_weight: float = 0.5  # Weight given to fundamentals
    
    herd_behavior: float = 0.5  
    contrarian_bias: float = 0.5 
    confidence_level: float = 0.5 
    
    adaptability: float = 0.5 
    exploration_rate: float = 0.5 
    memory_importance: float = 0.5
    
    order_size_preference: float = 0.5
    limit_order_preference: float = 0.5
    timing_precision: float = 0.5  # Precision in timing entries/exits
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2):
        for trait_name, trait_value in self.__dict__.items():
            if isinstance(trait_value, float) and random.random() < mutation_rate:
                mutation = random.uniform(-mutation_strength, mutation_strength)
                new_value = max(0.0, min(1.0, trait_value + mutation))
                setattr(self, trait_name, new_value)
    
    def crossover(self, other: 'AgentTraits') -> 'AgentTraits':
        new_traits = AgentTraits()
        for trait_name in self.__dict__.keys():
            if isinstance(getattr(self, trait_name), float):
                a,b = getattr(self,trait_name),getattr(other,trait_name)
                a,b = (b,a) if a > b else (a,b)
                setattr(new_traits, trait_name,random.uniform(a,b))
        return new_traits
    
    def get_behavior_modifiers(self) -> Dict[str, float]:
        modifiers = {}

        # Risk modifier based on risk tolerance and loss aversion
        modifiers['risk_modifier'] = self.risk_tolerance * 0.7 + (1 - self.loss_aversion) * 0.3
        # Patience modifier
        modifiers['patience_modifier'] = self.patience
        # Trend following modifier
        modifiers['trend_modifier'] = self.trend_following * 0.6 + (1 - self.contrarian_bias) * 0.4
        # Information processing modifier
        total_info_weight = self.information_weight + self.technical_analysis_weight + self.fundamental_analysis_weight
        modifiers['info_processing_modifier'] = (total_info_weight / 3.0)  if total_info_weight > 0 else 0.5
        # Social influence modifier
        modifiers['social_modifier'] = self.herd_behavior * 0.7 + (1 - self.confidence_level) * 0.3
        # Learning modifier
        modifiers['learning_modifier'] = self.adaptability * 0.5 + self.exploration_rate * 0.3 + self.memory_importance * 0.2
        # Trading style modifier
        modifiers['trading_style_modifier'] = self.limit_order_preference * 0.6 + self.timing_precision * 0.4
        return modifiers

@dataclass
class AgentState:
    agent_id: str
    agent_type: AgentType
    cash: float
    holdings: Dict[str, int] 
    total_value: float
    is_bad: bool = False
    villain_points: int = 0
    is_blocked: bool = False
    block_remaining: int = 0
    last_action: Optional[Dict] = None
    profit_history: List[float] = field(default_factory=list)
    traits: Optional[AgentTraits] = None
    mem_capacity: int = 100
    episode_performance: Dict[str, float] = field(default_factory=dict)
    memory: List[Dict] = field(default_factory=list)  # Agent's memory of past events
    age: int = 0
    peak: float = 0.0  # Track peak portfolio value for drawdown calculation
    max_drawdown: float = 0.0
    
    def update_portfolio_value(self, current_prices: Dict[str, float], initial_capital: float):
        holdings_value = sum(quantity * current_prices.get(asset, 0.0) for asset, quantity in self.holdings.items())
        self.total_value = self.cash + holdings_value
        self.peak = max(self.peak, self.total_value)
        self.max_drawdown = max(self.max_drawdown, 1-self.total_value/max(self.peak, 1e-8))
        self.profit_history.append(self.total_value - initial_capital)
    
    def add_memory(self, memory: Dict):
        self.memory.append(memory)
        if len(self.memory) > self.mem_capacity: 
            self.memory.pop(0)
    
    def get_recent_performance(self, window: int = 10) -> float:
        if len(self.profit_history) < window:
            window = len(self.profit_history)
        return 0.0 if window == 0 else sum(self.profit_history[-window:]) / window

class BaseAgent:
    
    def __init__(self,agent_id: str,agent_type: AgentType = AgentType.BASE,initial_cash: float = 1000.0,traits: Optional[AgentTraits] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.initial_cash = initial_cash
        self.traits = AgentTraits() if  traits is None else traits
        self.state = AgentState(agent_id=agent_id,agent_type=agent_type,cash=initial_cash,holdings={},total_value=initial_cash,traits=self.traits,peak=initial_cash)
            
    # Arbitrary Random Guess
    def decide_action(self,observation: Dict[str, Any],action_space: Any) -> np.ndarray:
        """ [action_type, asset_idx, quantity, price]
        action_type: 0=hold, 1=market_buy, 2=market_sell, 3=limit_buy, 4=limit_sell """
        assets = np.flatnonzero(np.asarray(observation['available_assets']) > 0)
        idx = int(random.choice(assets))
        price = float(observation['current_price'][idx])
        cash = float(observation.get('available_cash', observation['agent_cash']))
        holdings = observation.get('available_holdings', observation['agent_holdings'])
        kinds = [0]
        if cash >= price * 1.002:
            kinds.extend([1, 3])
        if holdings[idx] > 0:
            kinds.extend([2, 4])
        kind = random.choice(kinds) if observation.get('can_trade', True) else 0
        qty = min(10, int(cash/(price*1.002))) if kind in (1, 3) else min(10, int(holdings[idx]))
        quote = price * (.999 if kind == 3 else 1.001 if kind == 4 else 1.)
        return np.array([kind, idx, qty if kind else 0, quote], dtype=np.float32)

    def apply_trait_modifications(self, base_action: np.ndarray) -> np.ndarray:
        modifiers = self.traits.get_behavior_modifiers()
        action = base_action.copy()
        action_type,asset_idx,quantity,price = int(action[0]),int(action[1]),int(action[2]),action[3]
        
        if action_type in [1, 3]:  # Buy actions
            quantity = int(quantity * modifiers['risk_modifier'])
        elif action_type in [2, 4]: 
            quantity = int(quantity * (1.0 - modifiers['risk_modifier'] * 0.5))

        # Convert market to limit 
        if modifiers['patience_modifier'] > 0.7 and action_type == 1:
            action[0] = 3 
        elif modifiers['patience_modifier'] > 0.7 and action_type == 2:
            action[0] = 4 
        
        # Trading style modifier affects limit order placement
        if action_type in [3, 4]:
            price_adjustment = (modifiers['trading_style_modifier'] - 0.5) * 0.1
            price = price * (1.0 + price_adjustment)
        action[2] = max(1, quantity)  
        action[3] = max(0.1, price)   
        return action
    
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        modifiers = self.traits.get_behavior_modifiers()
        processed_obs = observation.copy()
        if 'price_history' in processed_obs and (modifiers['info_processing_modifier'] < 0.5):
                processed_obs['price_history'] = processed_obs['price_history'] * 0.8
        
        if 'active_events' in processed_obs:
            processed_obs['active_events'] = (processed_obs['active_events'] * modifiers['info_processing_modifier'])
        return processed_obs
    
    def update_state(self, reward: float, done: bool, info: Dict):
        memory_entry = {'reward': reward,'action': self.state.last_action,'info': info,'timestamp': len(self.state.profit_history)}
        self.state.add_memory(memory_entry)
        
        self.state.episode_performance['total_reward'] = self.state.episode_performance.get('total_reward', 0.0) + reward
        self.state.episode_performance['num_actions'] = self.state.episode_performance.get('num_actions', 0) + 1
        
        # Agent roles are fixed experimental treatments, not a consequence of losses.
    
    def transform_to_bad(self):
        self.state.is_bad = True
        self.state.villain_points = 10
        # Modify traits to reflect adversarial nature
        self.traits.risk_tolerance = min(1.0, self.traits.risk_tolerance + 0.3)
        self.traits.contrarian_bias = min(1.0, self.traits.contrarian_bias + 0.4)
        self.traits.herd_behavior = max(0.0, self.traits.herd_behavior - 0.3)
    
    def reset(self, initial_cash: Optional[float] = None):
        cash = initial_cash if initial_cash is not None else self.initial_cash
        self.state = AgentState(agent_id=self.agent_id,agent_type=self.agent_type,cash=cash,holdings={},total_value=cash,traits=self.traits,age=self.state.age + 1,peak=cash)
    
    def get_info(self) -> Dict[str, Any]:
        return {'initial_cash': self.initial_cash, 'agent_id': self.agent_id,'agent_type': self.agent_type.value,'cash': self.state.cash,'holdings': self.state.holdings,'total_value': self.state.total_value,'is_bad': self.state.is_bad,'villain_points': self.state.villain_points,'traits': self.traits.__dict__,'age': self.state.age,'recent_performance': self.state.get_recent_performance(10)}
