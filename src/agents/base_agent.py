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
    """
    Comprehensive trait system for agent behavior modification.
    Traits influence decision-making, risk tolerance, and trading patterns.
    """
    # Risk-related traits
    risk_tolerance: float = 0.5  # 0.0 (risk-averse) to 1.0 (risk-seeking)
    loss_aversion: float = 0.5   # Higher values = more sensitive to losses
    max_position_size: float = 0.3  # Max % of portfolio in single position
    
    # Time-related traits
    patience: float = 0.5        # 0.0 (impulsive) to 1.0 (patient)
    holding_period_preference: float = 0.5  # Short-term vs long-term preference
    reaction_speed: float = 0.5   # How quickly agent reacts to market changes
    
    # Strategy-related traits
    trend_following: float = 0.5  # 0.0 (contrarian) to 1.0 (trend follower)
    mean_reversion: float = 0.5   # Tendency to bet on reversions
    momentum_sensitivity: float = 0.5  # Sensitivity to momentum signals
    
    # Information-related traits
    information_weight: float = 0.5  # How much weight given to news/events
    technical_analysis_weight: float = 0.5  # Weight given to technical indicators
    fundamental_analysis_weight: float = 0.5  # Weight given to fundamentals
    
    # Social-related traits
    herd_behavior: float = 0.5    # Tendency to follow other agents
    contrarian_bias: float = 0.5  # Tendency to go against the crowd
    confidence_level: float = 0.5  # Confidence in own decisions
    
    # Learning-related traits
    adaptability: float = 0.5    # How quickly agent adapts to new conditions
    exploration_rate: float = 0.5  # Rate of trying new strategies
    memory_importance: float = 0.5  # Importance of historical performance
    
    # Trading-related traits
    order_size_preference: float = 0.5  # Preference for large vs small orders
    limit_order_preference: float = 0.5  # Preference for limit vs market orders
    timing_precision: float = 0.5  # Precision in timing entries/exits
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2):
        """
        Evolve traits through mutation for agent reproduction.
        """
        for trait_name, trait_value in self.__dict__.items():
            if isinstance(trait_value, float) and random.random() < mutation_rate:
                mutation = random.uniform(-mutation_strength, mutation_strength)
                new_value = max(0.0, min(1.0, trait_value + mutation))
                setattr(self, trait_name, new_value)
    
    def crossover(self, other: 'AgentTraits') -> 'AgentTraits':
        """
        Create new traits through crossover with another agent's traits.
        """
        new_traits = AgentTraits()
        for trait_name in self.__dict__.keys():
            if isinstance(getattr(self, trait_name), float):
                parent_choice = random.choice([self, other])
                setattr(new_traits, trait_name, getattr(parent_choice, trait_name))
        return new_traits
    
    def get_behavior_modifiers(self) -> Dict[str, float]:
        """
        Get behavior modifiers based on trait combinations.
        Returns dictionary of modifier names and their values.
        """
        modifiers = {}
        
        # Risk modifier based on risk tolerance and loss aversion
        modifiers['risk_modifier'] = (self.risk_tolerance * 0.7 + 
                                     (1 - self.loss_aversion) * 0.3)
        
        # Patience modifier
        modifiers['patience_modifier'] = self.patience
        
        # Trend following modifier
        modifiers['trend_modifier'] = (self.trend_following * 0.6 + 
                                      (1 - self.contrarian_bias) * 0.4)
        
        # Information processing modifier
        total_info_weight = (self.information_weight + 
                           self.technical_analysis_weight + 
                           self.fundamental_analysis_weight)
        if total_info_weight > 0:
            modifiers['info_processing_modifier'] = total_info_weight / 3.0
        else:
            modifiers['info_processing_modifier'] = 0.5
        
        # Social influence modifier
        modifiers['social_modifier'] = (self.herd_behavior * 0.7 + 
                                      (1 - self.confidence_level) * 0.3)
        
        # Learning modifier
        modifiers['learning_modifier'] = (self.adaptability * 0.5 + 
                                         self.exploration_rate * 0.3 +
                                         self.memory_importance * 0.2)
        
        # Trading style modifier
        modifiers['trading_style_modifier'] = (self.limit_order_preference * 0.6 +
                                              self.timing_precision * 0.4)
        
        return modifiers


@dataclass
class AgentState:
    """
    Current state of an agent in the simulation.
    """
    agent_id: str
    agent_type: AgentType
    cash: float
    holdings: int
    total_value: float
    is_bad: bool = False
    villain_points: int = 0
    is_blocked: bool = False
    block_remaining: int = 0
    last_action: Optional[Dict] = None
    profit_history: List[float] = field(default_factory=list)
    traits: Optional[AgentTraits] = None
    episode_performance: Dict[str, float] = field(default_factory=dict)
    memory: List[Dict] = field(default_factory=list)  # Agent's memory of past events
    age: int = 0  # Number of episodes agent has participated in
    
    def update_portfolio_value(self, current_price: float):
        """Update total portfolio value based on current price"""
        self.total_value = self.cash + (self.holdings * current_price)
        profit = self.total_value - 1000.0  # Assuming initial capital of 1000
        self.profit_history.append(profit)
    
    def add_memory(self, memory_entry: Dict):
        """Add a memory entry (limited to recent memories)"""
        self.memory.append(memory_entry)
        if len(self.memory) > 100:  # Keep last 100 memories
            self.memory.pop(0)
    
    def get_recent_performance(self, window: int = 10) -> float:
        """Get average performance over recent window"""
        if len(self.profit_history) < window:
            window = len(self.profit_history)
        if window == 0:
            return 0.0
        return sum(self.profit_history[-window:]) / window


class BaseAgent:
    """
    Base agent class with comprehensive trait system and decision-making interface.
    All specific agent types (RLAIF, Adversarial) inherit from this base class.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType = AgentType.BASE,
        initial_cash: float = 1000.0,
        traits: Optional[AgentTraits] = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.initial_cash = initial_cash
        
        # Initialize traits
        if traits is None:
            self.traits = AgentTraits()
        else:
            self.traits = traits
        
        # Initialize state
        self.state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            cash=initial_cash,
            holdings=0,
            total_value=initial_cash,
            traits=self.traits
        )
        
        # Agent lifecycle
        self.is_alive = True
        self.generation = 0  # Generation in evolutionary process
        
    def decide_action(
        self,
        observation: Dict[str, Any],
        action_space: Any
    ) -> np.ndarray:
        """
        Decide on an action based on current observation.
        This is the main decision-making interface that should be overridden by subclasses.
        
        Args:
            observation: Current observation from environment
            action_space: Available action space
            
        Returns:
            Action as numpy array [action_type, quantity, price]
        """
        # Default implementation: random action
        # Subclasses should override this with their specific logic
        action_type = random.randint(0, 4)  # 0=hold, 1=market_buy, 2=market_sell, 3=limit_buy, 4=limit_sell
        quantity = random.randint(1, 100)
        price = random.uniform(90.0, 110.0)
        
        return np.array([action_type, quantity, price], dtype=np.float32)
    
    def apply_trait_modifications(self, base_action: np.ndarray) -> np.ndarray:
        """
        Apply trait-based modifications to base action.
        This allows traits to influence the final decision.
        """
        modifiers = self.traits.get_behavior_modifiers()
        action = base_action.copy()
        
        # Modify action based on traits
        action_type = int(action[0])
        quantity = int(action[1])
        price = action[2]
        
        # Risk modifier affects position sizing
        if action_type in [1, 3]:  # Buy actions
            quantity = int(quantity * modifiers['risk_modifier'])
        elif action_type in [2, 4]:  # Sell actions
            quantity = int(quantity * (1.0 - modifiers['risk_modifier'] * 0.5))
        
        # Patience modifier affects order type preference
        if modifiers['patience_modifier'] > 0.7 and action_type == 1:
            action[0] = 3  # Convert market buy to limit buy
        elif modifiers['patience_modifier'] > 0.7 and action_type == 2:
            action[0] = 4  # Convert market sell to limit sell
        
        # Trading style modifier affects limit order placement
        if action_type in [3, 4]:
            price_adjustment = (modifiers['trading_style_modifier'] - 0.5) * 0.1
            price = price * (1.0 + price_adjustment)
        
        action[1] = max(1, quantity)  # Ensure minimum quantity
        action[2] = max(0.1, price)    # Ensure positive price
        
        return action
    
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and potentially transform observation based on agent traits.
        Agents may weight different observation features differently.
        """
        modifiers = self.traits.get_behavior_modifiers()
        processed_obs = observation.copy()
        
        # Apply information processing modifiers
        if 'price_history' in processed_obs:
            # Technical analysis weight affects how much price history is emphasized
            if modifiers['info_processing_modifier'] < 0.5:
                # Reduce emphasis on price history for agents that value other info
                processed_obs['price_history'] = processed_obs['price_history'] * 0.8
        
        if 'active_events' in processed_obs:
            # Information weight affects event importance
            processed_obs['active_events'] = (processed_obs['active_events'] * 
                                            modifiers['info_processing_modifier'])
        
        return processed_obs
    
    def update_state(self, reward: float, done: bool, info: Dict):
        """
        Update agent state after taking action and receiving reward.
        """
        # Add memory entry
        memory_entry = {
            'reward': reward,
            'action': self.state.last_action,
            'info': info,
            'timestamp': len(self.state.profit_history)
        }
        self.state.add_memory(memory_entry)
        
        # Update episode performance
        self.state.episode_performance['total_reward'] = \
            self.state.episode_performance.get('total_reward', 0.0) + reward
        self.state.episode_performance['num_actions'] = \
            self.state.episode_performance.get('num_actions', 0) + 1
        
        # Check for transformation conditions (good -> bad)
        if not self.state.is_bad and self.state.total_value < self.initial_cash * 0.5:
            self.transform_to_bad()
    
    def transform_to_bad(self):
        """Transform agent from good to bad (adversarial)"""
        self.state.is_bad = True
        self.state.villain_points = 10
        # Modify traits to reflect adversarial nature
        self.traits.risk_tolerance = min(1.0, self.traits.risk_tolerance + 0.3)
        self.traits.contrarian_bias = min(1.0, self.traits.contrarian_bias + 0.4)
        self.traits.herd_behavior = max(0.0, self.traits.herd_behavior - 0.3)
    
    def should_die(self) -> bool:
        """
        Check if agent should be removed from simulation.
        Based on performance and age.
        """
        # Agent dies if portfolio value is too low
        if self.state.total_value < self.initial_cash * 0.1:
            return True
        
        # Agent dies if too old and performing poorly
        if self.state.age > 50 and self.state.get_recent_performance(20) < -100:
            return True
        
        return False
    
    def reproduce(self, partner: Optional['BaseAgent'] = None) -> 'BaseAgent':
        """
        Create offspring through trait crossover and mutation.
        Used for evolutionary agent population dynamics.
        """
        # Determine parent traits
        if partner is None:
            parent_traits = self.traits
        else:
            parent_traits = self.traits.crossover(partner.traits)
        
        # Mutate traits
        parent_traits.mutate(mutation_rate=0.2, mutation_strength=0.15)
        
        # Create offspring
        offspring = BaseAgent(
            agent_id=f"{self.agent_id}_child_{random.randint(1000, 9999)}",
            agent_type=self.agent_type,
            initial_cash=self.initial_cash,
            traits=parent_traits
        )
        
        offspring.generation = self.generation + 1
        return offspring
    
    def reset(self, initial_cash: Optional[float] = None):
        """Reset agent state for new episode"""
        cash = initial_cash if initial_cash is not None else self.initial_cash
        self.state = AgentState(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            cash=cash,
            holdings=0,
            total_value=cash,
            traits=self.traits,
            age=self.state.age + 1
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information for logging/analysis"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'is_alive': self.is_alive,
            'generation': self.generation,
            'cash': self.state.cash,
            'holdings': self.state.holdings,
            'total_value': self.state.total_value,
            'is_bad': self.state.is_bad,
            'villain_points': self.state.villain_points,
            'traits': self.traits.__dict__,
            'age': self.state.age,
            'recent_performance': self.state.get_recent_performance(10)
        }
