import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market import OrderBook, OrderSide
from src.events import EventManager, MarketEvent, EventType
from src.agents import BaseAgent, RLAIFAgent, AdversarialAgent, AgentType
from src.environment.observation_space import EnhancedObservationSpace
from src.simulation.black_box import BlackBoxPredictionSystem


@dataclass
class AgentState:
    agent_id: str
    cash: float
    holdings: int
    total_value: float
    is_bad: bool = False
    villain_points: int = 0
    is_blocked: bool = False
    block_remaining: int = 0
    last_action: Optional[Dict] = None
    profit_history: List[float] = field(default_factory=list)


class MarketEnvironment(AECEnv):
    """
    Multi-agent market environment for RLAIF trading simulation.
    Uses PettingZoo AEC (Agent-Environment Cycle) API for multi-agent RL.
    """
    
    metadata = {
        "name": "RLAIF-Market-v0",
        "render_modes": ["human", "rgb_array"],
    }
    
    def __init__(
        self,
        n_agents: int = 5,
        max_bad_agents: int = 1,
        initial_cash: float = 1000.0,
        episode_length: int = 200,
        price_data: List[float] = None,
        black_box_duration: int = 5,
        black_box_frequency: int = 20,
        event_manager: Optional[EventManager] = None,
        use_enhanced_observation: bool = True,
        agent_types: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.max_bad_agents = max_bad_agents
        self.initial_cash = initial_cash
        self.episode_length = episode_length
        self.black_box_duration = black_box_duration
        self.black_box_frequency = black_box_frequency
        self.use_enhanced_observation = use_enhanced_observation
        
        # Price data
        if price_data is None:
            # Generate synthetic price data
            self.price_data = self._generate_synthetic_prices(episode_length * 10)
        else:
            self.price_data = price_data
        
        # Order book
        self.order_book = OrderBook(tick_size=0.01)
        
        # Event manager
        self.event_manager = event_manager if event_manager else EventManager()
        
        # Enhanced observation space
        if use_enhanced_observation:
            self.observation_space_enhanced = EnhancedObservationSpace(
                use_temporal_features=True,
                use_belief_states=True
            )
        else:
            self.observation_space_enhanced = None
        
        # Black box prediction system
        self.black_box_system = BlackBoxPredictionSystem(
            duration=black_box_duration,
            frequency=black_box_frequency,
            enable_prediction_market=True
        )
        
        # Agent states and agent objects
        self.agent_states: Dict[str, AgentState] = {}
        self.agent_objects: Dict[str, BaseAgent] = {}
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        
        # Determine agent types
        if agent_types is None:
            # Default: mix of RLAIF and some potential adversarial agents
            agent_types = ["rlaif"] * (n_agents - 1) + ["rlaif"]  # Start all as RLAIF
        self.agent_types = agent_types
        
        # Agent selector
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        # Simulation state
        self.current_step = 0
        self.current_day = 0
        self.current_movement = 0
        self.movements_per_day = 10
        self.is_black_box = False
        self.black_box_counter = 0
        self.black_box_remaining = 0
        
        # Market state
        self.current_price = self.price_data[0] if self.price_data else 100.0
        self.price_history = [self.current_price]
        self.market_cap = self.current_price * 10000  # Assume 10000 shares
        self.economy_collapsed = False
        self.economy_growth_threshold = 0.10  # 10% growth
        
        # Initialize agents
        self._initialize_agents()
        
        # Action and observation spaces
        self._init_spaces()
        
        # Rendering
        self.render_mode = None
        
    def _generate_synthetic_prices(self, n_steps: int) -> List[float]:
        """Generate simple synthetic price data"""
        prices = [100.0]
        for _ in range(n_steps - 1):
            change = np.random.normal(0, 0.01)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Minimum price of 1.0
        return prices
    
    def _initialize_agents(self):
        """Initialize agent states and agent objects"""
        for i, agent_id in enumerate(self.agents):
            # Determine agent type
            agent_type_str = self.agent_types[i] if i < len(self.agent_types) else "rlaif"
            
            # Create agent object based on type
            if agent_type_str == "rlaif":
                agent = RLAIFAgent(
                    agent_id=agent_id,
                    initial_cash=self.initial_cash
                )
            elif agent_type_str == "adversarial":
                agent = AdversarialAgent(
                    agent_id=agent_id,
                    initial_cash=self.initial_cash
                )
            else:
                agent = BaseAgent(
                    agent_id=agent_id,
                    initial_cash=self.initial_cash
                )
            
            self.agent_objects[agent_id] = agent
            
            # Create agent state (for compatibility with existing code)
            self.agent_states[agent_id] = AgentState(
                agent_id=agent_id,
                cash=self.initial_cash,
                holdings=0,
                total_value=self.initial_cash,
                is_bad=(agent_type_str == "adversarial"),
                villain_points=agent.state.villain_points if hasattr(agent.state, 'villain_points') else 0
            )
    
    def _init_spaces(self):
        """Initialize action and observation spaces"""
        # Action space: [action_type, quantity, price]
        # action_type: 0=hold, 1=market_buy, 2=market_sell, 3=limit_buy, 4=limit_sell
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([4, 1000, 10000]),
            dtype=np.float32
        )
        
        # Observation space: price history, order book state, agent state, events
        self.observation_space = spaces.Dict({
            'price_history': spaces.Box(
                low=0, high=10000, shape=(50,), dtype=np.float32
            ),
            'order_book_bids': spaces.Box(
                low=0, high=10000, shape=(10, 2), dtype=np.float32
            ),
            'order_book_asks': spaces.Box(
                low=0, high=10000, shape=(10, 2), dtype=np.float32
            ),
            'agent_cash': spaces.Box(low=0, high=100000, shape=(), dtype=np.float32),
            'agent_holdings': spaces.Box(low=0, high=10000, shape=(), dtype=np.int32),
            'agent_total_value': spaces.Box(low=0, high=100000, shape=(), dtype=np.float32),
            'is_black_box': spaces.Discrete(2),
            'current_price': spaces.Box(low=0, high=10000, shape=(), dtype=np.float32),
            'market_cap': spaces.Box(low=0, high=1e9, shape=(), dtype=np.float32),
            'active_events': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset order book
        self.order_book.reset()
        
        # Reset agents
        self._initialize_agents()
        
        # Reset simulation state
        self.current_step = 0
        self.current_day = 0
        self.current_movement = 0
        self.is_black_box = False
        self.black_box_counter = 0
        self.black_box_remaining = 0
        
        # Reset market state
        self.current_price = self.price_data[0] if self.price_data else 100.0
        self.price_history = [self.current_price]
        self.market_cap = self.current_price * 10000
        self.economy_collapsed = False
        
        # Reset event manager
        self.event_manager.reset()
        
        # Reset enhanced observation space
        if self.observation_space_enhanced:
            self.observation_space_enhanced.reset()
        
        # Reset black box system
        self.black_box_system.reset()
        
        # Reset agent objects
        for agent_id, agent in self.agent_objects.items():
            agent.reset(self.initial_cash)
        
        # Reset agent selector
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        # Generate initial observations
        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step for the current agent"""
        agent = self.agent_selection
        
        # Check if agent is blocked
        if self.agent_states[agent].is_blocked:
            self.agent_states[agent].block_remaining -= 1
            if self.agent_states[agent].block_remaining <= 0:
                self.agent_states[agent].is_blocked = False
            reward = self._calculate_reward(agent, None)
            self.agent_selection = self._agent_selector.next()
            return self._get_observations(), reward, False, False, {agent: {}}
        
        # Get agent object and let it decide action if not provided
        agent_obj = self.agent_objects[agent]
        if action is None:
            # Get observation for agent
            observation = self._observe(agent)
            # Let agent decide action
            action = agent_obj.decide_action(observation, self.action_space)
        
        # Execute action
        self._execute_action(agent, action)
        
        # Update market
        self._update_market()
        
        # Process events
        self._process_events()
        
        # Check black box system
        if self.black_box_system.should_trigger():
            self.is_black_box = True
            print(f"Black box period triggered at step {self.current_step}")
        
        if self.is_black_box:
            self.black_box_system.update()
            if not self.black_box_system.is_active:
                self.is_black_box = False
                # Evaluate predictions if black box period ended
                if hasattr(self, '_pending_predictions') and self._pending_predictions:
                    actual_values = self.price_data[-self.black_box_duration:]
                    results = self.black_box_system.evaluate_predictions(actual_values)
                    self._pending_predictions = None
        
        # Check for agent transformations (good -> bad)
        self._check_agent_transformations()
        
        # Update agent state
        agent_obj.state.cash = self.agent_states[agent].cash
        agent_obj.state.holdings = self.agent_states[agent].holdings
        agent_obj.state.total_value = self.agent_states[agent].total_value
        agent_obj.state.is_bad = self.agent_states[agent].is_bad
        agent_obj.state.villain_points = self.agent_states[agent].villain_points
        
        # Calculate reward
        reward = self._calculate_reward(agent, action)
        
        # Update agent with reward
        agent_obj.update_state(reward, False, {})
        
        # Check termination
        terminated, truncated = self._check_termination()
        
        # Move to next agent
        self.agent_selection = self._agent_selector.next()
        
        # Increment step if all agents have acted
        if self.agent_selection == self.agents[0]:
            self.current_step += 1
            self.current_movement += 1
            
            if self.current_movement >= self.movements_per_day:
                self.current_movement = 0
                self.current_day += 1
        
        return self._get_observations(), reward, terminated, truncated, {agent: {}}
    
    def _execute_action(self, agent_id: str, action: np.ndarray):
        """Execute an agent's action"""
        action_type = int(action[0])
        quantity = int(action[1])
        price = action[2]
        
        agent_state = self.agent_states[agent_id]
        agent_state.last_action = {'type': action_type, 'quantity': quantity, 'price': price}
        
        if action_type == 0:  # Hold
            return
        
        if action_type == 1:  # Market buy
            # Buy at best ask price
            best_ask = self.order_book.get_best_ask()
            if best_ask:
                actual_price = best_ask[0]
                max_quantity = min(quantity, int(agent_state.cash / actual_price))
                if max_quantity > 0:
                    cost = max_quantity * actual_price
                    agent_state.cash -= cost
                    agent_state.holdings += max_quantity
                    # Add market order to book for price impact
                    self.order_book.add_order(agent_id, OrderSide.BUY, actual_price, max_quantity)
        
        elif action_type == 2:  # Market sell
            # Sell at best bid price
            best_bid = self.order_book.get_best_bid()
            if best_bid:
                actual_price = best_bid[0]
                max_quantity = min(quantity, agent_state.holdings)
                if max_quantity > 0:
                    revenue = max_quantity * actual_price
                    agent_state.cash += revenue
                    agent_state.holdings -= max_quantity
                    # Add market order to book for price impact
                    self.order_book.add_order(agent_id, OrderSide.SELL, actual_price, max_quantity)
        
        elif action_type == 3:  # Limit buy
            if price > 0 and quantity > 0:
                cost = quantity * price
                if agent_state.cash >= cost:
                    self.order_book.add_order(agent_id, OrderSide.BUY, price, quantity)
        
        elif action_type == 4:  # Limit sell
            if price > 0 and quantity > 0 and agent_state.holdings >= quantity:
                self.order_book.add_order(agent_id, OrderSide.SELL, price, quantity)
        
        # Update total value
        agent_state.total_value = agent_state.cash + (agent_state.holdings * self.current_price)
        agent_state.profit_history.append(agent_state.total_value - self.initial_cash)
    
    def _update_market(self):
        """Update market state"""
        # Update current price based on order book or price data
        mid_price = self.order_book.get_mid_price()
        if mid_price is not None:
            self.current_price = mid_price
        else:
            # Use price data if no orders in book
            price_idx = min(self.current_step, len(self.price_data) - 1)
            self.current_price = self.price_data[price_idx]
        
        self.price_history.append(self.current_price)
        
        # Update market cap
        self.market_cap = self.current_price * 10000
        
        # Check for economy collapse
        if self.current_price < self.price_history[0] * 0.5:
            self.economy_collapsed = True
    
    def _process_events(self):
        """Process market events"""
        active_events = self.event_manager.get_active_events()
        
        for event in active_events:
            # Apply event effects to market
            if event.effects:
                self.current_price *= (1 + event.effects.price_drift)
                self.current_price = max(self.current_price, 1.0)
    
    def _check_agent_transformations(self):
        """Check if any agents should transform to bad agents"""
        current_bad = sum(1 for state in self.agent_states.values() if state.is_bad)
        
        if current_bad < self.max_bad_agents:
            for agent_id, state in self.agent_states.items():
                if not state.is_bad and state.total_value < self.initial_cash * 0.5:
                    # Agent lost more than 50% - transform to bad
                    state.is_bad = True
                    state.villain_points = 10
                    break
    
    def _calculate_reward(self, agent_id: str, action: Optional[np.ndarray]) -> float:
        """Calculate reward for an agent"""
        agent_state = self.agent_states[agent_id]
        
        # Base reward: profit/loss
        profit = agent_state.total_value - self.initial_cash
        reward = profit / self.initial_cash  # Normalize by initial cash
        
        # Penalty for not participating during black box
        if self.is_black_box and action is not None and action[0] == 0:
            reward -= 0.1  # Large penalty for holding
        
        # Bonus for bad agents if economy collapses
        if agent_state.is_bad and self.economy_collapsed:
            reward += 1.0
        
        # Penalty for economy being stagnant
        if len(self.price_history) > 10:
            recent_change = (self.price_history[-1] - self.price_history[-10]) / self.price_history[-10]
            if abs(recent_change) < 0.01:
                reward -= 0.05
        
        return reward
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate"""
        # Terminate if economy collapses
        if self.economy_collapsed:
            return True, False
        
        # Terminate if economy grows significantly
        if len(self.price_history) > 0:
            growth = (self.current_price - self.price_history[0]) / self.price_history[0]
            if growth > self.economy_growth_threshold:
                return True, False
        
        # Terminate if episode length reached
        if self.current_step >= self.episode_length:
            return True, False
        
        return False, False
    
    def _observe(self, agent_id: str) -> Dict:
        """Generate observation for an agent"""
        agent_state = self.agent_states[agent_id]
        
        # Base observation
        price_hist = self.price_history[-50:]
        if len(price_hist) < 50:
            price_hist = [0] * (50 - len(price_hist)) + price_hist
        price_hist = np.array(price_hist, dtype=np.float32)
        
        # Order book state
        order_book_state = self.order_book.get_order_book_state()
        bids_list = order_book_state['bids'][:10]
        asks_list = order_book_state['asks'][:10]
        
        # Convert to arrays with proper shape handling
        if len(bids_list) > 0:
            bids = np.array(bids_list, dtype=np.float32)
            if bids.ndim == 1:
                bids = bids.reshape(-1, 1)
                bids = np.hstack([bids, np.zeros((bids.shape[0], 1), dtype=np.float32)])
        else:
            bids = np.zeros((0, 2), dtype=np.float32)
        
        if len(asks_list) > 0:
            asks = np.array(asks_list, dtype=np.float32)
            if asks.ndim == 1:
                asks = asks.reshape(-1, 1)
                asks = np.hstack([asks, np.zeros((asks.shape[0], 1), dtype=np.float32)])
        else:
            asks = np.zeros((0, 2), dtype=np.float32)
        
        # Pad if necessary
        if bids.shape[0] < 10:
            bids = np.vstack([bids, np.zeros((10 - bids.shape[0], 2), dtype=np.float32)])
        if asks.shape[0] < 10:
            asks = np.vstack([asks, np.zeros((10 - asks.shape[0], 2), dtype=np.float32)])
        
        # Active events (one-hot encoded)
        active_events = self.event_manager.get_active_events()
        event_vector = np.zeros(10, dtype=np.float32)
        for i, event in enumerate(active_events[:10]):
            event_vector[i] = 1.0
        
        base_observation = {
            'price_history': price_hist,
            'order_book_bids': bids,
            'order_book_asks': asks,
            'agent_cash': np.array(agent_state.cash, dtype=np.float32),
            'agent_holdings': np.array(agent_state.holdings, dtype=np.int32),
            'agent_total_value': np.array(agent_state.total_value, dtype=np.float32),
            'is_black_box': 1 if self.is_black_box else 0,
            'current_price':  np.array(self.current_price, dtype=np.float32),
            'market_cap': np.array(self.market_cap, dtype=np.float32),
            'active_events': event_vector,
        }
        
        # Apply enhanced observation space if enabled
        if self.use_enhanced_observation and self.observation_space_enhanced:
            agent_state_dict = {
                'cash': agent_state.cash,
                'holdings': agent_state.holdings,
                'total_value': agent_state.total_value
            }
            
            best_bid = order_book_state.get('best_bid')
            best_ask = order_book_state.get('best_ask')
            
            market_state_dict = {
                'current_price': self.current_price,
                'best_bid': best_bid[0] if best_bid else self.current_price * 0.99,
                'best_ask': best_ask[0] if best_ask else self.current_price * 1.01,
                'spread': order_book_state.get('spread', 0.02)
            }
            
            enhanced_obs = self.observation_space_enhanced.create_observation(
                base_observation,
                agent_state_dict,
                market_state_dict
            )
            
            # Merge enhanced features into base observation
            base_observation.update(enhanced_obs)
        
        return base_observation
    
    def _get_observations(self) -> Dict:
        """Get observations for all agents"""
        return {agent: self._observe(agent) for agent in self.agents}
    
    def render(self, mode: str = "human"):
        """Render the environment"""
        if mode == "human":
            print(f"Step: {self.current_step}, Day: {self.current_day}")
            print(f"Current Price: {self.current_price:.2f}")
            print(f"Market Cap: {self.market_cap:.2f}")
            print(f"Black Box: {self.is_black_box}")
            print("\nAgent States:")
            for agent_id, state in self.agent_states.items():
                print(f"  {agent_id}: Cash={state.cash:.2f}, Holdings={state.holdings}, "
                      f"Total={state.total_value:.2f}, Bad={state.is_bad}")
            print("\nOrder Book:")
            ob_state = self.order_book.get_order_book_state()
            print(f"  Best Bid: {ob_state['best_bid']}")
            print(f"  Best Ask: {ob_state['best_ask']}")
            print(f"  Spread: {ob_state['spread']}")
            print("-" * 50)
    
    def close(self):
        """Clean up the environment"""
        pass
