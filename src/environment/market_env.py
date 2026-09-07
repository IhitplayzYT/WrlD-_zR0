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
from src.market.multi_asset_order_book import MultiAssetOrderBook
from src.events import EventManager, MarketEvent, EventType
from src.agents import BaseAgent, RLAIFAgent, AdversarialAgent, AgentType
from src.environment.observation_space import EnhancedObservationSpace
from src.simulation.black_box import BlackBoxPredictionSystem
from src.utils.logging import get_logger, LogCategory

logger = get_logger()


@dataclass
class AgentState:
    agent_id: str
    cash: float
    holdings: Dict[str, int]
    total_value: float
    is_bad: bool = False
    villain_points: int = 0
    is_blocked: bool = False
    block_remaining: int = 0
    last_action: Optional[Dict] = None
    profit_history: List[float] = field(default_factory=list)


class MarketEnvironment(AECEnv):
    metadata = {"name": "WrlD_Zr0","render_modes": ["human", "rgb_array"]}
    
    def __init__(self,n_agents: int = 5,max_bad_agents: int = 1,initial_cash: float = 1000.0,episode_length: int = 200,price_data: Dict[str, List[float]] = None, black_box_duration: int = 5,black_box_frequency: int = 20,event_manager: Optional[EventManager] = None,use_enhanced_observation: bool = True,agent_types: Optional[List[str]] = None):
        super().__init__()
        self.n_agents = n_agents
        self.max_bad_agents = max_bad_agents
        self.initial_cash = initial_cash
        self.episode_length = episode_length
        self.black_box_duration = black_box_duration
        self.black_box_frequency = black_box_frequency
        self.use_enhanced_observation = use_enhanced_observation
        if price_data is None:
            self.price_data = {"STOCK0": self._generate_synthetic_prices(episode_length * 10)}
        else:
            self.price_data = price_data
        
        self.available_assets = list(self.price_data.keys())
        self.order_book = MultiAssetOrderBook(assets=self.available_assets, tick_size=0.01)
        self.event_manager = event_manager if event_manager else EventManager()
        
        # Enhanced observation space
        if use_enhanced_observation:
            self.observation_space_enhanced = EnhancedObservationSpace(use_temporal_features=True,use_belief_states=True)
        else:
            self.observation_space_enhanced = None
        
        self.black_box_system = BlackBoxPredictionSystem(duration=black_box_duration,frequency=black_box_frequency,enable_prediction_market=True)
        
        self.agent_states: Dict[str, AgentState] = {}
        self.agent_objects: Dict[str, BaseAgent] = {}
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        
        if agent_types is None:
            agent_types = ["rlaif"] * (n_agents) 
        self.agent_types = agent_types
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.current_step = 0
        self.current_day = 0
        self.current_movement = 0
        self.movements_per_day = 10
        self.is_black_box = False
        self.black_box_counter = 0
        self.black_box_remaining = 0
        
        self.current_prices = {asset: prices[0] if prices else 100.0 for asset, prices in self.price_data.items()}
        self.price_history = {asset: [self.current_prices[asset]] for asset in self.available_assets}
        self.market_cap = {asset: price * 10000 for asset, price in self.current_prices.items()}
        self.economy_collapsed = False
        self.economy_growth_threshold = 0.10 
        
        self._initialize_agents()
        self._init_spaces()
        self.render_mode = None
        
    def _generate_synthetic_prices(self, n_steps: int) -> List[float]:
        prices = [100.0]
        for _ in range(n_steps - 1):
            change = np.random.normal(0, 0.01)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))
        return prices
    
    def _initialize_agents(self):
        for i, agent_id in enumerate(self.agents):
            # Determine agent type
            agent_type_str = self.agent_types[i] if i < len(self.agent_types) else "rlaif"
            
            if agent_type_str == "rlaif":
                agent = RLAIFAgent(agent_id=agent_id,initial_cash=self.initial_cash)
            elif agent_type_str == "adversarial":
                agent = AdversarialAgent(agent_id=agent_id,initial_cash=self.initial_cash)
            else:
                agent = BaseAgent(agent_id=agent_id,initial_cash=self.initial_cash)

            self.agent_objects[agent_id] = agent
            self.agent_states[agent_id] = AgentState(agent_id,self.initial_cash,{},self.initial_cash,(agent_type_str == "adversarial"),agent.state.villain_points if hasattr(agent.state, 'villain_points') else 0)
    
    def _init_spaces(self):
        """Initialize action and observation spaces"""
        # Action space: [action_type, asset_idx, quantity, price]
        # action_type: 0=hold, 1=market_buy, 2=market_sell, 3=limit_buy, 4=limit_sell
        # asset_idx: Index of asset to trade
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([4, 10, 1000, 10000]),  # Max 10 assets
            dtype=np.float32
        )
        
        # Observation space: price history, order book state, agent state, events
        # Now includes multi-asset holdings, prices, and order books
        self.observation_space = spaces.Dict({
            'price_history': spaces.Box(low=0, high=10000, shape=(50, 10), dtype=np.float32),  # 50 steps x 10 assets
            'order_book_bids': spaces.Box(low=0, high=10000, shape=(10, 10, 2), dtype=np.float32),  # 10 assets x 10 levels x 2
            'order_book_asks': spaces.Box(low=0, high=10000, shape=(10, 10, 2), dtype=np.float32),  # 10 assets x 10 levels x 2
            'agent_cash': spaces.Box(low=0, high=100000, shape=(), dtype=np.float32),
            'agent_holdings': spaces.Box(low=0, high=10000, shape=(10,), dtype=np.int32), 
            'agent_total_value': spaces.Box(low=0, high=100000, shape=(), dtype=np.float32),
            'is_black_box': spaces.Discrete(2),
            'current_price': spaces.Box(low=0, high=10000, shape=(10,), dtype=np.float32), 
            'market_cap': spaces.Box(low=0, high=1e9, shape=(10,), dtype=np.float32), 
            'active_events': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            'available_assets': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            np.random.seed(seed)
        
        self.order_book.reset()
        self._initialize_agents()
        self.current_step = 0
        self.current_day = 0
        self.current_movement = 0
        self.is_black_box = False
        self.black_box_counter = 0
        self.black_box_remaining = 0
        self.current_prices = {asset: prices[0] if prices else 100.0 for asset, prices in self.price_data.items()}
        self.price_history = {asset: [self.current_prices[asset]] for asset in self.available_assets}
        self.market_cap = {asset: price * 10000 for asset, price in self.current_prices.items()}
        self.economy_collapsed = False
        self.event_manager.reset()
        if self.observation_space_enhanced:
            self.observation_space_enhanced.reset()
        self.black_box_system.reset()
        for agent_id, agent in self.agent_objects.items():
            agent.reset(self.initial_cash)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    
    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict, float, bool, bool, Dict]:
        agent = self.agent_selection

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
            observation = self._observe(agent)
            action = agent_obj.decide_action(observation, self.action_space)
        
        self._execute_action(agent, action)
        self._update_market()
        self._process_events()
        if self.black_box_system.should_trigger():
            self.is_black_box = True
            logger.info(LogCategory.EVENT, f"Black box period triggered at step {self.current_step}", data={'step': self.current_step})
        
        if self.is_black_box:
            self.black_box_system.update()
            if not self.black_box_system.is_active:
                self.is_black_box = False
                if hasattr(self, '_pending_predictions') and self._pending_predictions:
                    actual_values = self.price_data[-self.black_box_duration:]
                    results = self.black_box_system.evaluate_predictions(actual_values)
                    self._pending_predictions = None
        
        self._check_agent_transformations()
        agent_obj.state.cash = self.agent_states[agent].cash
        agent_obj.state.holdings = self.agent_states[agent].holdings
        agent_obj.state.total_value = self.agent_states[agent].total_value
        agent_obj.state.is_bad = self.agent_states[agent].is_bad
        agent_obj.state.villain_points = self.agent_states[agent].villain_points
        reward = self._calculate_reward(agent, action)
        agent_obj.update_state(reward, False, {})
        terminated, truncated = self._check_termination()
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
        action_type = int(action[0])
        asset_idx = int(action[1])
        quantity = int(action[2])
        price = action[3]
        asset_name = self.available_assets[asset_idx] if asset_idx < len(self.available_assets) else self.available_assets[0]
        agent_state = self.agent_states[agent_id]
        agent_state.last_action = {'type': action_type, 'asset': asset_name, 'quantity': quantity, 'price': price}

        if action_type == 0:  # Hold
            return

        current_asset_price = self.current_prices.get(asset_name, float('inf')) # To indicate ghost assets are unaffordable
        
        if action_type == 1:  # Buy
            
            best_ask = self.order_book.get_best_ask(asset_name)
            actual_price = best_ask[0] if best_ask else current_asset_price
            max_quantity = min(quantity, int(agent_state.cash / actual_price))
            if max_quantity > 0:
                agent_state.cash -= max_quantity * actual_price
                agent_state.holdings[asset_name] = agent_state.holdings.get(asset_name, 0) + max_quantity

                # Add market order to book for price impact
                self.order_book.add_order(agent_id, asset_name, OrderSide.BUY, actual_price, max_quantity)
        
        elif action_type == 2:  # Sell
            # Sell at best bid price or current price
            best_bid = self.order_book.get_best_bid(asset_name)
            actual_price = best_bid[0] if best_bid else current_asset_price
            current_holdings = agent_state.holdings.get(asset_name, 0)
            max_quantity = min(quantity, current_holdings)
            if max_quantity > 0:
                agent_state.cash += max_quantity * actual_price
                agent_state.holdings[asset_name] = current_holdings - max_quantity
                if agent_state.holdings[asset_name] == 0:
                    del agent_state.holdings[asset_name]

                # Add market order to book for price impact
                self.order_book.add_order(agent_id, asset_name, OrderSide.SELL, actual_price, max_quantity)
        
        elif action_type == 3:  # Limit buy
            if price > 0 and quantity > 0:
                cost = quantity * price
                if agent_state.cash >= cost:
                    self.order_book.add_order(agent_id, asset_name, OrderSide.BUY, price, quantity)
        
        elif action_type == 4:  # Limit sell
            if price > 0 and quantity > 0:
                current_holdings = agent_state.holdings.get(asset_name, 0)
                if current_holdings >= quantity:
                    self.order_book.add_order(agent_id, asset_name, OrderSide.SELL, price, quantity)
        
        # Update total value across all assets
        holdings_value = sum(qty * self.current_prices.get(asset, 0.0) for asset, qty in agent_state.holdings.items())
        agent_state.total_value = agent_state.cash + holdings_value
        agent_state.profit_history.append(agent_state.total_value - self.initial_cash)
    
    def _update_market(self):
        # Update current prices based on order book or price data for each asset
        for asset in self.available_assets:
            mid_price = self.order_book.get_mid_price(asset)
            if mid_price is not None:
                self.current_prices[asset] = mid_price
            else:
                # Use price data if no orders in book
                asset_prices = self.price_data.get(asset, [])
                price_idx = min(self.current_step, len(asset_prices) - 1)
                self.current_prices[asset] = asset_prices[price_idx] if asset_prices else 100.0
        
        for asset in self.available_assets:
            self.price_history[asset].append(self.current_prices[asset])
        for asset in self.available_assets:
            self.market_cap[asset] = self.current_prices[asset] * 10000
        if self.current_prices[self.available_assets[0]] < self.price_history[self.available_assets[0]][0] * 0.5:
            self.economy_collapsed = True
    
    def _process_events(self):
        active_events = self.event_manager.get_active_events()
        
        for event in active_events:
            # Apply event effects to market (apply to all assets for simplicity)
            if event.effects:
                for asset in self.available_assets:
                    self.current_prices[asset] *= (1 + event.effects.price_drift)
                    self.current_prices[asset] = max(self.current_prices[asset], 1.0)
    
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
        
        # Terminate if economy grows significantly (based on first asset)
        if len(self.price_history[self.available_assets[0]]) > 0:
            growth = (self.current_prices[self.available_assets[0]] - 
                     self.price_history[self.available_assets[0]][0]) / \
                    self.price_history[self.available_assets[0]][0]
            if growth > self.economy_growth_threshold:
                return True, False
        
        # Terminate if episode length reached
        if self.current_step >= self.episode_length:
            return True, False
        
        return False, False
    
    def _observe(self, agent_id: str) -> Dict:
        """Generate observation for an agent"""
        agent_state = self.agent_states[agent_id]
        
        # Multi-asset price history: (50, 10) - 50 steps x 10 assets
        price_hist_matrix = np.zeros((50, 10), dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            asset_hist = self.price_history[asset][-50:]
            if len(asset_hist) < 50:
                asset_hist = [0] * (50 - len(asset_hist)) + asset_hist
            price_hist_matrix[:, i] = np.array(asset_hist, dtype=np.float32)
        
        # Multi-asset order book state: (10, 10, 2) - 10 assets x 10 levels x 2 (price, quantity)
        bids_matrix = np.zeros((10, 10, 2), dtype=np.float32)
        asks_matrix = np.zeros((10, 10, 2), dtype=np.float32)
        
        for i, asset in enumerate(self.available_assets[:10]):
            order_book_state = self.order_book.get_order_book_state(asset)
            bids_list = order_book_state.get('bids', [])[:10]
            asks_list = order_book_state.get('asks', [])[:10]
            
            # Convert bids to array
            if len(bids_list) > 0:
                bids = np.array(bids_list, dtype=np.float32)
                if bids.ndim == 1:
                    bids = bids.reshape(-1, 1)
                    bids = np.hstack([bids, np.zeros((bids.shape[0], 1), dtype=np.float32)])
                for j, bid in enumerate(bids[:10]):
                    bids_matrix[i, j] = bid
            
            # Convert asks to array
            if len(asks_list) > 0:
                asks = np.array(asks_list, dtype=np.float32)
                if asks.ndim == 1:
                    asks = asks.reshape(-1, 1)
                    asks = np.hstack([asks, np.zeros((asks.shape[0], 1), dtype=np.float32)])
                for j, ask in enumerate(asks[:10]):
                    asks_matrix[i, j] = ask
        
        # Multi-asset holdings (pad to 10 assets)
        holdings_array = np.zeros(10, dtype=np.int32)
        for i, asset in enumerate(self.available_assets[:10]):
            holdings_array[i] = agent_state.holdings.get(asset, 0)
        
        # Multi-asset prices (pad to 10 assets)
        prices_array = np.zeros(10, dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            prices_array[i] = self.current_prices.get(asset, 100.0)
        
        # Multi-asset market caps (pad to 10 assets)
        market_cap_array = np.zeros(10, dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            market_cap_array[i] = self.market_cap.get(asset, 1000000.0)
        
        # Available assets one-hot encoding
        available_assets_array = np.zeros(10, dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            available_assets_array[i] = 1.0
        
        # Active events (one-hot encoded)
        active_events = self.event_manager.get_active_events()
        event_vector = np.zeros(10, dtype=np.float32)
        for i, event in enumerate(active_events[:10]):
            event_vector[i] = 1.0
        
        base_observation = {
            'price_history': price_hist_matrix,
            'order_book_bids': bids_matrix,
            'order_book_asks': asks_matrix,
            'agent_cash': np.array(agent_state.cash, dtype=np.float32),
            'agent_holdings': holdings_array,
            'agent_total_value': np.array(agent_state.total_value, dtype=np.float32),
            'is_black_box': 1 if self.is_black_box else 0,
            'current_price': prices_array,
            'market_cap': market_cap_array,
            'active_events': event_vector,
            'available_assets': available_assets_array,
        }
        
        # Apply enhanced observation space if enabled (using primary asset for simplicity)
        if self.use_enhanced_observation and self.observation_space_enhanced:
            primary_asset = self.available_assets[0]
            agent_state_dict = {
                'cash': agent_state.cash,
                'holdings': agent_state.holdings,
                'total_value': agent_state.total_value
            }
            
            primary_order_book_state = self.order_book.get_order_book_state(primary_asset)
            best_bid = primary_order_book_state.get('best_bid')
            best_ask = primary_order_book_state.get('best_ask')
            
            market_state_dict = {
                'current_price': self.current_prices[primary_asset],
                'best_bid': best_bid[0] if best_bid else self.current_prices[primary_asset] * 0.99,
                'best_ask': best_ask[0] if best_ask else self.current_prices[primary_asset] * 1.01,
                'spread': primary_order_book_state.get('spread', 0.02)
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
            print(f"Available Assets: {self.available_assets}")
            print(f"Current Prices: {self.current_prices}")
            print(f"Market Caps: {self.market_cap}")
            print(f"Black Box: {self.is_black_box}")
            print("\nAgent States:")
            for agent_id, state in self.agent_states.items():
                print(f"  {agent_id}: Cash={state.cash:.2f}, Holdings={state.holdings}, "
                      f"Total={state.total_value:.2f}, Bad={state.is_bad}")
            print("\nOrder Books:")
            for asset in self.available_assets:
                ob_state = self.order_book.get_order_book_state(asset)
                print(f"  {asset}: Bid={ob_state.get('best_bid')}, Ask={ob_state.get('best_ask')}, Spread={ob_state.get('spread')}")
            print("-" * 50)
    
    def close(self):
        """Clean up the environment"""
        pass
