from src.utils.logging import JSONLogger, LogCategory, get_logger
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from src.market import OrderSide, MultiAssetOrderBook, MultiStockEnvironment
from src.events import EventManager
from src.agents import BaseAgent, RLAIFAgent, AdversarialAgent
from src.agents.adversarial_agent import MAX_TRIGGERABLE_EVENTS
from src.environment.observation_space import EnhancedObservationSpace
from src.simulation.black_box import BlackBoxPredictionSystem

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
    metadata = {"name": "WrlD_Zr0", "render_modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        n_agents: int = 5,
        max_bad_agents: int = 1,
        initial_cash: float = 1000.0,
        episode_length: int = 200,
        price_data: Optional[Dict[str, List[float]]] = None,
        black_box_duration: int = 5,
        black_box_frequency: int = 20,
        event_manager: Optional[EventManager] = None,
        use_enhanced_observation: bool = True,
        agent_types: Optional[List[str]] = None,
        logger: Optional[JSONLogger] = None
    ):
        super().__init__()
        self.logger = logger or get_logger()
        self.n_agents = n_agents
        # Enforce constraint: max_bad_agents <= 0.2 * total_agents
        max_bad_allowed = max(1, int(0.2 * n_agents))
        self.max_bad_agents = min(max_bad_agents, max_bad_allowed)
        
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
        self.order_book = MultiAssetOrderBook(assets=self.available_assets)
        self.multi_stock_env = MultiStockEnvironment(assets=self.available_assets, stock_data=self.price_data)
        self.event_manager = event_manager if event_manager is not None else EventManager(load_default=True)
        
        if use_enhanced_observation:
            self.observation_space_enhanced = EnhancedObservationSpace(use_temporal_features=True, use_belief_states=True)
        else:
            self.observation_space_enhanced = None
            
        self.black_box_system = BlackBoxPredictionSystem(duration=black_box_duration, frequency=black_box_frequency, enable_prediction_market=True)
        self.agent_states: Dict[str, AgentState] = {}
        self.agent_objects: Dict[str, BaseAgent] = {}
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        
        if agent_types is None:
            agent_types = ["rlaif"] * n_agents
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
        self.world_model_transitions: List[Dict[str, Any]] = []
        self._prev_agent_portfolio: Dict[str, float] = {}
        self._agent_action_hist: Dict[str, List[int]] = {aid: [] for aid in self.agents}
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
            agent_type_str = self.agent_types[i] if i < len(self.agent_types) else "rlaif"
            if agent_type_str == "rlaif":
                agent = RLAIFAgent(agent_id=agent_id, logger=self.logger, initial_cash=self.initial_cash)
            elif agent_type_str == "adversarial":
                agent = AdversarialAgent(agent_id=agent_id, logger=self.logger, initial_cash=self.initial_cash)
                agent.bind_event_manager(self.event_manager)
            else:
                agent = BaseAgent(agent_id=agent_id, initial_cash=self.initial_cash)
            self.agent_objects[agent_id] = agent
            self.agent_states[agent_id] = AgentState(
                agent_id, self.initial_cash, {}, self.initial_cash,
                (agent_type_str == "adversarial"),
                agent.state.villain_points if hasattr(agent.state, 'villain_points') else 0
            )
    
    def _init_spaces(self):
        self.action_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([4, 10, 1000, 10000]), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'price_history': spaces.Box(low=0, high=10000, shape=(50, 10), dtype=np.float32),
            'order_book_bids': spaces.Box(low=0, high=10000, shape=(10, 10, 2), dtype=np.float32),
            'order_book_asks': spaces.Box(low=0, high=10000, shape=(10, 10, 2), dtype=np.float32),
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
            random.seed(seed)
        self.order_book.reset()
        self.multi_stock_env.reset()
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
            if isinstance(agent, AdversarialAgent):
                agent.bind_event_manager(self.event_manager)
        self._prev_agent_portfolio = {aid: self.initial_cash for aid in self.agents}
        self._agent_action_hist = {aid: [] for aid in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        observations = {agent: self._observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos
    
    def step(self, action: Any = None) -> Tuple[Dict, float | Dict, bool, bool, Dict]:
        """
        Step method supporting both joint dict actions (all agents at once) with a
        randomized step iterator for execution fairness, AND single-agent actions.
        """
        if isinstance(action, dict):
            return self._step_joint(action)
        else:
            return self._step_single(action)

    def _step_joint(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict[str, float], bool, bool, Dict]:
        """Process actions for all agents in a randomized execution order each step for fairness."""
        rewards = {}
        infos = {agent_id: {} for agent_id in self.agents}

        # Randomized iterator for step execution fairness across agents
        randomized_agent_order = random.sample(self.agents, len(self.agents))

        for agent_id in randomized_agent_order:
            agent_state = self.agent_states[agent_id]
            if agent_state.is_blocked:
                agent_state.block_remaining -= 1
                if agent_state.block_remaining <= 0:
                    agent_state.is_blocked = False
                rewards[agent_id] = self._calculate_reward(agent_id, None)
                continue

            agent_obj = self.agent_objects[agent_id]
            act = actions.get(agent_id)
            if act is None:
                obs = self._observe(agent_id)
                act = agent_obj.decide_action(obs, self.action_space)

            self._execute_action(agent_id, act)
            self._apply_adversarial_event_trigger(agent_id, agent_obj)

            # Sync agent state
            agent_obj.state.cash = agent_state.cash
            agent_obj.state.holdings = agent_state.holdings
            agent_obj.state.total_value = agent_state.total_value
            agent_obj.state.is_bad = agent_state.is_bad
            agent_obj.state.villain_points = agent_state.villain_points

            info = self._build_agent_step_info(agent_id, act)
            r = self._calculate_reward(agent_id, act, info)
            rewards[agent_id] = r
            agent_obj.update_state(r, False, info)
            if act is not None:
                self._agent_action_hist[agent_id].append(int(act[0]))
                if len(self._agent_action_hist[agent_id]) > 30:
                    self._agent_action_hist[agent_id].pop(0)

        # Update market, events, and black box after joint actions
        self._update_market()
        self._process_events()

        if self.black_box_system.should_trigger():
            self.is_black_box = True
            self.logger.info(LogCategory.EVENT, f"Black box period triggered at step {self.current_step}", data={'step': self.current_step})

        if self.is_black_box:
            self.black_box_system.update()
            if not self.black_box_system.is_active:
                self.is_black_box = False
                primary_asset = self.available_assets[0]
                actual_values = self.price_data.get(primary_asset, [100.0])[-self.black_box_duration:]
                self.black_box_system.evaluate_predictions(actual_values)

        self._check_agent_transformations()

        self.current_step += 1
        self.current_movement += 1
        if self.current_movement >= self.movements_per_day:
            self.current_movement = 0
            self.current_day += 1

        terminated, truncated = self._check_termination()
        return self._get_observations(), rewards, terminated, truncated, infos

    def _step_single(self, action: Optional[np.ndarray] = None) -> Tuple[Dict, float, bool, bool, Dict]:
        """AEC single-agent step."""
        agent = self.agent_selection
        if self.agent_states[agent].is_blocked:
            self.agent_states[agent].block_remaining -= 1
            if self.agent_states[agent].block_remaining <= 0:
                self.agent_states[agent].is_blocked = False
            reward = self._calculate_reward(agent, None)
            self.agent_selection = self._agent_selector.next()
            return self._get_observations(), reward, False, False, {agent: {}}

        agent_obj = self.agent_objects[agent]
        if action is None:
            observation = self._observe(agent)
            action = agent_obj.decide_action(observation, self.action_space)

        self._execute_action(agent, action)
        self._update_market()
        self._process_events()

        if self.black_box_system.should_trigger():
            self.is_black_box = True
            self.logger.info(LogCategory.EVENT, f"Black box period triggered at step {self.current_step}", data={'step': self.current_step})

        if self.is_black_box:
            self.black_box_system.update()
            if not self.black_box_system.is_active:
                self.is_black_box = False
                primary_asset = self.available_assets[0]
                actual_values = self.price_data.get(primary_asset, [100.0])[-self.black_box_duration:]
                self.black_box_system.evaluate_predictions(actual_values)

        self._check_agent_transformations()
        agent_obj.state.cash = self.agent_states[agent].cash
        agent_obj.state.holdings = self.agent_states[agent].holdings
        agent_obj.state.total_value = self.agent_states[agent].total_value
        agent_obj.state.is_bad = self.agent_states[agent].is_bad
        agent_obj.state.villain_points = self.agent_states[agent].villain_points

        info = self._build_agent_step_info(agent, action)
        reward = self._calculate_reward(agent, action, info)
        agent_obj.update_state(reward, False, info)
        terminated, truncated = self._check_termination()
        self.agent_selection = self._agent_selector.next()

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
        price = float(action[3])
        asset_name = self.available_assets[asset_idx] if asset_idx < len(self.available_assets) else self.available_assets[0]
        agent_state = self.agent_states[agent_id]
        agent_state.last_action = {'type': action_type, 'asset': asset_name, 'quantity': quantity, 'price': price}

        if action_type == 0:
            return

        current_asset_price = self.current_prices.get(asset_name, 100.0)
        
        if action_type == 1:
            best_ask = self.order_book.get_best_ask(asset_name)
            actual_price = best_ask[0] if best_ask else current_asset_price
            max_quantity = min(quantity, int(agent_state.cash / max(actual_price, 1e-5)))
            if max_quantity > 0:
                agent_state.cash -= max_quantity * actual_price
                agent_state.holdings[asset_name] = agent_state.holdings.get(asset_name, 0) + max_quantity
                self.order_book.add_order(agent_id, asset_name, OrderSide.BUY, actual_price, max_quantity)
        
        elif action_type == 2: 
            best_bid = self.order_book.get_best_bid(asset_name)
            actual_price = best_bid[0] if best_bid else current_asset_price
            current_holdings = agent_state.holdings.get(asset_name, 0)
            max_quantity = min(quantity, current_holdings)
            if max_quantity > 0:
                agent_state.cash += max_quantity * actual_price
                agent_state.holdings[asset_name] = current_holdings - max_quantity
                if agent_state.holdings[asset_name] == 0:
                    del agent_state.holdings[asset_name]
                self.order_book.add_order(agent_id, asset_name, OrderSide.SELL, actual_price, max_quantity)
        
        elif action_type == 3: 
            if price > 0 and quantity > 0:
                cost = quantity * price
                if agent_state.cash >= cost:
                    self.order_book.add_order(agent_id, asset_name, OrderSide.BUY, price, quantity)
        
        elif action_type == 4:
            if price > 0 and quantity > 0:
                current_holdings = agent_state.holdings.get(asset_name, 0)
                if current_holdings >= quantity:
                    self.order_book.add_order(agent_id, asset_name, OrderSide.SELL, price, quantity)
        
        holdings_value = sum(qty * self.current_prices.get(asset, 0.0) for asset, qty in agent_state.holdings.items())
        agent_state.total_value = agent_state.cash + holdings_value
        agent_state.profit_history.append(agent_state.total_value - self.initial_cash)
    
    def _update_market(self):
        for asset in self.available_assets:
            mid_price = self.order_book.get_mid_price(asset)
            if mid_price is not None:
                self.current_prices[asset] = mid_price
            else:
                asset_prices = self.price_data.get(asset, [])
                price_idx = min(self.current_step, len(asset_prices) - 1)
                self.current_prices[asset] = asset_prices[price_idx] if asset_prices else 100.0
        self.multi_stock_env.update_prices(self.current_prices)
        for asset in self.available_assets:
            self.price_history[asset].append(self.current_prices[asset])
        for asset in self.available_assets:
            self.market_cap[asset] = self.current_prices[asset] * 10000
        primary_asset = self.available_assets[0]
        if self.current_prices[primary_asset] < self.price_history[primary_asset][0] * 0.5:
            self.economy_collapsed = True
    
    def _market_state_for_events(self) -> Dict[str, Any]:
        primary = self.available_assets[0]
        hist = self.price_history.get(primary, [100.0])
        vol = 0.02
        if len(hist) > 5:
            rets = np.diff(np.array(hist[-20:], dtype=np.float64)) / np.maximum(np.array(hist[-20:-1], dtype=np.float64), 1e-5)
            vol = float(np.std(rets)) if len(rets) else 0.02
        return {'current_price': self.current_prices.get(primary, 100.0), 'market_cap': self.market_cap.get(primary, 1e6), 'volatility': vol}

    def _apply_adversarial_event_trigger(self, agent_id: str, agent_obj: BaseAgent):
        if not isinstance(agent_obj, AdversarialAgent):
            return
        event_id = agent_obj.consume_pending_event()
        if event_id is None:
            return
        market_state = self._market_state_for_events()
        event = self.event_manager.get_event(event_id)
        if event is None or not self.event_manager.trigger_event(event_id, market_state):
            agent_obj.earn_villain_points(2)
            return
        damage = abs(event.effects.price_drift) + abs(event.effects.volatility_multiplier - 1.0) + event.effects.jump_probability
        agent_obj.record_event_trigger(event_id, event.name, damage)

    def _process_events(self):
        self.event_manager.update(self._market_state_for_events())
        active_events = self.event_manager.get_active_events()
        for event in active_events:
            if event.effects:
                for asset in self.available_assets:
                    self.current_prices[asset] *= (1 + event.effects.price_drift)
                    self.current_prices[asset] = max(self.current_prices[asset], 1.0)
    
    def _check_agent_transformations(self):
        current_bad = sum(1 for state in self.agent_states.values() if state.is_bad)
        if current_bad < self.max_bad_agents:
            for agent_id, state in self.agent_states.items():
                if not state.is_bad and state.total_value < self.initial_cash * 0.5:
                    state.is_bad = True
                    state.villain_points = 10
                    break
    
    def _action_entropy(self, agent_id: str) -> float:
        hist = self._agent_action_hist.get(agent_id, [])
        if len(hist) < 3:
            return 0.3
        counts = np.bincount(hist, minlength=5)
        probs = counts / max(1, counts.sum())
        return float(-np.sum(probs * np.log(probs + 1e-8)) / np.log(5))

    def _market_unpredictability(self) -> float:
        primary_asset = self.available_assets[0]
        hist = self.price_history.get(primary_asset, [])
        if len(hist) < 10:
            return 0.2
        rets = np.diff(np.array(hist[-10:], dtype=np.float64)) / np.maximum(np.array(hist[-10:-1], dtype=np.float64), 1e-5)
        vol_score = min(1.0, float(np.std(rets)) * 25.0)
        jump_score = min(1.0, float(np.max(np.abs(rets))) * 10.0)
        return max(0.0, min(1.0, 0.6 * vol_score + 0.4 * jump_score))

    def _build_agent_step_info(self, agent_id: str, action: Optional[np.ndarray]) -> Dict[str, Any]:
        agent_state = self.agent_states[agent_id]
        prev_val = self._prev_agent_portfolio.get(agent_id, agent_state.total_value)
        profit_delta = (agent_state.total_value - prev_val) / max(self.initial_cash, 1e-5)
        profit_component = (agent_state.total_value - self.initial_cash) / max(self.initial_cash, 1e-5)
        unpredictability = max(self._action_entropy(agent_id), self._market_unpredictability())
        info = {'profit_component': profit_component, 'profit_delta': profit_delta, 'unpredictability_score': unpredictability, 'market_damage': 0.0, 'max_bad_agents_reached': False}
        self._prev_agent_portfolio[agent_id] = agent_state.total_value
        if isinstance(self.agent_objects.get(agent_id), AdversarialAgent):
            adv = self.agent_objects[agent_id]
            if adv.triggered_events:
                info['market_damage'] = float(adv.triggered_events[-1].get('market_damage', 0.0))
        return info

    def _calculate_reward(self, agent_id: str, action: Optional[np.ndarray], info: Optional[Dict[str, Any]] = None) -> float:
        agent_state = self.agent_states[agent_id]
        info = info or self._build_agent_step_info(agent_id, action)
        profit_component = float(info.get('profit_component', 0.0))
        unpredictability = float(info.get('unpredictability_score', 0.0))
        if agent_state.is_bad:
            reward = 0.35 * profit_component + 0.65 * unpredictability
            if unpredictability < 0.15:
                reward -= 0.08
        else:
            reward = profit_component
            primary_asset = self.available_assets[0]
            hist = self.price_history.get(primary_asset, [])
            if len(hist) > 10 and hist[-10] > 0:
                recent_change = (hist[-1] - hist[-10]) / hist[-10]
                if abs(recent_change) < 0.01:
                    reward -= 0.05
        if self.is_black_box and action is not None and action[0] == 0:
            reward -= 0.1
        if agent_state.is_bad and self.economy_collapsed:
            reward += 0.5
        return reward
    
    def _check_termination(self) -> Tuple[bool, bool]:
        if self.economy_collapsed:
            return True, False
        
        primary_asset = self.available_assets[0]
        hist = self.price_history.get(primary_asset, [])
        if len(hist) > 0 and hist[0] > 0:
            growth = (self.current_prices[primary_asset] - hist[0]) / hist[0]
            if growth > self.economy_growth_threshold:
                return True, False
        if self.current_step >= self.episode_length:
            return True, False
        return False, False
    
    def _observe(self, agent_id: str) -> Dict:
        agent_state = self.agent_states[agent_id]
        price_hist_matrix = np.zeros((50, 10), dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            asset_hist = self.price_history[asset][-50:]
            if len(asset_hist) < 50:
                asset_hist = [0.0] * (50 - len(asset_hist)) + asset_hist
            price_hist_matrix[:, i] = np.array(asset_hist, dtype=np.float32)
        bids_matrix = np.zeros((10, 10, 2), dtype=np.float32)
        asks_matrix = np.zeros((10, 10, 2), dtype=np.float32)
        
        for i, asset in enumerate(self.available_assets[:10]):
            order_book_state = self.order_book.get_order_book_state(asset)
            bids_list = order_book_state.get('bids', [])[:10]
            asks_list = order_book_state.get('asks', [])[:10]
            
            if len(bids_list) > 0:
                bids = np.array(bids_list, dtype=np.float32)
                if bids.ndim == 1:
                    bids = bids.reshape(-1, 1)
                    bids = np.hstack([bids, np.zeros((bids.shape[0], 1), dtype=np.float32)])
                for j, bid in enumerate(bids[:10]):
                    bids_matrix[i, j] = bid
            
            if len(asks_list) > 0:
                asks = np.array(asks_list, dtype=np.float32)
                if asks.ndim == 1:
                    asks = asks.reshape(-1, 1)
                    asks = np.hstack([asks, np.zeros((asks.shape[0], 1), dtype=np.float32)])
                for j, ask in enumerate(asks[:10]):
                    asks_matrix[i, j] = ask
        
        holdings_array = np.zeros(10, dtype=np.int32)
        for i, asset in enumerate(self.available_assets[:10]):
            holdings_array[i] = agent_state.holdings.get(asset, 0)
        
        prices_array = np.zeros(10, dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            prices_array[i] = self.current_prices.get(asset, 100.0)
        
        market_cap_array = np.zeros(10, dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            market_cap_array[i] = self.market_cap.get(asset, 1000000.0)
        
        available_assets_array = np.zeros(10, dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            available_assets_array[i] = 1.0
        
        active_events = self.event_manager.get_active_events()
        event_vector = np.zeros(10, dtype=np.float32)
        for i, event in enumerate(active_events[:10]):
            event_vector[i] = 1.0
        market_state = self._market_state_for_events()
        triggerable_ids, triggerable_mask = self.event_manager.agent_trigger_catalog(market_state, MAX_TRIGGERABLE_EVENTS)
        padded_ids = triggerable_ids + [-1] * (MAX_TRIGGERABLE_EVENTS - len(triggerable_ids))
        base_observation = {'price_history': price_hist_matrix, 'order_book_bids': bids_matrix, 'order_book_asks': asks_matrix, 'agent_cash': np.array(agent_state.cash, dtype=np.float32), 'agent_holdings': holdings_array, 'agent_total_value': np.array(agent_state.total_value, dtype=np.float32), 'is_black_box': 1 if self.is_black_box else 0, 'current_price': prices_array, 'market_cap': market_cap_array, 'active_events': event_vector, 'available_assets': available_assets_array, 'asset_names': self.available_assets[:10], 'triggerable_event_ids': padded_ids[:MAX_TRIGGERABLE_EVENTS], 'triggerable_event_mask': triggerable_mask}

        if self.use_enhanced_observation and self.observation_space_enhanced:
            primary_asset = self.available_assets[0]
            agent_state_dict = {'cash': agent_state.cash, 'holdings': agent_state.holdings, 'total_value': agent_state.total_value}
            primary_order_book_state = self.order_book.get_order_book_state(primary_asset)
            best_bid = primary_order_book_state.get('best_bid')
            best_ask = primary_order_book_state.get('best_ask')
            market_state_dict = {
                'current_price': self.current_prices[primary_asset],
                'best_bid': best_bid[0] if best_bid else self.current_prices[primary_asset] * 0.99,
                'best_ask': best_ask[0] if best_ask else self.current_prices[primary_asset] * 1.01,
                'spread': primary_order_book_state.get('spread', 0.02)
            }
            enhanced_obs = self.observation_space_enhanced.create_observation(base_observation, agent_state_dict, market_state_dict)
            base_observation.update(enhanced_obs)
        return base_observation
    
    def _get_observations(self) -> Dict:
        return {agent: self._observe(agent) for agent in self.agents}
    
    def render(self, mode: str = "human"):
        if mode == "human":
            print(f"Step: {self.current_step}, Day: {self.current_day}")
            print(f"Available Assets: {self.available_assets}")
            print(f"Current Prices: {self.current_prices}")
            print(f"Market Caps: {self.market_cap}")
            print(f"Black Box: {self.is_black_box}")
            print("\nAgent States:")
            for agent_id, state in self.agent_states.items():
                print(f"  {agent_id}: Cash={state.cash:.2f}, Holdings={state.holdings}, Total={state.total_value:.2f}, Bad={state.is_bad}")
            print("\nOrder Books:")
            for asset in self.available_assets:
                ob_state = self.order_book.get_order_book_state(asset)
                print(f"  {asset}: Bid={ob_state.get('best_bid')}, Ask={ob_state.get('best_ask')}, Spread={ob_state.get('spread')}")
            print("-" * 50)
    
    def close(self):
        pass
