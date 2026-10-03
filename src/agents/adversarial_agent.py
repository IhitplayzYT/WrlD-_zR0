from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
import numpy as np
import random
from .base_agent import BaseAgent, AgentTraits, AgentType
from src.utils.logging import JSONLogger, get_logger, LogCategory


@dataclass
class ManipulationStrategy:
    strategy_type: str = "pump_and_dump"
    target_price: Optional[float] = None
    duration: int = 10
    intensity: float = 0.5
    cooldown: int = 20
    
    def execute(self, current_price: float, agent_cash: float, agent_holdings: int, asset_idx: int = 0) -> Tuple[int, int, int, float]:
        """ Returns (action_type, asset_idx, quantity, price) """
        if self.strategy_type == "pump_and_dump":
            return self._pump_and_dump(current_price, agent_cash, agent_holdings, asset_idx)
        elif self.strategy_type == "bear_raid":
            return self._bear_raid(current_price, agent_cash, agent_holdings, asset_idx)
        elif self.strategy_type == "spoofing":
            return self._spoofing(current_price, agent_cash, agent_holdings, asset_idx)
        elif self.strategy_type == "layering":
            return self._layering(current_price, agent_cash, agent_holdings, asset_idx)
        else:
            return self._random_manipulation(current_price, agent_cash, agent_holdings, asset_idx)
    
    def _pump_and_dump(self, current_price: float, agent_cash: float, agent_holdings: int, asset_idx: int = 0) -> Tuple[int, int, int, float]:
        if agent_holdings < 100:
            quantity = min(50, max(1, int(agent_cash / max(current_price, 1e-5))))
            return (1, asset_idx, quantity, current_price * 1.01) 
        else:
            return (2, asset_idx, max(1, agent_holdings), current_price * 0.99) 
    
    def _bear_raid(self, current_price: float, agent_cash: float, agent_holdings: int, asset_idx: int = 0) -> Tuple[int, int, int, float]:
        if agent_holdings >= 0:
            quantity = min(30, max(1, int(agent_cash / max(current_price, 1e-5))))
            return (2, asset_idx, quantity, current_price * 0.98)  
        else:
            return (1, asset_idx, max(1, abs(agent_holdings)), current_price * 0.99)
    
    def _spoofing(self, current_price: float, agent_cash: float, agent_holdings: int, asset_idx: int = 0) -> Tuple[int, int, int, float]:
        if random.random() < 0.5:
            quantity = 100
            price = current_price * 1.02
            return (3, asset_idx, quantity, price)
        else:
            quantity = 100
            price = current_price * 0.98
            return (4, asset_idx, quantity, price)
    
    def _layering(self, current_price: float, agent_cash: float, agent_holdings: int, asset_idx: int = 0) -> Tuple[int, int, int, float]:
        level = random.randint(1, 5)
        if random.random() < 0.5:
            price = current_price * (1 - level * 0.005)
            quantity = 20
            return (3, asset_idx, quantity, price)
        else:
            price = current_price * (1 + level * 0.005)
            quantity = 20
            return (4, asset_idx, quantity, price)
    
    def _random_manipulation(
        self,
        current_price: float,
        agent_cash: float,
        agent_holdings: int,
        asset_idx: int = 0
    ) -> Tuple[int, int, int, float]:
        action_type = random.choice([1, 2, 3, 4])
        quantity = random.randint(10, 50)
        price = current_price * random.uniform(0.95, 1.05)
        return (action_type, asset_idx, quantity, price)


class AdversarialAgent(BaseAgent):
    """
    Adversarial agent that tries to manipulate the market and crash good stock momentum.
    """
    
    def __init__(
        self,
        agent_id: str,
        logger: Optional[JSONLogger] = None,
        initial_cash: float = 1000.0,
        traits: Optional[AgentTraits] = None,
        villain_points: int = 10
    ):
        if traits is None:
            traits = self._create_adversarial_traits()
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ADVERSARIAL,
            initial_cash=initial_cash,
            traits=traits
        )
        self.logger = logger or get_logger()
        self.state.is_bad = True
        self.state.villain_points = villain_points
        
        self.manipulation_strategies = [
            ManipulationStrategy("pump_and_dump"),
            ManipulationStrategy("bear_raid"),
            ManipulationStrategy("spoofing"),
            ManipulationStrategy("layering")
        ]
        self.current_strategy = None
        self.strategy_cooldown = 0
        self.triggered_events = []
        self.event_trigger_cost = 5
    
    def _create_adversarial_traits(self) -> AgentTraits:
        traits = AgentTraits()
        traits.risk_tolerance = 0.8
        traits.loss_aversion = 0.3
        traits.patience = 0.2
        traits.holding_period_preference = 0.1
        traits.contrarian_bias = 0.9
        traits.trend_following = 0.1
        traits.herd_behavior = 0.1
        traits.confidence_level = 0.8
        traits.adaptability = 0.9
        traits.exploration_rate = 0.8
        traits.order_size_preference = 0.9
        traits.limit_order_preference = 0.3
        return traits
    
    def decide_action(
        self,
        observation: Dict[str, Any],
        action_space: Any
    ) -> np.ndarray:
        processed_obs = self.process_observation(observation)
        
        if self._should_trigger_event(processed_obs):
            self._trigger_bad_event(processed_obs)
        
        if self.strategy_cooldown > 0:
            self.strategy_cooldown -= 1
        elif self.current_strategy is None or random.random() < 0.1:
            self.current_strategy = random.choice(self.manipulation_strategies)
            self.strategy_cooldown = self.current_strategy.cooldown
        
        available_assets = processed_obs.get('available_assets', [1.0] + [0.0] * 9)
        if isinstance(available_assets, np.ndarray):
            asset_idx = int(np.argmax(available_assets))
        else:
            asset_idx = 0
            
        current_prices = processed_obs.get('current_price', [100.0] * 10)
        if isinstance(current_prices, np.ndarray) and len(current_prices) > asset_idx:
            current_price = float(current_prices[asset_idx])
        elif isinstance(current_prices, list) and len(current_prices) > asset_idx:
            current_price = float(current_prices[asset_idx])
        else:
            current_price = 100.0
            
        agent_cash = self.state.cash
        asset_name = f"ASSET{asset_idx}"
        agent_holdings = self.state.holdings.get(asset_name, 0)
        
        action_type, asset_idx, quantity, price = self.current_strategy.execute(
            current_price,
            agent_cash,
            agent_holdings,
            asset_idx
        )
        
        base_action = np.array([action_type, asset_idx, quantity, price], dtype=np.float32)
        modified_action = self.apply_trait_modifications(base_action)
        
        self.state.last_action = {
            'type': int(modified_action[0]),
            'asset_idx': int(modified_action[1]),
            'quantity': int(modified_action[2]),
            'price': float(modified_action[3]),
            'strategy': self.current_strategy.strategy_type if self.current_strategy else None
        }
        
        return modified_action
    
    def _should_trigger_event(self, observation: Dict) -> bool:
        if self.state.villain_points < self.event_trigger_cost:
            return False
        
        price_history = observation.get('price_history', [])
        if isinstance(price_history, np.ndarray) and price_history.ndim == 2 and price_history.shape[0] >= 10:
            p0 = price_history[-10, 0]
            p1 = price_history[-1, 0]
            if p0 > 0:
                recent_trend = (p1 - p0) / p0
                if recent_trend > 0.05:
                    return True
        
        if random.random() < 0.02:
            return True
        
        return False
    
    def _trigger_bad_event(self, observation: Dict):
        self.state.villain_points -= self.event_trigger_cost
        cp = observation.get('current_price', [100.0])
        current_price = float(cp[0]) if isinstance(cp, (list, np.ndarray)) and len(cp) > 0 else 100.0
        
        event_info = {
            'type': 'bad_event',
            'triggered_by': self.agent_id,
            'timestamp': len(self.state.profit_history),
            'market_state': current_price
        }
        self.triggered_events.append(event_info)
        self.state.add_memory(event_info)
        self.logger.info(LogCategory.EVENT, f"Adversarial agent {self.agent_id} triggered a bad market event", data=event_info)
    
    def earn_villain_points(self, points: int):
        self.state.villain_points += points
    
    def update_state(self, reward: float, done: bool, info: Dict):
        super().update_state(reward, done, info)
        if info.get('market_damage', 0) > 0:
            damage = info['market_damage']
            earned_points = int(damage * 0.1)
            self.earn_villain_points(earned_points)
        if info.get('max_bad_agents_reached', False):
            self.state.is_blocked = True
    
    def get_manipulation_metrics(self) -> Dict[str, Any]:
        return {
            'villain_points': self.state.villain_points,
            'events_triggered': len(self.triggered_events),
            'current_strategy': self.current_strategy.strategy_type if self.current_strategy else None,
            'strategy_cooldown': self.strategy_cooldown,
            'total_manipulation_impact': sum(e.get('market_damage', 0) for e in self.triggered_events)
        }
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(self.get_manipulation_metrics())
        return info
    
    def reset(self, initial_cash: Optional[float] = None):
        super().reset(initial_cash)
        self.current_strategy = None
        self.strategy_cooldown = 0
        self.triggered_events = []
