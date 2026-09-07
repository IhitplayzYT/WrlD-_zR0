from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import random

from .base_agent import BaseAgent, AgentTraits, AgentType, AgentState


@dataclass
class ManipulationStrategy:
    """
    Strategy for market manipulation by adversarial agents.
    """
    strategy_type: str = "pump_and_dump"
    target_price: Optional[float] = None
    duration: int = 10
    intensity: float = 0.5
    cooldown: int = 20
    
    def execute(
        self,
        current_price: float,
        agent_cash: float,
        agent_holdings: int,
        asset_idx: int = 0
    ) -> Tuple[int, int, int, float]:
        """
        Execute manipulation strategy.
        Returns (action_type, asset_idx, quantity, price)
        """
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
    
    def _pump_and_dump(
        self,
        current_price: float,
        agent_cash: float,
        agent_holdings: int,
        asset_idx: int = 0
    ) -> Tuple[int, int, int, float]:
        """Pump price up then sell"""
        # Phase 1: Buy aggressively to pump price
        if agent_holdings < 100:
            quantity = min(50, int(agent_cash / current_price))
            return (1, asset_idx, quantity, current_price * 1.01)  # Market buy
        # Phase 2: Sell to dump
        else:
            return (2, asset_idx, agent_holdings, current_price * 0.99)  # Market sell
    
    def _bear_raid(
        self,
        current_price: float,
        agent_cash: float,
        agent_holdings: int,
        asset_idx: int = 0
    ) -> Tuple[int, int, int, float]:
        """Drive price down aggressively"""
        # Short selling simulation (sell without holding)
        if agent_holdings >= 0:
            quantity = min(30, int(agent_cash / current_price))
            return (2, asset_idx, quantity, current_price * 0.98)  # Aggressive sell
        else:
            # Cover short position
            return (1, asset_idx, abs(agent_holdings), current_price * 0.99)  # Buy to cover
    
    def _spoofing(
        self,
        current_price: float,
        agent_cash: float,
        agent_holdings: int,
        asset_idx: int = 0
    ) -> Tuple[int, int, int, float]:
        """Place fake orders to create illusion of demand/supply"""
        # Place large limit orders then cancel them
        if random.random() < 0.5:
            # Fake buy order
            quantity = 100
            price = current_price * 1.02
            return (3, asset_idx, quantity, price)  # Limit buy (will be cancelled)
        else:
            # Fake sell order
            quantity = 100
            price = current_price * 0.98
            return (4, asset_idx, quantity, price)  # Limit sell (will be cancelled)
    
    def _layering(
        self,
        current_price: float,
        agent_cash: float,
        agent_holdings: int,
        asset_idx: int = 0
    ) -> Tuple[int, int, int, float]:
        """Place multiple orders at different price levels"""
        # Place orders at multiple levels to create false depth
        level = random.randint(1, 5)
        if random.random() < 0.5:
            # Buy side layering
            price = current_price * (1 - level * 0.005)
            quantity = 20
            return (3, asset_idx, quantity, price)  # Limit buy
        else:
            # Sell side layering
            price = current_price * (1 + level * 0.005)
            quantity = 20
            return (4, asset_idx, quantity, price)  # Limit sell
    
    def _random_manipulation(
        self,
        current_price: float,
        agent_cash: float,
        agent_holdings: int,
        asset_idx: int = 0
    ) -> Tuple[int, int, int, float]:
        """Random manipulation strategy"""
        action_type = random.choice([1, 2, 3, 4])
        quantity = random.randint(10, 50)
        price = current_price * random.uniform(0.95, 1.05)
        return (action_type, asset_idx, quantity, price)


class AdversarialAgent(BaseAgent):
    """
    Adversarial agent that tries to manipulate the market and crash good stock momentum.
    Transforms from good agent when performance drops below threshold.
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 1000.0,
        traits: Optional[AgentTraits] = None,
        villain_points: int = 10
    ):
        # Initialize with adversarial traits if none provided
        if traits is None:
            traits = self._create_adversarial_traits()
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ADVERSARIAL,
            initial_cash=initial_cash,
            traits=traits
        )
        
        self.state.is_bad = True
        self.state.villain_points = villain_points
        
        # Manipulation strategies
        self.manipulation_strategies = [
            ManipulationStrategy("pump_and_dump"),
            ManipulationStrategy("bear_raid"),
            ManipulationStrategy("spoofing"),
            ManipulationStrategy("layering")
        ]
        self.current_strategy = None
        self.strategy_cooldown = 0
        
        # Event triggering capabilities
        self.triggered_events = []
        self.event_trigger_cost = 5  # Villain points needed to trigger event
    
    def _create_adversarial_traits(self) -> AgentTraits:
        """Create traits optimized for adversarial behavior"""
        traits = AgentTraits()
        
        # High risk tolerance for aggressive manipulation
        traits.risk_tolerance = 0.8
        traits.loss_aversion = 0.3  # Less sensitive to losses
        
        # Low patience for quick manipulation
        traits.patience = 0.2
        traits.holding_period_preference = 0.1
        
        # High contrarian bias to go against market trends
        traits.contrarian_bias = 0.9
        traits.trend_following = 0.1
        
        # Low herd behavior to act independently
        traits.herd_behavior = 0.1
        traits.confidence_level = 0.8
        
        # High adaptability to change strategies quickly
        traits.adaptability = 0.9
        traits.exploration_rate = 0.8
        
        # Large order sizes for market impact
        traits.order_size_preference = 0.9
        traits.limit_order_preference = 0.3  # Prefer market orders for impact
        
        return traits
    
    def decide_action(
        self,
        observation: Dict[str, Any],
        action_space: Any
    ) -> np.ndarray:
        """
        Decide action based on manipulation strategy and adversarial objectives.
        """
        # Process observation
        processed_obs = self.process_observation(observation)
        
        # Check if we should trigger a bad event
        if self._should_trigger_event(processed_obs):
            self._trigger_bad_event(processed_obs)
        
        # Select or continue manipulation strategy
        if self.strategy_cooldown > 0:
            self.strategy_cooldown -= 1
        elif self.current_strategy is None or random.random() < 0.1:
            self.current_strategy = random.choice(self.manipulation_strategies)
            self.strategy_cooldown = self.current_strategy.cooldown
        
        # Execute strategy
        available_assets = processed_obs.get('available_assets', [1.0] + [0.0] * 9)
        asset_idx = int(np.argmax(available_assets))  # Select first available asset
        current_price = processed_obs.get('current_price', [100.0] * 10)[asset_idx]
        agent_cash = self.state.cash
        agent_holdings = self.state.holdings.get(f'ASSET{asset_idx}', 0)
        
        action_type, asset_idx, quantity, price = self.current_strategy.execute(
            current_price,
            agent_cash,
            agent_holdings,
            asset_idx
        )
        
        # Apply trait modifications
        base_action = np.array([action_type, asset_idx, quantity, price], dtype=np.float32)
        modified_action = self.apply_trait_modifications(base_action)
        
        # Store action
        self.state.last_action = {
            'type': int(modified_action[0]),
            'asset_idx': int(modified_action[1]),
            'quantity': int(modified_action[2]),
            'price': float(modified_action[3]),
            'strategy': self.current_strategy.strategy_type if self.current_strategy else None
        }
        
        return modified_action
    
    def _should_trigger_event(self, observation: Dict) -> bool:
        """Determine if agent should spend villain points to trigger a bad event"""
        # Need enough villain points
        if self.state.villain_points < self.event_trigger_cost:
            return False
        
        # Check market conditions
        current_price = observation.get('current_price', 100.0)
        price_history = observation.get('price_history', [])
        
        if len(price_history) >= 10:
            recent_trend = (price_history[-1] - price_history[-10]) / price_history[-10]
            
            # Trigger event if market is doing well (to crash it)
            if recent_trend > 0.05:  # 5% upward trend
                return True
        
        # Random chance to trigger event
        if random.random() < 0.02:  # 2% chance per step
            return True
        
        return False
    
    def _trigger_bad_event(self, observation: Dict):
        """Trigger a bad market event using villain points"""
        self.state.villain_points -= self.event_trigger_cost
        
        # Record triggered event
        event_info = {
            'type': 'bad_event',
            'triggered_by': self.agent_id,
            'timestamp': len(self.state.profit_history),
            'market_state': observation.get('current_price', 100.0)
        }
        self.triggered_events.append(event_info)
        
        # Add to memory
        self.state.add_memory(event_info)
    
    def earn_villain_points(self, points: int):
        """Earn villain points through successful market manipulation"""
        self.state.villain_points += points
    
    def update_state(self, reward: float, done: bool, info: Dict):
        """Update state with adversarial-specific logic"""
        super().update_state(reward, done, info)
        
        # Earn villain points if causing market damage
        if info.get('market_damage', 0) > 0:
            damage = info['market_damage']
            earned_points = int(damage * 0.1)
            self.earn_villain_points(earned_points)
        
        # Check if agent should be removed (max bad agent limit)
        # This is handled by the environment, but we track it here
        if info.get('max_bad_agents_reached', False):
            self.state.is_blocked = True
    
    def get_manipulation_metrics(self) -> Dict[str, Any]:
        """Get metrics about manipulation activities"""
        return {
            'villain_points': self.state.villain_points,
            'events_triggered': len(self.triggered_events),
            'current_strategy': self.current_strategy.strategy_type if self.current_strategy else None,
            'strategy_cooldown': self.strategy_cooldown,
            'total_manipulation_impact': sum(
                e.get('market_damage', 0) for e in self.triggered_events
            )
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get extended agent information including adversarial-specific data"""
        info = super().get_info()
        info.update(self.get_manipulation_metrics())
        return info
    
    def reset(self, initial_cash: Optional[float] = None):
        """Reset agent state for new episode"""
        super().reset(initial_cash)
        # Reset strategy
        self.current_strategy = None
        self.strategy_cooldown = 0
        # Keep villain points (they persist across episodes)
        # But reset triggered events for new episode
        self.triggered_events = []


if __name__ == "__main__":
    # Test the adversarial agent
    agent = AdversarialAgent("test_adversarial", initial_cash=1000.0)
    
    # Create a dummy observation
    dummy_obs = {
        'price_history': [100.0 + i * 0.2 for i in range(50)],  # Upward trend
        'order_book_bids': np.zeros((10, 2)),
        'order_book_asks': np.zeros((10, 2)),
        'agent_cash': 1000.0,
        'agent_holdings': 0,
        'agent_total_value': 1000.0,
        'is_black_box': 0,
        'current_price': 110.0,
        'market_cap': 1000000.0,
        'active_events': np.zeros(10)
    }
    
    # Test action decision
    action = agent.decide_action(dummy_obs, None)
    print(f"Action: {action}")
    print(f"Agent info: {agent.get_info()}")
    print(f"Manipulation metrics: {agent.get_manipulation_metrics()}")
