"""Budgeted adversarial minority: trades plus simulated event interventions."""
import numpy as np
from .base_agent import AgentType, AgentTraits
from .rlaif_agent import RLAIFAgent

MAX_TRIGGERABLE_EVENTS = 32


class AdversarialAgent(RLAIFAgent):
    def __init__(self, agent_id, logger=None, initial_cash=1000., traits=None,
                 villain_points=10, unpredictability_weight=.65):
        traits = traits or AgentTraits(risk_tolerance=.8, max_position_size=.6)
        super().__init__(agent_id, logger, initial_cash, traits, extra_actions=MAX_TRIGGERABLE_EVENTS)
        self.agent_type = self.state.agent_type = AgentType.ADVERSARIAL
        self.initial_villain_points = villain_points
        self.state.is_bad = True
        self.state.villain_points = villain_points
        self.event_trigger_cost = 5
        self.pending_event_id = None
        self.triggered_events = []
        self.event_manager = None
        # Smart triggering parameters
        self.market_memory = []  # Track recent market states for pattern recognition
        self.triggering_strategy = 'adaptive'  # 'adaptive', 'aggressive', 'conservative'
        # Kaiser targeting parameters
        self.kaiser_id = None
        self.kaiser_history = []  # Track Kaiser's performance and actions
        self.kaiser_damage_bonus = 0.0  # Bonus for damaging Kaiser
        # Heterogeneous agent characteristics
        self.aggression_style = np.random.choice(['front_run', 'liquidity_trap', 'momentum_reverse', 'volatility_spike'])
        self.risk_preference = np.random.uniform(0.3, 0.9)  # Higher = more risk-taking
        self.patience_factor = np.random.uniform(0.5, 2.0)  # Higher = more patient
        self.adaptation_rate = np.random.uniform(0.1, 0.5)  # How quickly they adapt to Kaiser
        # Stochastic decision parameters (higher for adversarial agents)
        self.decision_volatility = np.random.uniform(0.1, 0.3)  # Higher randomness than normal agents
        self.event_selection_noise = np.random.uniform(0.05, 0.15)  # Noise in event selection scoring

    def bind_event_manager(self, event_manager):
        self.event_manager = event_manager

    def _evaluate_market_conditions(self, obs):
        """Evaluate current market conditions for smart event triggering"""
        current_price = float(obs['current_price'][0])
        price_history = np.asarray(obs['price_history'])[:, 0]
        price_history = price_history[price_history > 0]
        
        if len(price_history) < 5:
            return {'volatility': 0.01, 'trend': 0, 'liquidity_stress': 0.5}
        
        returns = np.diff(np.log(price_history[-20:]))
        volatility = float(np.std(returns)) if len(returns) > 0 else 0.01
        trend = float(np.mean(returns)) if len(returns) > 0 else 0
        
        # Check order book depth (simplified from observation)
        bids = obs.get('order_book_bids', np.zeros((10, 10, 2)))
        asks = obs.get('order_book_asks', np.zeros((10, 10, 2)))
        bid_depth = np.sum(bids[:, :, 1]) if bids.ndim == 3 else 0
        ask_depth = np.sum(asks[:, :, 1]) if asks.ndim == 3 else 0
        total_depth = bid_depth + ask_depth
        liquidity_stress = 1.0 - min(1.0, total_depth / 10000.0)  # Higher when liquidity is low
        
        # Market confidence from active events
        active_events = obs.get('active_events', np.zeros(10))
        event_pressure = float(np.sum(active_events)) / 10.0
        
        return {
            'volatility': volatility,
            'trend': trend,
            'liquidity_stress': liquidity_stress,
            'event_pressure': event_pressure,
            'price': current_price
        }

    def _select_smart_event(self, obs, triggerable_ids, triggerable_mask):
        """Select the most damaging event based on market conditions and Kaiser targeting"""
        if not self.event_manager:
            return None
        
        conditions = self._evaluate_market_conditions(obs)
        
        # Score each triggerable event based on market conditions and Kaiser targeting
        best_event_id = None
        best_score = -float('inf')
        
        for event_id, is_triggerable in zip(triggerable_ids, triggerable_mask):
            if not is_triggerable or event_id < 0:
                continue
            
            event = self.event_manager.get_event(event_id)
            if not event:
                continue
            
            score = 0.0
            effects = event.effects
            
            # Base market condition scoring
            if conditions['volatility'] > 0.02 and effects.volatility_multiplier > 1.5:
                score += 2.0
            
            if conditions['liquidity_stress'] > 0.7 and effects.liquidity_multiplier < 0.8:
                score += 2.5
            
            if conditions['trend'] < -0.01 and effects.price_drift < -0.02:
                score += 1.5
            
            if conditions['trend'] > 0.01 and effects.price_drift > 0.02:
                score += 1.0
            
            if conditions['event_pressure'] > 0.3:
                if abs(effects.price_drift) > 0.05:
                    score += 1.0
            
            # Kaiser-targeting bonus based on aggression style
            if self.kaiser_history and len(self.kaiser_history) > 0:
                kaiser_info = self.kaiser_history[-1]
                kaiser_trend = kaiser_info.get('trend', 0)
                kaiser_position = kaiser_info.get('position_bias', 0)  # positive = long, negative = short
                
                if self.aggression_style == 'front_run':
                    # Try to front-run Kaiser's position
                    if kaiser_position > 0 and effects.price_drift < -0.03:
                        score += 3.0  # Crash while Kaiser is long
                    elif kaiser_position < 0 and effects.price_drift > 0.03:
                        score += 3.0  # Pump while Kaiser is short
                
                elif self.aggression_style == 'liquidity_trap':
                    # Trap Kaiser in illiquid positions
                    if effects.liquidity_multiplier < 0.5:
                        score += 2.5
                
                elif self.aggression_style == 'momentum_reverse':
                    # Reverse Kaiser's momentum
                    if kaiser_trend > 0 and effects.price_drift < -0.04:
                        score += 2.0
                    elif kaiser_trend < 0 and effects.price_drift > 0.04:
                        score += 2.0
                
                elif self.aggression_style == 'volatility_spike':
                    # Spike volatility to hurt Kaiser's predictions
                    if effects.volatility_multiplier > 2.0:
                        score += 2.0
            
            # Heterogeneous risk preference affects scoring
            if self.risk_preference > 0.7:
                # High risk: prefer high-impact events
                if abs(effects.price_drift) > 0.05:
                    score += 1.5
            else:
                # Low risk: prefer safer events
                if abs(effects.price_drift) < 0.03:
                    score += 1.0
            
            # Random factor varies by patience (less patient = more random)
            score += np.random.uniform(0, 0.5 * (2.0 - self.patience_factor))
            
            # Add stochastic noise to score for non-deterministic event selection
            score += np.random.normal(0, self.event_selection_noise * abs(score))
            
            if score > best_score:
                best_score = score
                best_event_id = event_id
        
        return best_event_id

    def _choices(self, obs):
        choices, mask = super()._choices(obs)
        event_mask = obs.get('triggerable_event_mask', np.zeros(MAX_TRIGGERABLE_EVENTS))
        for i in range(MAX_TRIGGERABLE_EVENTS):
            choices.append(np.zeros(4, dtype=np.float32))
            mask.append(bool(event_mask[i]) and self.state.villain_points >= self.event_trigger_cost and bool(obs.get('can_trade', True)))
        return choices, mask

    def decide_action(self, observation, action_space=None):
        choices, mask = self._choices(observation)
        
        # Use smart event selection for adversarial agent
        triggerable_ids = observation.get('triggerable_event_ids', [-1] * MAX_TRIGGERABLE_EVENTS)
        triggerable_mask = observation.get('triggerable_event_mask', np.zeros(MAX_TRIGGERABLE_EVENTS))
        
        # Check if we should trigger an event based on market conditions
        smart_event_id = self._select_smart_event(observation, triggerable_ids, triggerable_mask)
        
        # If smart selection found a good event, bias towards it
        if smart_event_id is not None:
            # Find the index of the smart event in triggerable_ids
            try:
                event_idx = triggerable_ids.index(smart_event_id)
                # Bias the policy by temporarily increasing the mask for this event
                if event_idx < len(mask):
                    # Let the policy decide, but if it chooses an event, prefer the smart one
                    pass
            except ValueError:
                pass
        
        choice = self.policy.select(observation, np.asarray(mask, dtype=bool), self.training_enabled)
        self.pending_event_id = None
        if choice >= 41:
            # If policy chose an event, use smart selection if available
            if smart_event_id is not None and choice - 41 < len(triggerable_ids):
                self.pending_event_id = smart_event_id
            else:
                self.pending_event_id = int(observation['triggerable_event_ids'][choice - 41])
        action = choices[choice]
        self.state.last_action = dict(type=int(action[0]), asset_idx=int(action[1]),
                                     quantity=int(action[2]), price=float(action[3]), event_id=self.pending_event_id)
        
        # Update market memory
        self.market_memory.append(self._evaluate_market_conditions(observation))
        if len(self.market_memory) > 50:
            self.market_memory = self.market_memory[-50:]
        
        return action

    def observe_kaiser_info(self, kaiser_id: str, kaiser_state: dict):
        """Called when Kaiser info is broadcast after bbox evaluation"""
        self.kaiser_id = kaiser_id
        # Extract relevant Kaiser information for targeting
        kaiser_info = {
            'agent_id': kaiser_id,
            'total_value': kaiser_state.get('total_value', 0),
            'cash': kaiser_state.get('cash', 0),
            'holdings': kaiser_state.get('holdings', {}),
            'last_action': kaiser_state.get('last_action', {}),
            'trend': self._calculate_kaiser_trend(kaiser_state),
            'position_bias': self._calculate_position_bias(kaiser_state)
        }
        self.kaiser_history.append(kaiser_info)
        if len(self.kaiser_history) > 30:
            self.kaiser_history = self.kaiser_history[-30:]

    def _calculate_kaiser_trend(self, kaiser_state: dict) -> float:
        """Calculate Kaiser's recent performance trend"""
        if len(self.kaiser_history) < 2:
            return 0.0
        recent_values = [k['total_value'] for k in self.kaiser_history[-5:]]
        if len(recent_values) < 2:
            return 0.0
        return (recent_values[-1] - recent_values[0]) / max(recent_values[0], 1.0)

    def _calculate_position_bias(self, kaiser_state: dict) -> float:
        """Calculate Kaiser's position bias (positive = net long, negative = net short)"""
        holdings = kaiser_state.get('holdings', {})
        current_prices = kaiser_state.get('current_prices', {})
        total_position = 0.0
        for asset, qty in holdings.items():
            price = current_prices.get(asset, 100.0)
            total_position += qty * price
        return total_position / max(kaiser_state.get('total_value', 1.0), 1.0)

    def calculate_kaiser_damage_bonus(self, kaiser_old_value: float, kaiser_new_value: float, 
                                     my_old_value: float, my_new_value: float) -> float:
        """Calculate bonus for damaging Kaiser relative to own performance"""
        kaiser_loss = (kaiser_old_value - kaiser_new_value) / max(kaiser_old_value, 1.0)
        my_gain = (my_new_value - my_old_value) / max(my_old_value, 1.0)
        
        # Bonus is higher if Kaiser lost more than I gained (pure damage)
        # or if I gained while Kaiser lost (profit + damage)
        if kaiser_loss > 0:
            if my_gain > 0:
                # Both: I gained, Kaiser lost - best case
                base_bonus = kaiser_loss * 2.0 + my_gain
            else:
                # Only Kaiser lost - still good
                base_bonus = kaiser_loss * 1.5
        else:
            # Kaiser gained - no bonus
            base_bonus = 0.0
        
        # Scale by aggression style
        if self.aggression_style == 'front_run':
            multiplier = 1.5
        elif self.aggression_style == 'liquidity_trap':
            multiplier = 1.3
        elif self.aggression_style == 'momentum_reverse':
            multiplier = 1.2
        else:  # volatility_spike
            multiplier = 1.1
        
        return base_bonus * multiplier * self.risk_preference

    def consume_pending_event(self):
        event_id, self.pending_event_id = self.pending_event_id, None
        return event_id

    def record_event_trigger(self, event_id, event_name, market_damage):
        self.triggered_events.append(dict(event_id=event_id, event_name=event_name, market_damage=market_damage))

    def earn_villain_points(self, points):
        self.state.villain_points += points

    def train_offline(self, batch_size=32, epochs=2):
        # This is an on-policy learner. Replaying old policy decisions is invalid.
        return

    def reset(self, initial_cash=None):
        super().reset(initial_cash)
        self.state.is_bad = True
        self.state.villain_points = self.initial_villain_points
        self.pending_event_id = None
        self.triggered_events.clear()

    def get_manipulation_metrics(self):
        return dict(villain_points=self.state.villain_points, events_triggered=len(self.triggered_events),
                    policy_updates=self.policy.updates)
