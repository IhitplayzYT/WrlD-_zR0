"""Budgeted adversarial minority: trades plus simulated event interventions."""
import numpy as np
from .base_agent import AgentType, AgentTraits
from .rlaif_agent import RLAIFAgent

MAX_TRIGGERABLE_EVENTS = 32

class AdversarialAgent(RLAIFAgent):
    def __init__(self, agent_id, logger=None, initial_cash=1000., traits=None,villain_points=10, unpredictability_weight=.65):
        traits = traits or AgentTraits(risk_tolerance=.8, max_position_size=.3)
        super().__init__(agent_id, logger, initial_cash, traits, extra_actions=MAX_TRIGGERABLE_EVENTS)
        self.agent_type = self.state.agent_type = AgentType.ADVERSARIAL
        self.initial_villain_points = villain_points
        self.state.is_bad = True
        self.state.villain_points = villain_points
        self.event_trigger_cost = 5
        self.pending_event_id = None
        self.triggered_events = []
        self.event_manager = None
        self.market_memory = []
        self.triggering_strategy = 'adaptive'  # 'adaptive', 'aggressive', 'conservative'
        self.kaiser_id = None
        self.kaiser_history = [] 
        self.kaiser_damage_bonus = 0.0
        self.aggression_style = np.random.choice(['front_run', 'liquidity_trap', 'momentum_reverse', 'volatility_spike'])
        self.risk_preference = np.random.uniform(0.3, 0.9)
        self.patience_factor = np.random.uniform(0.5, 2.0) # Higher = more patient
        self.adaptation_rate = np.random.uniform(0.1, 0.5) 
        self.decision_volatility = np.random.uniform(0.1, 0.3)
        self.event_selection_noise = np.random.uniform(0.05, 0.15)
        self.event_choice = 'prior'      # 'prior' | 'learned' | 'heuristic' (see decide_action)
        self.event_prior_strength = 0.5  # logit bias per scoring point; the policy can learn to overrule it
        self.kaiser_bonus_scale = 0.3    # puts the Kaiser term on par with ordinary disturbance (measured 1.5 vs 0.44 per episode at scale 1)

    def bind_event_manager(self, event_manager): self.event_manager = event_manager

    def _evaluate_market_conditions(self, obs):
        current_price = float(obs['current_price'][0])
        price_history = np.asarray(obs['price_history'])[:, 0]
        price_history = price_history[price_history > 0]
        if len(price_history) < 5: return {'volatility': 0.01, 'trend': 0, 'liquidity_stress': 0.5}
        returns = np.diff(np.log(price_history[-20:]))
        volatility = float(np.std(returns)) if len(returns) > 0 else 0.01
        trend = float(np.mean(returns)) if len(returns) > 0 else 0
        bids = obs.get('order_book_bids', np.zeros((10, 10, 2)))
        asks = obs.get('order_book_asks', np.zeros((10, 10, 2)))
        bid_depth = np.sum(bids[:, :, 1]) if bids.ndim == 3 else 0
        ask_depth = np.sum(asks[:, :, 1]) if asks.ndim == 3 else 0
        total_depth = bid_depth + ask_depth
        liquidity_stress = 1.0 - min(1.0, total_depth / 10000.0) 
        
        active_events = obs.get('active_events', np.zeros(10))
        event_pressure = float(np.sum(active_events)) / 10.0
        
        return {'volatility': volatility,'trend': trend,'liquidity_stress': liquidity_stress,'event_pressure': event_pressure,'price': current_price}

    def _event_scores(self, obs, triggerable_ids, triggerable_mask):
        scores = [None] * len(triggerable_ids)
        if not self.event_manager: return scores
        conditions = self._evaluate_market_conditions(obs)
        for slot, (event_id, is_triggerable) in enumerate(zip(triggerable_ids, triggerable_mask)):
            if not is_triggerable or event_id < 0: continue
            event = self.event_manager.get_event(event_id)
            if not event: continue
            score = 0.0
            effects = event.effects
            if conditions['volatility'] > 0.02 and effects.volatility_multiplier > 1.5: score += 2.0
            if conditions['liquidity_stress'] > 0.7 and effects.liquidity_multiplier < 0.8: score += 2.5
            if conditions['trend'] < -0.01 and effects.price_drift < -0.02: score += 1.5
            if conditions['trend'] > 0.01 and effects.price_drift > 0.02: score += 1.0
            if conditions['event_pressure'] > 0.3 and abs(effects.price_drift) > 0.05: score += 1.0
            if self.kaiser_history:
                kaiser_info = self.kaiser_history[-1]
                kaiser_trend = kaiser_info.get('trend', 0)
                kaiser_position = kaiser_info.get('position_bias', 0)  # > 0 net long, < 0 (near) flat
                if self.aggression_style == 'front_run':
                    if kaiser_position > 0 and effects.price_drift < -0.03: score += 3.0  # crash while the Kaiser is long
                    elif kaiser_position < 0 and effects.price_drift > 0.03: score += 3.0  # pump while the Kaiser is flat (long-only market: the 'short' case)
                elif self.aggression_style == 'liquidity_trap':
                    if effects.liquidity_multiplier < 0.5: score += 2.5
                elif self.aggression_style == 'momentum_reverse':
                    if kaiser_trend > 0 and effects.price_drift < -0.04: score += 2.0
                    elif kaiser_trend < 0 and effects.price_drift > 0.04: score += 2.0
                elif self.aggression_style == 'volatility_spike':
                    if effects.volatility_multiplier > 2.0: score += 2.0
            if self.risk_preference > 0.7:
                if abs(effects.price_drift) > 0.05: score += 1.5
            elif abs(effects.price_drift) < 0.03: score += 1.0
            score += np.random.uniform(0, 0.5 * (2.0 - self.patience_factor))
            score += np.random.normal(0, self.event_selection_noise * abs(score))
            scores[slot] = float(score)
        return scores

    # Argmax event selection
    def _select_smart_event(self, obs, triggerable_ids, triggerable_mask):
        scores = self._event_scores(obs, triggerable_ids, triggerable_mask)
        best = [(sc, i) for i, sc in enumerate(scores) if sc is not None]
        return int(triggerable_ids[max(best)[1]]) if best else None

    def _choices(self, obs):
        choices, mask = super()._choices(obs)
        event_mask = obs.get('triggerable_event_mask', np.zeros(MAX_TRIGGERABLE_EVENTS))
        for i in range(MAX_TRIGGERABLE_EVENTS):
            choices.append(np.zeros(4, dtype=np.float32))
            mask.append(bool(event_mask[i]) and self.state.villain_points >= self.event_trigger_cost and bool(obs.get('can_trade', True)))
        return choices, mask

    def decide_action(self, observation, action_space=None):
        """Event choice modes (``event_choice``):
        'prior' use score guidance 
        'learned' No score guidance 
        'heuristic' Scoring fn decides
        """
        choices, mask = self._choices(observation)
        triggerable_ids = list(observation.get('triggerable_event_ids', [-1] * MAX_TRIGGERABLE_EVENTS))
        triggerable_mask = observation.get('triggerable_event_mask', np.zeros(MAX_TRIGGERABLE_EVENTS))
        bias = None
        if self.event_choice == 'prior':
            scores = self._event_scores(observation, triggerable_ids, triggerable_mask)
            bias = np.zeros(len(choices), dtype=np.float32)
            for slot, sc in enumerate(scores):
                if sc is not None and 41 + slot < len(bias): bias[41 + slot] = self.event_prior_strength * sc

        choice = self.policy.select(observation, np.asarray(mask, dtype=bool), self.training_enabled, bias=bias)
        self.pending_event_id = None
        if choice >= 41:
            if self.event_choice == 'heuristic':
                smart_event_id = self._select_smart_event(observation, triggerable_ids, triggerable_mask)
                self.pending_event_id = smart_event_id if smart_event_id is not None else int(triggerable_ids[choice - 41])
            else: self.pending_event_id = int(triggerable_ids[choice - 41])
        
        action = choices[choice]
        self.state.last_action = dict(type=int(action[0]), asset_idx=int(action[1]),quantity=int(action[2]), price=float(action[3]), event_id=self.pending_event_id)
        self.action_history.append(action.copy())
        self.action_history = self.action_history[-50:]
        self.market_memory.append(self._evaluate_market_conditions(observation))
        if len(self.market_memory) > 50: self.market_memory = self.market_memory[-50:]
        return action

    def observe_kaiser_info(self, kaiser_id: str, kaiser_state: dict):
        self.kaiser_id = kaiser_id
        kaiser_info = {'agent_id': kaiser_id,'total_value': kaiser_state.get('total_value', 0),'cash': kaiser_state.get('cash', 0),'holdings': kaiser_state.get('holdings', {}),'last_action': kaiser_state.get('last_action', {}),'trend': self._calculate_kaiser_trend(kaiser_state),'position_bias': self._calculate_position_bias(kaiser_state)}
        self.kaiser_history.append(kaiser_info)
        if len(self.kaiser_history) > 30: self.kaiser_history = self.kaiser_history[-30:]

    def _calculate_kaiser_trend(self, kaiser_state: dict) -> float:
        """Relative change of the Kaiser wealth btwn 5 broadcast"""
        values = [k['total_value'] for k in self.kaiser_history[-4:]] + [kaiser_state.get('total_value', 0)]
        if len(values) < 2: return 0.0
        return (values[-1] - values[0]) / max(values[0], 1.0)

    def _calculate_position_bias(self, kaiser_state: dict) -> float:
        holdings = kaiser_state.get('holdings', {})
        current_prices = kaiser_state.get('current_prices', {})
        total_position = 0.0
        for asset, qty in holdings.items():
            total_position += qty * current_prices.get(asset, 100.0)
        # Long-only market: exposure is never negative, so centre it; below 10% invested counts as flat/'short'.
        return total_position / max(kaiser_state.get('total_value', 1.0), 1.0) - 0.10

    def calculate_kaiser_damage_bonus(self, kaiser_old_value: float, kaiser_new_value: float, my_old_value: float, my_new_value: float) -> float:
        kaiser_loss = (kaiser_old_value - kaiser_new_value) / max(kaiser_old_value, 1.0)
        my_gain = (my_new_value - my_old_value) / max(my_old_value, 1.0)
        if kaiser_loss > 0:
            if my_gain > 0:base_bonus = kaiser_loss * 2.0 + my_gain
            else:base_bonus = kaiser_loss * 1.5
        else:
            base_bonus = 0.0
        if self.aggression_style == 'front_run': multiplier = 1.5
        elif self.aggression_style == 'liquidity_trap': multiplier = 1.3
        elif self.aggression_style == 'momentum_reverse': multiplier = 1.2
        else: multiplier = 1.1
        
        return self.kaiser_bonus_scale * base_bonus * multiplier * self.risk_preference / 0.01

    def consume_pending_event(self):
        event_id, self.pending_event_id = self.pending_event_id, None
        return event_id

    def record_event_trigger(self, event_id, event_name, market_damage, step=None): self.triggered_events.append(dict(event_id=event_id, event_name=event_name, market_damage=market_damage, step=step))

    def earn_villain_points(self, points): self.state.villain_points += points

    def train_offline(self, batch_size=32, epochs=2): return

    def reset(self, initial_cash=None):
        super().reset(initial_cash)
        self.state.is_bad = True
        self.state.villain_points = self.initial_villain_points
        self.pending_event_id = None
        self.triggered_events.clear()
        self.kaiser_id = None
        self.kaiser_history.clear()
        self.market_memory.clear()

    TRAIT_FIELDS = RLAIFAgent.TRAIT_FIELDS + ('aggression_style', 'risk_preference', 'patience_factor', 'adaptation_rate','event_selection_noise', 'event_prior_strength', 'kaiser_bonus_scale')

    def profile(self):
        return dict(aggression_style=str(self.aggression_style), risk_preference=float(self.risk_preference),patience_factor=float(self.patience_factor), adaptation_rate=float(self.adaptation_rate),decision_volatility=float(self.decision_volatility), event_selection_noise=float(self.event_selection_noise))

    def get_manipulation_metrics(self):
        return dict(villain_points=self.state.villain_points, events_triggered=len(self.triggered_events),policy_updates=self.policy.updates)
