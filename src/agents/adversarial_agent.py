from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, List
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from datetime import datetime
from .base_agent import BaseAgent, AgentTraits, AgentType
from src.utils.logging import JSONLogger, get_logger, LogCategory

MAX_TRIGGERABLE_EVENTS = 32
MODE_NORMAL, MODE_MANIPULATE, MODE_EVENT = 0, 1, 2

@dataclass
class ManipulationStrategy:
    strategy_type: str = "pump_and_dump"
    target_price: Optional[float] = None
    duration: int = 10
    intensity: float = 0.5
    cooldown: int = 20

    def execute_strat(self, curr_p: float, cash: float, holdings: int, asset_idx: int = 0) -> Tuple[int, int, int, float]:
        if self.strategy_type == "pump_and_dump":
            return self._pump_and_dump(curr_p, cash, holdings, asset_idx)
        if self.strategy_type == "bear_raid":
            return self._bear_raid(curr_p, cash, holdings, asset_idx)
        if self.strategy_type == "spoofing":
            return self._spoofing(curr_p, cash, holdings, asset_idx)
        if self.strategy_type == "layering":
            return self._layering(curr_p, cash, holdings, asset_idx)
        return self._random_manipulation(curr_p, cash, holdings, asset_idx)

    def _pump_and_dump(self, current_price: float, agent_cash: float, agent_holdings: int, asset_idx: int = 0) -> Tuple[int, int, int, float]:
        if agent_holdings < 100:
            return (1, asset_idx, min(50, max(1, int(agent_cash / max(current_price, 1e-5)))), current_price * 1.01)
        return (2, asset_idx, max(1, agent_holdings), current_price * 0.99)

    def _bear_raid(self, current_price: float, agent_cash: float, agent_holdings: int, asset_idx: int = 0) -> Tuple[int, int, int, float]:
        if agent_holdings >= 0:
            return (2, asset_idx, min(30, max(1, int(agent_cash / max(current_price, 1e-5)))), current_price * 0.98)
        return (1, asset_idx, max(1, abs(agent_holdings)), current_price * 0.99)

    def _spoofing(self, current_price: float, _agent_cash: float, _agent_holdings: int, asset_idx: int = 0) -> Tuple[int, int, int, float]:
        qty = 100
        return (3, asset_idx, qty, current_price * 1.02) if random.random() < 0.5 else (4, asset_idx, qty, current_price * 0.98)

    def _layering(self, current_price: float, _agent_cash: float, _agent_holdings: int, asset_idx: int = 0) -> Tuple[int, int, int, float]:
        qty, level = 20, random.randint(2, 6)
        return (3, asset_idx, qty, current_price * (1 - level * 0.005)) if random.random() < 0.5 else (4, asset_idx, qty, current_price * (1 + level * 0.005))

    def _random_manipulation(self, current_price: float, _agent_cash: float, _agent_holdings: int, asset_idx: int = 0) -> Tuple[int, int, int, float]:
        vol, vol2 = random.uniform(0.5, 1.5), random.uniform(0.5, 1.5) * 1.15
        price = current_price * random.uniform(0.95 * vol, 1.05 * vol2)
        return (random.choice([1, 2, 3, 4]), asset_idx, random.randint(10, 50), price)


class AdversarialPolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_strategies: int = 4, max_events: int = MAX_TRIGGERABLE_EVENTS):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.mode_head = nn.Linear(64, 3)
        self.strategy_head = nn.Linear(64, n_strategies)
        self.event_head = nn.Linear(64, max_events)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        return self.mode_head(h), self.strategy_head(h), self.event_head(h)


class PatternMemory:
    def __init__(self, decay: float = 0.92):
        self.decay = decay
        self.stats: Dict[Tuple[int, int], Dict[str, float]] = {}

    def pattern_key(self, observation: Dict[str, Any]) -> Tuple[int, int]:
        price_history = observation.get('price_history', [])
        if isinstance(price_history, np.ndarray) and price_history.ndim == 2 and price_history.shape[0] >= 5:
            col = 0
            p = price_history[-5:, col]
            rets = np.diff(p) / np.maximum(p[:-1], 1e-5)
            vol = float(np.std(rets))
            trend = float(np.mean(rets))
        else:
            vol, trend = 0.01, 0.0
        return (min(9, int(vol * 200)), min(9, int((trend + 0.05) * 100)))

    def update(self, key: Tuple[int, int], strategy: str, unpredictability: float, profit: float):
        bucket = self.stats.setdefault(key, {})
        old_u = bucket.get(f"{strategy}_u", 0.5)
        old_p = bucket.get(f"{strategy}_p", 0.0)
        bucket[f"{strategy}_u"] = old_u * self.decay + unpredictability * (1.0 - self.decay)
        bucket[f"{strategy}_p"] = old_p * self.decay + profit * (1.0 - self.decay)

    def best_strategy(self, key: Tuple[int, int], strategies: List[ManipulationStrategy], unpred_w: float = 0.65) -> ManipulationStrategy:
        bucket = self.stats.get(key, {})
        best, best_score = strategies[0], -1e9
        for strat in strategies:
            u = bucket.get(f"{strat.strategy_type}_u", 0.5)
            p = bucket.get(f"{strat.strategy_type}_p", 0.0)
            score = unpred_w * u + (1.0 - unpred_w) * p
            if score > best_score:
                best, best_score = strat, score
        return best


class AdversarialAgent(BaseAgent):
    def __init__(self, agent_id: str, logger: Optional[JSONLogger] = None, initial_cash: float = 1000.0, traits: Optional[AgentTraits] = None, villain_points: int = 10, unpredictability_weight: float = 0.65):
        traits = self._create_adversarial_traits() if traits is None else traits
        super().__init__(agent_id=agent_id, agent_type=AgentType.ADVERSARIAL, initial_cash=initial_cash, traits=traits)
        self.logger = logger or get_logger()
        self.state.is_bad = True
        self.state.villain_points = villain_points
        self.unpredictability_weight = unpredictability_weight
        self.profit_weight = 1.0 - unpredictability_weight
        self.manipulation_strategies = [ManipulationStrategy("pump_and_dump"), ManipulationStrategy("bear_raid"), ManipulationStrategy("spoofing"), ManipulationStrategy("layering")]
        self.current_strategy: Optional[ManipulationStrategy] = None
        self.strategy_cooldown = 0
        self.triggered_events: List[Dict] = []
        self.event_trigger_cost = 5
        self.event_manager = None
        self.pending_event_id: Optional[int] = None
        self.pattern_memory = PatternMemory()
        self.action_entropy_history: deque = deque(maxlen=20)
        self.last_mode = MODE_MANIPULATE
        self.obs_dim = 48
        self.policy = AdversarialPolicyNet(self.obs_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.replay: deque = deque(maxlen=512)
        self.training_enabled = True
        self.exploration_rate = 0.15
        self.last_policy_meta: Optional[Dict[str, Any]] = None
        self._last_pattern_key: Optional[Tuple[int, int]] = None

    def bind_event_manager(self, event_manager: Any):
        self.event_manager = event_manager

    def _create_adversarial_traits(self) -> AgentTraits:
        traits = AgentTraits()
        traits.risk_tolerance, traits.loss_aversion = 0.8, 0.3
        traits.patience, traits.holding_period_preference = 0.2, 0.1
        traits.contrarian_bias, traits.trend_following = 0.9, 0.1
        traits.herd_behavior, traits.confidence_level = 0.1, 0.8
        traits.adaptability, traits.exploration_rate = 0.9, 0.8
        traits.order_size_preference, traits.limit_order_preference = 0.9, 0.3
        return traits

    def consume_pending_event(self) -> Optional[int]:
        eid, self.pending_event_id = self.pending_event_id, None
        return eid

    def decide_action(self, observation: Dict[str, Any], action_space: Any) -> np.ndarray:
        processed_obs = self.process_observation(observation)
        self._last_pattern_key = self.pattern_memory.pattern_key(processed_obs)
        mode, strat_idx, event_slot = self._select_rl_decision(processed_obs)
        self.last_mode = mode
        if mode == MODE_EVENT:
            self._queue_event_trigger(processed_obs, event_slot)
        if self.strategy_cooldown > 0:
            self.strategy_cooldown -= 1
        elif mode == MODE_MANIPULATE or self.current_strategy is None:
            self.current_strategy = self.manipulation_strategies[strat_idx]
            self.strategy_cooldown = self.current_strategy.cooldown
        asset_idx = self._pick_asset_idx(processed_obs)
        current_price = self._asset_price(processed_obs, asset_idx)
        asset_name = self._asset_name(processed_obs, asset_idx)
        agent_cash, agent_holdings = self.state.cash, self.state.holdings.get(asset_name, 0)
        if mode == MODE_NORMAL:
            action_type, quantity, price = self._normal_trade(processed_obs, asset_idx, current_price, agent_cash, agent_holdings)
        else:
            action_type, _, quantity, price = self.current_strategy.execute_strat(current_price, agent_cash, agent_holdings, asset_idx) if self.current_strategy else (0, asset_idx, 1, current_price)
        base_action = np.array([action_type, asset_idx, quantity, price], dtype=np.float32)
        modified_action = self.apply_trait_modifications(base_action)
        self.action_entropy_history.append(int(modified_action[0]))
        self.state.last_action = {'type': int(modified_action[0]), 'asset_idx': int(modified_action[1]), 'quantity': int(modified_action[2]), 'price': float(modified_action[3]), 'strategy': self.current_strategy.strategy_type if self.current_strategy else None, 'mode': mode, 'event_slot': event_slot}
        return modified_action

    def _pick_asset_idx(self, observation: Dict[str, Any]) -> int:
        available_assets = observation.get('available_assets', [1.0] + [0.0] * 9)
        return int(np.argmax(available_assets)) if isinstance(available_assets, np.ndarray) else 0

    def _asset_price(self, observation: Dict[str, Any], asset_idx: int) -> float:
        current_prices = observation.get('current_price', [100.0] * 10)
        if isinstance(current_prices, (list, np.ndarray)) and len(current_prices) > asset_idx:
            return float(current_prices[asset_idx])
        return 100.0

    def _asset_name(self, observation: Dict[str, Any], asset_idx: int) -> str:
        names = observation.get('asset_names', [])
        if isinstance(names, list) and asset_idx < len(names):
            return str(names[asset_idx])
        return f"ASSET{asset_idx}"

    def _normal_trade(self, observation: Dict[str, Any], asset_idx: int, current_price: float, agent_cash: float, agent_holdings: int) -> Tuple[int, int, float]:
        modifiers = self.traits.get_behavior_modifiers()
        price_history = observation.get('price_history', [])
        action_type = 0
        if isinstance(price_history, np.ndarray) and price_history.ndim == 2 and price_history.shape[0] >= 5:
            p_slice = price_history[-5:, min(asset_idx, price_history.shape[1] - 1)]
            recent_trend = (p_slice[-1] - p_slice[0]) / max(1e-5, p_slice[0])
            if recent_trend > 0.01 and modifiers['trend_modifier'] > 0.55:
                action_type = 1
            elif recent_trend < -0.01 and modifiers['trend_modifier'] < 0.45:
                action_type = 2
        quantity = max(1, min(80, int(10 + modifiers['risk_modifier'] * 40)))
        price = max(1.0, current_price * (1 + random.uniform(-0.015, 0.015)))
        if modifiers['patience_modifier'] > 0.7 and action_type in (1, 2):
            action_type += 2
        return action_type, quantity, price

    def _obs_vector(self, observation: Dict[str, Any]) -> np.ndarray:
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        cp = observation.get('current_price', [100.0] * 10)
        if isinstance(cp, np.ndarray):
            vec[:10] = cp[:10]
        ph = observation.get('price_history', [])
        if isinstance(ph, np.ndarray) and ph.ndim == 2:
            vec[10:20] = ph[-1, :10] if ph.shape[0] else 0.0
            if ph.shape[0] >= 2:
                vec[20:30] = (ph[-1, :10] - ph[-2, :10]) / np.maximum(ph[-2, :10], 1e-5)
        vec[30] = float(observation.get('agent_cash', self.state.cash)) / max(1.0, float(observation.get('agent_total_value', self.state.total_value)))
        vec[31] = self.state.villain_points / 100.0
        ev_mask = observation.get('triggerable_event_mask', np.zeros(MAX_TRIGGERABLE_EVENTS))
        if isinstance(ev_mask, np.ndarray):
            vec[32:40] = ev_mask[:8]
        active = observation.get('active_events', np.zeros(10))
        if isinstance(active, np.ndarray):
            vec[40:45] = active[:5]
        vec[45] = len(self.action_entropy_history) / 20.0
        if len(self.action_entropy_history) >= 3:
            counts = np.bincount(list(self.action_entropy_history), minlength=5)
            probs = counts / max(1, counts.sum())
            vec[46] = float(-np.sum(probs * np.log(probs + 1e-8)) / np.log(5))
        vec[47] = self.unpredictability_weight
        return vec

    def _select_rl_decision(self, observation: Dict[str, Any]) -> Tuple[int, int, int]:
        obs_vec = self._obs_vector(observation)
        if random.random() < self.exploration_rate:
            mode = random.choice([MODE_NORMAL, MODE_MANIPULATE, MODE_EVENT])
            return mode, random.randint(0, len(self.manipulation_strategies) - 1), random.randint(0, MAX_TRIGGERABLE_EVENTS - 1)
        with torch.no_grad():
            logits_mode, logits_strat, logits_event = self.policy(torch.FloatTensor(obs_vec).unsqueeze(0))
        mode = int(torch.argmax(logits_mode, dim=-1).item())
        strat_idx = int(torch.argmax(logits_strat, dim=-1).item())
        event_mask = observation.get('triggerable_event_mask', np.zeros(MAX_TRIGGERABLE_EVENTS, dtype=np.float32))
        event_logits = logits_event.squeeze(0).numpy()
        if isinstance(event_mask, np.ndarray):
            event_logits = event_logits - (1.0 - event_mask) * 1e6
        event_slot = int(np.argmax(event_logits))
        if self._last_pattern_key is not None and mode == MODE_MANIPULATE and random.random() > self.exploration_rate:
            learned = self.pattern_memory.best_strategy(self._last_pattern_key, self.manipulation_strategies, self.unpredictability_weight)
            for i, s in enumerate(self.manipulation_strategies):
                if s.strategy_type == learned.strategy_type:
                    strat_idx = i
                    break
        self.last_policy_meta = {'obs': obs_vec, 'mode': mode, 'strat_idx': strat_idx, 'event_slot': event_slot, 'logits_mode': logits_mode.squeeze(0), 'logits_strat': logits_strat.squeeze(0), 'logits_event': logits_event.squeeze(0)}
        return mode, strat_idx, event_slot

    def _queue_event_trigger(self, observation: Dict[str, Any], event_slot: int):
        if self.state.villain_points < self.event_trigger_cost:
            return
        event_ids = observation.get('triggerable_event_ids', [])
        if not isinstance(event_ids, (list, np.ndarray)) or len(event_ids) == 0:
            return
        slot = min(event_slot, len(event_ids) - 1)
        self.pending_event_id = int(event_ids[slot])
        self.state.villain_points -= self.event_trigger_cost

    def record_event_trigger(self, event_id: int, event_name: str, market_damage: float):
        event_info = {'type': 'market_event', 'event_id': event_id, 'event_name': event_name, 'triggered_by': self.agent_id, 'timestamp': len(self.state.profit_history), 'market_damage': market_damage}
        self.triggered_events.append(event_info)
        self.state.add_memory(event_info)
        self.logger.info(LogCategory.EVENT, f"Adversarial agent {self.agent_id} triggered {event_name}", data=event_info)

    def earn_villain_points(self, points: int):
        self.state.villain_points += points

    def _policy_update(self, reward: float):
        if not self.training_enabled or self.last_policy_meta is None:
            return
        meta = self.last_policy_meta
        obs_t = torch.FloatTensor(meta['obs']).unsqueeze(0)
        logits_mode, logits_strat, logits_event = self.policy(obs_t)
        mode_t = torch.tensor([meta['mode']], dtype=torch.long)
        strat_t = torch.tensor([meta['strat_idx']], dtype=torch.long)
        event_t = torch.tensor([meta['event_slot']], dtype=torch.long)
        loss = F.cross_entropy(logits_mode, mode_t) + F.cross_entropy(logits_strat, strat_t) + F.cross_entropy(logits_event, event_t)
        self.optimizer.zero_grad()
        (loss * reward).backward()
        self.optimizer.step()
        self.replay.append((meta['obs'], meta['mode'], meta['strat_idx'], meta['event_slot'], reward))

    def update_state(self, reward: float, done: bool, info: Dict):
        unpred = float(info.get('unpredictability_score', 0.0))
        profit = float(info.get('profit_component', reward))
        shaped = self.unpredictability_weight * unpred + self.profit_weight * profit
        if self._last_pattern_key is not None and self.state.last_action:
            strat = self.state.last_action.get('strategy') or 'normal'
            self.pattern_memory.update(self._last_pattern_key, strat, unpred, profit)
        self._policy_update(shaped)
        super().update_state(shaped, done, info)
        if info.get('market_damage', 0) > 0:
            self.earn_villain_points(int(info['market_damage'] * 0.1))
        if info.get('max_bad_agents_reached', False):
            self.state.is_blocked = True

    def train_offline(self, batch_size: int = 32, epochs: int = 2):
        if len(self.replay) < batch_size:
            return
        for _ in range(epochs):
            batch = random.sample(list(self.replay), batch_size)
            loss_total = 0.0
            for obs, mode, strat, event, rw in batch:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                logits_mode, logits_strat, logits_event = self.policy(obs_t)
                loss = F.cross_entropy(logits_mode, torch.tensor([mode])) + F.cross_entropy(logits_strat, torch.tensor([strat])) + F.cross_entropy(logits_event, torch.tensor([event]))
                loss_total += float((loss * rw).item())
                self.optimizer.zero_grad()
                (loss * rw).backward()
                self.optimizer.step()

    def save_model(self, path: str):
        torch.save({'policy': self.policy.state_dict(), 'optimizer': self.optimizer.state_dict(), 'pattern_stats': self.pattern_memory.stats}, path)

    def load_model(self, path: str, env: Any = None):
        try:
            ckpt = torch.load(path, map_location='cpu')
            self.policy.load_state_dict(ckpt['policy'])
            if 'optimizer' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            if 'pattern_stats' in ckpt:
                self.pattern_memory.stats = ckpt['pattern_stats']
        except Exception as e:
            self.logger.warning(LogCategory.AGENT, f"Failed to load adversarial policy for {self.agent_id}", data={'error': str(e)})

    def get_manipulation_metrics(self) -> Dict[str, Any]:
        return {'villain_points': self.state.villain_points, 'events_triggered': len(self.triggered_events), 'current_strategy': self.current_strategy.strategy_type if self.current_strategy else None, 'strategy_cooldown': self.strategy_cooldown, 'total_manipulation_impact': sum(e.get('market_damage', 0) for e in self.triggered_events), 'pattern_buckets': len(self.pattern_memory.stats), 'replay_size': len(self.replay)}

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(self.get_manipulation_metrics())
        info.update({'unpredictability_weight': self.unpredictability_weight, 'exploration_rate': self.exploration_rate})
        return info

    def generate_prediction(self, observation: Dict[str, Any], horizon: int = 5) -> Optional[Dict]:
        from src.simulation.black_box import Prediction, PredictionType
        price_history = observation.get('price_history', [])
        if not isinstance(price_history, np.ndarray) or price_history.ndim != 2 or price_history.shape[0] < 10:
            return None
        asset_idx = 0
        recent_prices = price_history[-10:, asset_idx]
        if len(recent_prices) < 2:
            return None
        returns = np.diff(recent_prices) / np.maximum(recent_prices[:-1], 1e-5)
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        last_price = float(recent_prices[-1])
        predicted_prices = []
        current_p = last_price
        manipulation_bias = 0.02 if self.current_strategy else 0.0
        for _ in range(horizon):
            change = np.random.normal(mean_return + manipulation_bias, std_return * 1.2)
            current_p = max(1.0, current_p * (1 + change))
            predicted_prices.append(current_p)
        conf_level = min(0.95, max(0.3, 0.6 - std_return * 8))
        price_std = std_return * last_price * 1.5
        conf_interval = (min(predicted_prices) - price_std, max(predicted_prices) + price_std)
        prediction = Prediction(
            agent_id=self.agent_id,
            pred_t=PredictionType.PRICE_LEVEL,
            pred_val=predicted_prices,
            conf_interv=conf_interval,
            conf_lvl=conf_level
        )
        return prediction

    def reset(self, initial_cash: Optional[float] = None):
        super().reset(initial_cash)
        self.current_strategy, self.pending_event_id = None, None
        self.strategy_cooldown = 0
        self.triggered_events = []
        self.last_policy_meta = None
