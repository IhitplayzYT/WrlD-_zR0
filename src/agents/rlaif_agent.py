import numpy as np
import torch
from src.utils.logging import get_logger
from .base_agent import BaseAgent, AgentTraits, AgentType
from .learning import ActorCritic
from .forecasting import ProbabilisticForecastRNN
from src.models.preference_reward import PreferenceReward
from src.simulation.black_box import Prediction, PredictionType

class RLAIFAgent(BaseAgent):
    """ Independent A2C trader with a synthetic pairwise preference teacher """

    def __init__(self, agent_id, logger=None, initial_cash=1000., traits=None,rl_algorithm='A2C', curriculum_level=0, extra_actions=0):
        super().__init__(agent_id, AgentType.RLAIF, initial_cash, traits)
        self.logger = logger or get_logger()
        self.rl_algorithm = 'A2C'
        self.curriculum_level = curriculum_level
        self.policy = ActorCritic(41 + extra_actions)
        self.preference_model = PreferenceReward()
        self.training_enabled = True
        self.training_step = 0
        self.action_history = []
        self.episode_rewards = []
        self.prediction_volatility = np.random.uniform(0.5, 2.0) 
        self.high_volatility_prob = np.random.uniform(0.05, 0.15) 
        self.kaiser_influence_susceptibility = np.random.uniform(0.1, 0.3)
        self.kaiser_action_memory = []
        self.decision_volatility = np.random.uniform(0.05, 0.2)
        # Off by default: noise applied after sampling means the executed action is not the credited one.
        self.execution_noise = False
        self.forecast_lookback = int(np.random.randint(5, 61))
        # Each trader owns and trains a separate probabilistic bbox forecaster.
        self.forecaster = ProbabilisticForecastRNN(behavior_dim=len(self.traits.__dict__) + 29)
        self.forecast_training_enabled = True
        self._pending_forecast = None
        self.deciction_temperature = np.random.uniform(0.8, 1.5)  # Softmax temperature for exploration

    def _choices(self, obs):
        choices = [np.zeros(4, dtype=np.float32)]
        mask = [True]
        cash = float(obs.get('available_cash', obs['agent_cash']))
        wealth = float(obs['agent_total_value'])
        holdings = obs.get('available_holdings', obs['agent_holdings'])
        for idx in range(10):
            price = float(obs['current_price'][idx])
            active = bool(obs['available_assets'][idx]) and price > 0
            buy_room = max(0.0, wealth * self.traits.max_position_size - obs['agent_holdings'][idx] * price)
            buy_qty = min(1000, int(min(cash, buy_room, wealth * 0.1) / max(price * 1.002, 1e-8))) if active else 0
            sell_qty = min(1000, int(holdings[idx]), max(1, int(wealth * .1 / max(price, 1e-8)))) if active else 0
            for kind, qty, quote in [(1, buy_qty, price), (2, sell_qty, price),(3, buy_qty, price * .999), (4, sell_qty, price * 1.001)]:
                choices.append(np.array([kind, idx, qty, quote], dtype=np.float32))
                mask.append(active and qty > 0 and bool(obs.get('can_trade', True)))
        return choices, mask

    def decide_action(self, observation, action_space=None):
        choices, mask = self._choices(observation)
        choice = self.policy.select(observation, np.asarray(mask, dtype=bool), self.training_enabled)
        action = choices[choice]
        # Kaiser nudge
        if self.kaiser_action_memory:
            kaiser_type = self.kaiser_action_memory[-1].get('type', 0)
            current_type = int(action[0])
            wants_buy = kaiser_type in (1, 3) and current_type in (0, 2, 4)
            wants_sell = kaiser_type in (2, 4) and current_type in (0, 1, 3)
            if (wants_buy or wants_sell) and np.random.random() < self.kaiser_influence_susceptibility and np.random.random() < 0.3:
                idx = int(action[1])
                for j, candidate in enumerate(choices):
                    if int(candidate[0]) == kaiser_type and int(candidate[1]) == idx and mask[j]:
                        choice, action = j, candidate.copy()
                        self.policy.override_choice(choice)
                        break
        if self.execution_noise and action[3] > 0: action[3] = max(action[3] * 0.95, action[3] +  np.random.normal(0, self.decision_volatility * action[3]))
        if self.execution_noise and action[2] > 0: action[2] = max(1, int(action[2] +  np.random.normal(0, self.decision_volatility * action[2])))
        self.state.last_action = dict(type=int(action[0]), asset_idx=int(action[1]), quantity=int(action[2]), price=float(action[3]))
        self.action_history.append(action.copy())
        self.action_history = self.action_history[-50:]
        return action
    
    def observe_kaiser_action(self, kaiser_action: dict):
        self.kaiser_action_memory.append(kaiser_action)
        if len(self.kaiser_action_memory) > 20: self.kaiser_action_memory = self.kaiser_action_memory[-20:]

    def learn(self, reward, next_obs, terminated, truncated):
        if self.training_enabled:
            self.policy.observe(reward, next_obs, terminated, truncated)
            self.training_step += 1

    def update_state(self, reward, done, info):
        super().update_state(reward, done, info)
        self.episode_rewards.append(float(reward))
        self.episode_rewards = self.episode_rewards[-1000:]

    def generate_prediction(self, observation, horizon=5):
        prices = np.asarray(observation['price_history'], dtype=np.float64)[:, 0]
        prices = prices[prices > 0]
        if not len(prices):
            raise ValueError('The bbox forecaster needs at least one positive observed price')
        origin = float(prices[-1])
        sequence = self._forecast_market_sequence(prices)
        behavior = self._forecast_behavior_features(observation)
        mean, std = self.forecaster.predict(sequence, behavior, horizon)
        # Sampling makes the forecast probabilistic even with frozen model weights.
        sampled_returns = np.random.normal(mean, std)
        values = (origin * np.exp(np.cumsum(sampled_returns))).tolist()
        cumulative_mean = np.cumsum(mean)
        cumulative_std = np.sqrt(np.cumsum(np.square(std)))
        lower = origin * np.exp(cumulative_mean - 1.645 * cumulative_std)
        upper = origin * np.exp(cumulative_mean + 1.645 * cumulative_std)
        self._pending_forecast = (sequence, behavior, origin)
        return Prediction(
            self.agent_id,
            PredictionType.PRICE_LEVEL,
            values,
            (float(np.min(lower)), float(np.max(upper))),
            0.9,
            origin_price=origin,
            return_mean=mean.tolist(),
            return_std=std.tolist(),
        )

    @staticmethod
    def _forecast_market_sequence(prices):
        returns = np.diff(np.log(prices[-51:]))
        features = np.zeros((50, 3), dtype=np.float32)
        if len(returns):
            features[-len(returns):, 0] = np.clip(returns, -0.1, 0.1)
            for i in range(len(returns)):
                recent = returns[max(0, i - 19):i + 1]
                features[50 - len(returns) + i, 1] = np.std(recent)
                features[50 - len(returns) + i, 2] = np.mean(returns[max(0, i - 4):i + 1])
        return features

    def _forecast_behavior_features(self, observation):
        traits = np.asarray(list(self.traits.__dict__.values()), dtype=np.float32)
        wealth = max(float(observation.get('agent_total_value', 1.0)), 1e-8)
        prices = np.asarray(observation['current_price'], dtype=np.float32)
        holdings = np.asarray(observation['agent_holdings'], dtype=np.float32)
        exposure = np.zeros(10, dtype=np.float32)
        count = min(len(prices), len(holdings), 10)
        exposure[:count] = holdings[:count] * prices[:count] / wealth
        cash_fraction = float(observation.get('agent_cash', 0.0)) / wealth
        market_returns = np.diff(np.log(np.maximum(np.asarray(observation['price_history'])[:, 0], 1e-8)))
        market_features = np.asarray([
            np.mean(market_returns[-20:]) if len(market_returns) else 0.0,
            np.std(market_returns[-20:]) if len(market_returns) else 0.0,
        ], dtype=np.float32)
        actions = np.asarray(self.action_history[-50:], dtype=np.float32)
        action_features = np.zeros(4, dtype=np.float32)
        if len(actions):
            kinds = actions[:, 0].astype(int)
            action_features[:] = [
                np.mean(np.isin(kinds, (1, 3))),
                np.mean(np.isin(kinds, (2, 4))),
                np.mean(np.isin(kinds, (3, 4))),
                np.mean(kinds == 0),
            ]
        style = getattr(self, 'aggression_style', '')
        styles = ['front_run', 'liquidity_trap', 'momentum_reverse', 'volatility_spike']
        adversarial = np.asarray([
            getattr(self, 'risk_preference', 0.5),
            getattr(self, 'patience_factor', 1.0) / 2.0,
            getattr(self, 'adaptation_rate', 0.0),
            getattr(self, 'event_selection_noise', 0.0),
            *[float(style == name) for name in styles],
        ], dtype=np.float32)
        personal = np.asarray([
            self.forecast_lookback / 60.0,
            self.prediction_volatility / 2.0,
            self.high_volatility_prob,
            self.decision_volatility / 0.3,
        ], dtype=np.float32)
        return np.concatenate((traits, personal, [cash_fraction], exposure, market_features, action_features, adversarial)).astype(np.float32)

    def learn_prediction(self, actual_values):
        pending, self._pending_forecast = self._pending_forecast, None
        if pending is None or not self.training_enabled or not self.forecast_training_enabled:
            return None
        sequence, behavior, origin = pending
        actual = np.asarray(actual_values, dtype=np.float64)
        if not len(actual) or not np.isfinite(actual).all() or np.any(actual <= 0):
            return None
        path = np.concatenate(([origin], actual))
        target_returns = np.diff(np.log(path))
        return self.forecaster.train_forecast(sequence, behavior, target_returns)

    def reset(self, initial_cash=None):
        super().reset(initial_cash)
        self.action_history.clear()
        self.episode_rewards.clear()
        self.policy.pending = None
        self.policy.rollout.clear()
        self.kaiser_action_memory.clear()
        self._pending_forecast = None

    TRAIT_FIELDS = ('prediction_volatility', 'high_volatility_prob', 'kaiser_influence_susceptibility','decision_volatility', 'forecast_lookback', 'execution_noise')

    def snapshot(self): return {'policy': self.policy.snapshot(), 'preference': self.preference_model.state(), 'forecaster': self.forecaster.snapshot(), 'training_step': self.training_step, 'agent_traits': dict(self.traits.__dict__), 'traits':  {name: (v.item() if hasattr(v, 'item') else v) for name in self.TRAIT_FIELDS if hasattr(self, name) for v in [getattr(self, name)]}}

    def restore(self, state):
        self.policy.restore(state['policy'])
        self.preference_model.restore(state['preference'])
        if 'forecaster' in state: self.forecaster.restore(state['forecaster'])
        if 'agent_traits' in state:
            self.traits = AgentTraits(**state['agent_traits'])
            self.state.traits = self.traits
        self.training_step = state['training_step']
        for name, value in state.get('traits', {}).items(): setattr(self, name, value)

    def save_model(self, path): torch.save(self.snapshot(), path)
    def load_model(self, path, env=None): self.restore(torch.load(path, map_location='cpu', weights_only=True))
    def get_info(self): return {**super().get_info(), 'rl_algorithm': self.rl_algorithm,'training_step': self.training_step, 'policy_updates': self.policy.updates,'preference_updates': self.preference_model.updates, 'forecast_updates': self.forecaster.updates, 'forecast_mean_loss': self.forecaster.loss_sum / max(1, self.forecaster.updates), 'forecast_last_loss': self.forecaster.last_loss}
