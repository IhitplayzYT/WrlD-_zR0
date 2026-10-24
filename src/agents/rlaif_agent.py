"""Trainable trader with synthetic preference feedback."""
import numpy as np
import torch
from src.utils.logging import get_logger
from .base_agent import BaseAgent, AgentType
from .learning import ActorCritic
from src.models.preference_reward import PreferenceReward


class RLAIFAgent(BaseAgent):
    """Independent A2C trader with a synthetic pairwise preference teacher.

    The historical RLAIF name is retained; the teacher is a stated utility
    function, not an LLM or human feedback dataset.
    """
    def __init__(self, agent_id, logger=None, initial_cash=1000., traits=None,
                 rl_algorithm='A2C', curriculum_level=0, extra_actions=0):
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
        # Agent-specific prediction volatility characteristics
        self.prediction_volatility = np.random.uniform(0.5, 2.0)  # Base volatility multiplier
        self.high_volatility_prob = np.random.uniform(0.05, 0.15)  # Prob of high volatility prediction
        # Kaiser influence parameters
        self.kaiser_influence_susceptibility = np.random.uniform(0.1, 0.3)  # How much this agent is influenced by Kaiser
        self.kaiser_action_memory = []  # Track Kaiser's recent actions
        # Stochastic decision parameters
        self.decision_volatility = np.random.uniform(0.05, 0.2)  # Random noise in decision-making
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
            buy_room = max(0., wealth * self.traits.max_position_size - obs['agent_holdings'][idx] * price)
            buy_qty = min(1000, int(min(cash, buy_room, wealth * .1) / max(price * 1.002, 1e-8))) if active else 0
            sell_qty = min(1000, int(holdings[idx]), max(1, int(wealth * .1 / max(price, 1e-8)))) if active else 0
            for kind, qty, quote in [(1, buy_qty, price), (2, sell_qty, price),
                                     (3, buy_qty, price * .999), (4, sell_qty, price * 1.001)]:
                choices.append(np.array([kind, idx, qty, quote], dtype=np.float32))
                mask.append(active and qty > 0 and bool(obs.get('can_trade', True)))
        return choices, mask

    def decide_action(self, observation, action_space=None):
        choices, mask = self._choices(observation)
        choice = self.policy.select(observation, np.asarray(mask, dtype=bool), self.training_enabled)
        action = choices[choice]
        
        # Apply stochastic noise to action parameters
        # Add small random noise to price (action[3])
        if action[3] > 0:
            price_noise = np.random.normal(0, self.decision_volatility * action[3])
            action[3] = max(action[3] * 0.95, action[3] + price_noise)
        
        # Add small random noise to quantity (action[2])
        if action[2] > 0:
            qty_noise = np.random.normal(0, self.decision_volatility * action[2])
            action[2] = max(1, int(action[2] + qty_noise))
        
        # Apply Kaiser influence if there's a Kaiser action in memory
        if self.kaiser_action_memory and len(self.kaiser_action_memory) > 0:
            kaiser_action = self.kaiser_action_memory[-1]
            # Small probability to be influenced by Kaiser's action type
            if np.random.random() < self.kaiser_influence_susceptibility:
                # Nudge towards Kaiser's action type with variance
                kaiser_type = kaiser_action.get('type', 0)
                current_type = int(action[0])
                
                # If Kaiser is buying (type 1 or 3), slightly bias towards buying
                # If Kaiser is selling (type 2 or 4), slightly bias towards selling
                if kaiser_type in (1, 3) and current_type in (0, 2, 4):
                    # Convert to buy with small probability
                    if np.random.random() < 0.3:
                        action[0] = kaiser_type
                elif kaiser_type in (2, 4) and current_type in (0, 1, 3):
                    # Convert to sell with small probability
                    if np.random.random() < 0.3:
                        action[0] = kaiser_type
        
        self.state.last_action = dict(type=int(action[0]), asset_idx=int(action[1]), quantity=int(action[2]), price=float(action[3]))
        self.action_history.append(action.copy())
        self.action_history = self.action_history[-50:]
        return action
    
    def observe_kaiser_action(self, kaiser_action: dict):
        """Called when the Kaiser (bbox winner) takes an action"""
        self.kaiser_action_memory.append(kaiser_action)
        if len(self.kaiser_action_memory) > 20:
            self.kaiser_action_memory = self.kaiser_action_memory[-20:]

    def learn(self, reward, next_obs, terminated, truncated):
        if self.training_enabled:
            self.policy.observe(reward, next_obs, terminated, truncated)
            self.training_step += 1

    def update_state(self, reward, done, info):
        super().update_state(reward, done, info)
        self.episode_rewards.append(float(reward))
        self.episode_rewards = self.episode_rewards[-1000:]

    def generate_prediction(self, observation, horizon=5):
        from src.simulation.black_box import Prediction, PredictionType
        p = np.asarray(observation['price_history'])[:, 0]
        p = p[p > 0]
        returns = np.diff(np.log(p[-20:]))
        trend = float(np.mean(returns)) if len(returns) else 0.
        vol = max(float(np.std(returns)), .001) if len(returns) else .001
        
        # Apply agent-specific volatility characteristics
        # Small probability of high volatility prediction
        if np.random.random() < self.high_volatility_prob:
            vol_multiplier = self.prediction_volatility * 3.0  # High volatility spike
        else:
            vol_multiplier = self.prediction_volatility
        
        # Add day-to-day volatility to predictions
        base_trend = np.clip(trend, -.02, .02)
        daily_trends = []
        for i in range(horizon):
            # Each day has small random variation around the base trend
            daily_noise = np.random.normal(0, 0.005 * vol_multiplier)
            daily_trend = base_trend + daily_noise
            daily_trends.append(np.clip(daily_trend, -.03, .03))
        
        values = (p[-1] * np.exp(np.array(daily_trends) * np.arange(1, horizon+1))).tolist()
        width = 1.645 * vol * vol_multiplier * p[-1] * np.sqrt(horizon)
        
        # Confidence varies with volatility
        confidence = max(0.4, 0.7 - 0.1 * vol_multiplier)
        
        return Prediction(self.agent_id, PredictionType.PRICE_LEVEL, values,
                          (max(.01, min(values)-width), max(values)+width), confidence)

    def reset(self, initial_cash=None):
        super().reset(initial_cash)
        self.action_history.clear()
        self.episode_rewards.clear()
        self.policy.pending = None
        self.policy.rollout.clear()

    def snapshot(self):
        return {'policy': self.policy.snapshot(), 'preference': self.preference_model.state(),
                'training_step': self.training_step}

    def restore(self, state):
        self.policy.restore(state['policy'])
        self.preference_model.restore(state['preference'])
        self.training_step = state['training_step']

    def save_model(self, path):
        torch.save(self.snapshot(), path)

    def load_model(self, path, env=None):
        self.restore(torch.load(path, map_location='cpu', weights_only=True))

    def get_info(self):
        return {**super().get_info(), 'rl_algorithm': self.rl_algorithm,
                'training_step': self.training_step, 'policy_updates': self.policy.updates,
                'preference_updates': self.preference_model.updates}
