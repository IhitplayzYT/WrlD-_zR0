from src.utils.logging import JSONLogger, get_logger, LogCategory
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple
import random
import numpy as np
from datetime import datetime
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from .base_agent import BaseAgent, AgentTraits, AgentType 

@dataclass
class PreferenceModel:
    """ Multi-Objective RLAIF (MORLAIF) with decomposed preference models """
    profitability_weight: float = 0.4
    risk_weight: float = 0.3
    market_impact_weight: float = 0.2
    consistency_weight: float = 0.1
    
    def evaluate_action(self,action: np.ndarray,state: Dict[str, Any],context: Dict[str, Any]) -> tuple[float, dict[str, float]]:
        scores = {}
        scores['profitability'] = self._evaluate_profitability(action, state, context)
        scores['risk'] = self._evaluate_risk(action, state, context)
        scores['market_impact'] = self._evaluate_market_impact(action, state, context)
        scores['consistency'] = self._evaluate_consistency(action, state, context)
        total_score = (scores['profitability'] * self.profitability_weight +scores['risk'] * self.risk_weight +scores['market_impact'] * self.market_impact_weight +scores['consistency'] * self.consistency_weight)
        return total_score, scores
    
    def _evaluate_profitability(self, action: np.ndarray, state: Dict, context: Dict) -> float:
        current_price = 100.0
        if 'current_price' in state:
            cp = state['current_price']
            current_price = cp[0] if isinstance(cp, (list, np.ndarray)) and len(cp) > 0 else float(cp)
        price_history = state.get('price_history', [])
        if len(price_history) < 5:
            return 0.5
        recent_trend = (price_history[-1] - price_history[-5]) / price_history[-5] if isinstance(price_history[-1], (int, float)) else 0.0
        action_type = int(action[0])
        
        if action_type in [1, 3]:  # Buy
            return max(0.0, min(1.0, 0.5 + float(np.mean(recent_trend)) * 10))
        elif action_type in [2, 4]:  # Sell
            return max(0.0, min(1.0, 0.5 - float(np.mean(recent_trend)) * 10))
        else:  # Hold
            return 0.5
    
    def _evaluate_risk(self, action: np.ndarray, state: Dict, context: Dict) -> float:
        action_type = int(action[0])
        quantity = int(action[2]) if len(action) > 2 else 1
        price = float(action[3]) if len(action) > 3 else 100.0
        cash = float(state.get('agent_cash', 1000.0))
        
        if action_type in [1, 3]:  # Buy actions
            position_size = (quantity * price) / cash if cash > 0 else 1.0
            return max(0.0, min(1.0, 1.0 - position_size))
        elif action_type in [2, 4]:  # Sell actions
            return 0.7
        else:  # Hold
            return 0.9
    
    def _evaluate_market_impact(self, action: np.ndarray, state: Dict, context: Dict) -> float:
        quantity = int(action[2]) if len(action) > 2 else 1
        order_book_depth = context.get('order_book_depth', 1000)
        
        impact = quantity / order_book_depth if order_book_depth > 0 else 1.0
        return max(0.0, min(1.0, 1.0 - impact))
    
    def _evaluate_consistency(self, action: np.ndarray, state: Dict, context: Dict) -> float:
        history = context.get('action_history', [])
        if len(history) < 3:
            return 0.5
        
        recent_actions = [int(h[0]) for h in history[-3:]]
        current_action_type = int(action[0])
        
        consistency = recent_actions.count(current_action_type) / len(recent_actions)
        return consistency


class RLAIFAgent(BaseAgent):
    """
    RLAIF (Reinforcement Learning from AI Feedback) Agent.
    Implements advanced RLAIF with multi-objective feedback and curriculum learning.
    """
    
    def __init__(self, agent_id: str, logger: Optional[JSONLogger] = None, initial_cash: float = 1000.0, traits: Optional[AgentTraits] = None, rl_algorithm: str = "PPO", curriculum_level: int = 0):
        super().__init__(agent_id=agent_id, agent_type=AgentType.RLAIF, initial_cash=initial_cash, traits=traits)
        self.logger = logger or get_logger()
        self.rl_algorithm = rl_algorithm
        self.curriculum_level = curriculum_level
        
        self.preference_model = PreferenceModel()
        
        self.rl_model = None
        self.model_initialized = False
        
        self.training_step = 0
        self.episode_rewards = []
        self.action_history = []
        
        self.curriculum_thresholds = {0: {'max_complexity': 0.3, 'event_frequency': 0.5}, 1: {'max_complexity': 0.5, 'event_frequency': 0.7}, 2: {'max_complexity': 0.7, 'event_frequency': 0.9}, 3: {'max_complexity': 1.0, 'event_frequency': 1.0}}
    
    def initialize_rl_model(self, env):
        if self.model_initialized or not HAS_SB3:
            return
        
        try:
            vec_env = DummyVecEnv([lambda: env])
            if self.rl_algorithm == "PPO":
                self.rl_model = PPO(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    verbose=0
                )
            elif self.rl_algorithm == "SAC":
                self.rl_model = SAC(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=3e-4,
                    buffer_size=100000,
                    batch_size=256,
                    gamma=0.99,
                    verbose=0
                )
            self.model_initialized = True
        except Exception as e:
            self.logger.warning(LogCategory.AGENT, f"Could not initialize SB3 model for {self.agent_id}: {e}")
            self.model_initialized = False
    
    def decide_action(self, observation: Dict[str, Any], action_space: Any) -> np.ndarray:
        processed_obs = self.process_observation(observation)

        if self.model_initialized and self.rl_model is not None and HAS_SB3:
            obs_array = self._obs_to_array(processed_obs)
            try:
                action, _ = self.rl_model.predict(obs_array, deterministic=False)
                formatted_action = self._format_rl_action(action, observation)
            except Exception:
                formatted_action = self._trait_based_action(processed_obs, action_space)
        else:
            formatted_action = self._trait_based_action(processed_obs, action_space)

        modified_action = self.apply_trait_modifications(formatted_action)

        ai_feedback_score, component_scores = self.preference_model.evaluate_action(
            modified_action,
            observation,
            {'action_history': self.action_history}
        )

        self.action_history.append(modified_action.copy())
        if len(self.action_history) > 50:
            self.action_history.pop(0)

        self.state.add_memory({'action': modified_action.copy(), 'ai_feedback_score': ai_feedback_score, 'component_scores': component_scores, 'curriculum_level': self.curriculum_level})
        self.state.last_action = {'type': int(modified_action[0]), 'asset_idx': int(modified_action[1]), 'quantity': int(modified_action[2]), 'price': float(modified_action[3]), 'ai_feedback_score': ai_feedback_score}

        return modified_action

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
        for _ in range(horizon):
            change = np.random.normal(mean_return, std_return)
            current_p = max(1.0, current_p * (1 + change))
            predicted_prices.append(current_p)
        conf_level = min(0.95, max(0.3, 0.7 - std_return * 10))
        price_std = std_return * last_price
        conf_interval = (min(predicted_prices) - price_std, max(predicted_prices) + price_std)
        prediction = Prediction(
            agent_id=self.agent_id,
            pred_t=PredictionType.PRICE_LEVEL,
            pred_val=predicted_prices,
            conf_interv=conf_interval,
            conf_lvl=conf_level
        )
        return prediction
    
    def _obs_to_array(self, observation: Dict[str, Any]) -> np.ndarray:
        obs_array = []
        
        if 'price_history' in observation:
            price_hist = observation['price_history']
            if isinstance(price_hist, np.ndarray):
                obs_array.extend(price_hist.flatten())
            elif isinstance(price_hist, list):
                obs_array.extend(price_hist + [0.0] * (500 - len(price_hist)))
            else:
                obs_array.extend([0.0] * 500)
        else:
            obs_array.extend([0.0] * 500)
        
        if 'order_book_bids' in observation:
            bids = observation['order_book_bids']
            if isinstance(bids, np.ndarray):
                obs_array.extend(bids.flatten())
            else:
                obs_array.extend([0.0] * 200)
        else:
            obs_array.extend([0.0] * 200)
        
        if 'order_book_asks' in observation:
            asks = observation['order_book_asks']
            if isinstance(asks, np.ndarray):
                obs_array.extend(asks.flatten())
            else:
                obs_array.extend([0.0] * 200)
        else:
            obs_array.extend([0.0] * 200)
        
        obs_array.append(float(observation.get('agent_cash', 1000.0)))
        
        if 'agent_holdings' in observation:
            holdings = observation['agent_holdings']
            if isinstance(holdings, np.ndarray):
                obs_array.extend(holdings)
            else:
                obs_array.extend([0.0] * 10)
        else:
            obs_array.extend([0.0] * 10)
        
        obs_array.append(float(observation.get('agent_total_value', 1000.0)))
        obs_array.append(float(observation.get('is_black_box', 0)))
        
        if 'current_price' in observation:
            cp = observation['current_price']
            if isinstance(cp, np.ndarray):
                obs_array.extend(cp)
            else:
                obs_array.extend([100.0] * 10)
        else:
            obs_array.extend([100.0] * 10)
        
        if 'market_cap' in observation:
            mc = observation['market_cap']
            if isinstance(mc, np.ndarray):
                obs_array.extend(mc)
            else:
                obs_array.extend([1000000.0] * 10)
        else:
            obs_array.extend([1000000.0] * 10)
        
        if 'active_events' in observation:
            ev = observation['active_events']
            if isinstance(ev, np.ndarray):
                obs_array.extend(ev)
            else:
                obs_array.extend([0.0] * 10)
        else:
            obs_array.extend([0.0] * 10)
        
        if 'available_assets' in observation:
            aa = observation['available_assets']
            if isinstance(aa, np.ndarray):
                obs_array.extend(aa)
            else:
                obs_array.extend([1.0] + [0.0] * 9)
        else:
            obs_array.extend([1.0] + [0.0] * 9)
        
        return np.array(obs_array, dtype=np.float32)
    
    def _format_rl_action(self, rl_action: np.ndarray, observation: Dict) -> np.ndarray:
        action_type = int(abs(rl_action[0]) % 5)
        asset_idx = int(abs(rl_action[1]) % 10) if len(rl_action) > 1 else 0
        quantity = int(abs(rl_action[2]) % 100) + 1 if len(rl_action) > 2 else 10
        
        current_prices = observation.get('current_price', [100.0] * 10)
        if isinstance(current_prices, np.ndarray) and len(current_prices) > asset_idx:
            current_price = float(current_prices[asset_idx])
        else:
            current_price = 100.0
        
        price = current_price * (1 + (rl_action[3] * 0.1)) if len(rl_action) > 3 else current_price
        return np.array([action_type, asset_idx, quantity, price], dtype=np.float32)
    
    def _trait_based_action(self, observation: Dict, action_space: Any) -> np.ndarray:
        modifiers = self.traits.get_behavior_modifiers()
        
        available_assets = observation.get('available_assets', [1.0] + [0.0] * 9)
        if isinstance(available_assets, np.ndarray):
            asset_idx = int(np.argmax(available_assets))
        else:
            asset_idx = 0
        
        current_prices = observation.get('current_price', [100.0] * 10)
        if isinstance(current_prices, np.ndarray) and len(current_prices) > asset_idx:
            current_price = float(current_prices[asset_idx])
        elif isinstance(current_prices, list) and len(current_prices) > asset_idx:
            current_price = float(current_prices[asset_idx])
        else:
            current_price = 100.0
            
        price_history = observation.get('price_history', [])
        
        if len(price_history) >= 5 and isinstance(price_history, np.ndarray) and price_history.ndim == 2:
            p_slice = price_history[-5:, asset_idx]
            recent_trend = (p_slice[-1] - p_slice[0]) / max(1e-5, p_slice[0])
            trend_modifier = modifiers['trend_modifier']
            
            if recent_trend > 0.01 and trend_modifier > 0.6:
                action_type = 1  # Market buy
            elif recent_trend < -0.01 and trend_modifier < 0.4:
                action_type = 2  # Market sell
            else:
                action_type = 0  # Hold
        else:
            action_type = random.choice([0, 1, 2, 3, 4])
        
        risk_modifier = modifiers['risk_modifier']
        base_quantity = int(10 + risk_modifier * 50)
        quantity = max(1, min(100, base_quantity))
        price = max(1.0, current_price * (1 + random.uniform(-0.02, 0.02)))
        
        if modifiers['patience_modifier'] > 0.7 and action_type in [1, 2]:
            action_type = action_type + 2
        
        return np.array([action_type, asset_idx, quantity, price], dtype=np.float32)
    
    def update_state(self, reward: float, done: bool, info: Dict):
        super().update_state(reward, done, info)
        self.episode_rewards.append(reward)
        self.training_step += 1
        self._check_curriculum_progression()
    
    def _check_curriculum_progression(self):
        if self.curriculum_level < 3:
            recent_performance = self.state.get_recent_performance(20)
            if recent_performance > 50:
                self.curriculum_level += 1
                self.logger.info(LogCategory.AGENT, f"Agent {self.agent_id} progressed to curriculum level {self.curriculum_level}", data={'agent_id': self.agent_id, 'curriculum_level': self.curriculum_level})
    
    def train_step(self, env, n_steps: int = 1000):
        if not HAS_SB3:
            return
        if not self.model_initialized or self.rl_model is None:
            self.initialize_rl_model(env)
        
        try:
            if self.rl_model:
                self.rl_model.learn(n_steps)
                self.training_step += n_steps
        except Exception as e:
            self.logger.error(LogCategory.AGENT, f"Training error for agent {self.agent_id}", data={'agent_id': self.agent_id, 'error': str(e)})
    
    def get_curriculum_config(self) -> Dict[str, float]:
        return self.curriculum_thresholds.get(self.curriculum_level, self.curriculum_thresholds[3])
    
    def save_model(self, path: str):
        if HAS_SB3 and self.model_initialized and self.rl_model is not None:
            self.rl_model.save(path)
    
    def load_model(self, path: str, env):
        if not HAS_SB3:
            return
        try:
            vec_env = DummyVecEnv([lambda: env])
            if self.rl_algorithm == "PPO":
                self.rl_model = PPO.load(path, env=vec_env)
            elif self.rl_algorithm == "SAC":
                self.rl_model = SAC.load(path, env=vec_env)
            self.model_initialized = True
        except Exception as e:
            self.logger.warning(LogCategory.AGENT, f"Failed to load RL model for {self.agent_id}: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({'rl_algorithm': self.rl_algorithm, 'curriculum_level': self.curriculum_level, 'training_step': self.training_step, 'model_initialized': self.model_initialized, 'average_episode_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0, 'preference_weights': self.preference_model.__dict__})
        return info

