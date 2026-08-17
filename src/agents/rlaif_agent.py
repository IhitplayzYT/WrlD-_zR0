from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import random

from .base_agent import BaseAgent, AgentTraits, AgentType, AgentState


@dataclass
class PreferenceModel:
    """
    Preference model for RLAIF - evaluates agent actions based on multiple principles.
    Implements Multi-Objective RLAIF (MORLAIF) with decomposed preference models.
    """
    profitability_weight: float = 0.4
    risk_weight: float = 0.3
    market_impact_weight: float = 0.2
    consistency_weight: float = 0.1
    
    def evaluate_action(
        self,
        action: np.ndarray,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """
        Evaluate an action based on multiple principles and return combined score.
        """
        scores = {}
        
        # Profitability score
        scores['profitability'] = self._evaluate_profitability(action, state, context)
        
        # Risk score
        scores['risk'] = self._evaluate_risk(action, state, context)
        
        # Market impact score
        scores['market_impact'] = self._evaluate_market_impact(action, state, context)
        
        # Consistency score
        scores['consistency'] = self._evaluate_consistency(action, state, context)
        
        # Combine scores using weights
        total_score = (
            scores['profitability'] * self.profitability_weight +
            scores['risk'] * self.risk_weight +
            scores['market_impact'] * self.market_impact_weight +
            scores['consistency'] * self.consistency_weight
        )
        
        return total_score, scores
    
    def _evaluate_profitability(self, action: np.ndarray, state: Dict, context: Dict) -> float:
        """Evaluate potential profitability of action"""
        # Simple heuristic: buy low, sell high
        current_price = state.get('current_price', 100.0)
        price_history = state.get('price_history', [])
        
        if len(price_history) < 5:
            return 0.5
        
        recent_trend = (price_history[-1] - price_history[-5]) / price_history[-5]
        action_type = int(action[0])
        
        if action_type in [1, 3]:  # Buy actions
            # Good to buy if trend is positive
            return max(0.0, min(1.0, 0.5 + recent_trend * 10))
        elif action_type in [2, 4]:  # Sell actions
            # Good to sell if trend is negative
            return max(0.0, min(1.0, 0.5 - recent_trend * 10))
        else:  # Hold
            return 0.5
    
    def _evaluate_risk(self, action: np.ndarray, state: Dict, context: Dict) -> float:
        """Evaluate riskiness of action"""
        action_type = int(action[0])
        quantity = int(action[1])
        cash = state.get('agent_cash', 1000.0)
        
        if action_type in [1, 3]:  # Buy actions
            # Higher quantity = higher risk
            position_size = (quantity * action[2]) / cash if cash > 0 else 1.0
            return max(0.0, min(1.0, 1.0 - position_size))
        elif action_type in [2, 4]:  # Sell actions
            # Selling is generally lower risk
            return 0.7
        else:  # Hold
            return 0.9  # Holding is low risk
    
    def _evaluate_market_impact(self, action: np.ndarray, state: Dict, context: Dict) -> float:
        """Evaluate market impact of action"""
        quantity = int(action[1])
        order_book_depth = context.get('order_book_depth', 1000)
        
        # Lower market impact is better
        impact = quantity / order_book_depth if order_book_depth > 0 else 1.0
        return max(0.0, min(1.0, 1.0 - impact))
    
    def _evaluate_consistency(self, action: np.ndarray, state: Dict, context: Dict) -> float:
        """Evaluate consistency with agent's historical behavior"""
        history = context.get('action_history', [])
        if len(history) < 3:
            return 0.5
        
        # Check if action is consistent with recent actions
        recent_actions = [int(h[0]) for h in history[-3:]]
        current_action_type = int(action[0])
        
        consistency = recent_actions.count(current_action_type) / len(recent_actions)
        return consistency


class RLAIFAgent(BaseAgent):
    """
    RLAIF (Reinforcement Learning from AI Feedback) Agent.
    Implements advanced RLAIF with multi-objective feedback and curriculum learning.
    """
    
    def __init__(
        self,
        agent_id: str,
        initial_cash: float = 1000.0,
        traits: Optional[AgentTraits] = None,
        rl_algorithm: str = "PPO",
        curriculum_level: int = 0
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.RLAIF,
            initial_cash=initial_cash,
            traits=traits
        )
        
        self.rl_algorithm = rl_algorithm
        self.curriculum_level = curriculum_level
        
        # Preference model for AI feedback
        self.preference_model = PreferenceModel()
        
        # RL model (will be initialized when environment is available)
        self.rl_model = None
        self.model_initialized = False
        
        # Training state
        self.training_step = 0
        self.episode_rewards = []
        self.action_history = []
        
        # Curriculum learning parameters
        self.curriculum_thresholds = {
            0: {'max_complexity': 0.3, 'event_frequency': 0.5},
            1: {'max_complexity': 0.5, 'event_frequency': 0.7},
            2: {'max_complexity': 0.7, 'event_frequency': 0.9},
            3: {'max_complexity': 1.0, 'event_frequency': 1.0}
        }
    
    def initialize_rl_model(self, env):
        """
        Initialize the RL model with the given environment.
        This should be called once the environment is available.
        """
        if self.model_initialized:
            return
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Initialize RL algorithm
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
    
    def decide_action(
        self,
        observation: Dict[str, Any],
        action_space: Any
    ) -> np.ndarray:
        """
        Decide action using RL policy with AI feedback integration.
        """
        # Process observation based on traits
        processed_obs = self.process_observation(observation)
        
        if self.model_initialized and self.rl_model is not None:
            # Use RL policy to get action
            # Convert observation to format expected by RL model
            obs_array = self._obs_to_array(processed_obs)
            
            try:
                action, _ = self.rl_model.predict(obs_array, deterministic=False)
                # Convert action to expected format [action_type, quantity, price]
                formatted_action = self._format_rl_action(action, observation)
            except Exception as e:
                # Fallback to trait-based action if RL fails
                formatted_action = self._trait_based_action(processed_obs, action_space)
        else:
            # Use trait-based action if model not initialized
            formatted_action = self._trait_based_action(processed_obs, action_space)
        
        # Apply trait modifications
        modified_action = self.apply_trait_modifications(formatted_action)
        
        # Get AI feedback on the action
        ai_feedback_score, component_scores = self.preference_model.evaluate_action(
            modified_action,
            observation,
            {'action_history': self.action_history}
        )
        
        # Store action for history
        self.action_history.append(modified_action.copy())
        if len(self.action_history) > 50:
            self.action_history.pop(0)
        
        # Store feedback in memory
        self.state.add_memory({
            'action': modified_action.copy(),
            'ai_feedback_score': ai_feedback_score,
            'component_scores': component_scores,
            'curriculum_level': self.curriculum_level
        })
        
        self.state.last_action = {
            'type': int(modified_action[0]),
            'quantity': int(modified_action[1]),
            'price': float(modified_action[2]),
            'ai_feedback_score': ai_feedback_score
        }
        
        return modified_action
    
    def _obs_to_array(self, observation: Dict[str, Any]) -> np.ndarray:
        """Convert observation dictionary to numpy array for RL model"""
        obs_array = []
        
        # Price history
        if 'price_history' in observation:
            obs_array.extend(observation['price_history'])
        else:
            obs_array.extend([0.0] * 50)
        
        # Order book state
        if 'order_book_bids' in observation:
            obs_array.extend(observation['order_book_bids'].flatten())
        else:
            obs_array.extend([0.0] * 20)
        
        if 'order_book_asks' in observation:
            obs_array.extend(observation['order_book_asks'].flatten())
        else:
            obs_array.extend([0.0] * 20)
        
        # Agent state
        obs_array.append(observation.get('agent_cash', 1000.0))
        obs_array.append(observation.get('agent_holdings', 0))
        obs_array.append(observation.get('agent_total_value', 1000.0))
        
        # Market state
        obs_array.append(observation.get('is_black_box', 0))
        obs_array.append(observation.get('current_price', 100.0))
        obs_array.append(observation.get('market_cap', 1000000.0))
        
        # Events
        if 'active_events' in observation:
            obs_array.extend(observation['active_events'])
        else:
            obs_array.extend([0.0] * 10)
        
        return np.array(obs_array, dtype=np.float32)
    
    def _format_rl_action(self, rl_action: np.ndarray, observation: Dict) -> np.ndarray:
        """Format RL action to expected [action_type, quantity, price] format"""
        # This is a simplified conversion - in practice, you'd need to handle
        # the specific action space of your RL algorithm
        action_type = int(abs(rl_action[0]) % 5)  # Ensure valid action type
        quantity = int(abs(rl_action[1]) % 100) + 1  # Ensure positive quantity
        
        current_price = observation.get('current_price', 100.0)
        price = current_price * (1 + (rl_action[2] * 0.1))  # Price around current
        
        return np.array([action_type, quantity, price], dtype=np.float32)
    
    def _trait_based_action(self, observation: Dict, action_space: Any) -> np.ndarray:
        """Generate action based on traits when RL model is not available"""
        modifiers = self.traits.get_behavior_modifiers()
        
        # Base action on traits and observation
        current_price = observation.get('current_price', 100.0)
        price_history = observation.get('price_history', [])
        
        # Determine action type based on trend and traits
        if len(price_history) >= 5:
            recent_trend = (price_history[-1] - price_history[-5]) / price_history[-5]
            trend_modifier = modifiers['trend_modifier']
            
            if recent_trend > 0.01 and trend_modifier > 0.6:
                action_type = 1  # Market buy (follow trend)
            elif recent_trend < -0.01 and trend_modifier < 0.4:
                action_type = 2  # Market sell (contrarian)
            else:
                action_type = 0  # Hold
        else:
            action_type = random.choice([0, 1, 2])
        
        # Determine quantity based on risk modifier
        risk_modifier = modifiers['risk_modifier']
        base_quantity = int(10 + risk_modifier * 50)
        quantity = max(1, min(100, base_quantity))
        
        # Determine price for limit orders
        price = current_price * (1 + random.uniform(-0.02, 0.02))
        
        # Convert to limit order if patient
        if modifiers['patience_modifier'] > 0.7 and action_type in [1, 2]:
            action_type = action_type + 2  # Convert to limit order
        
        return np.array([action_type, quantity, price], dtype=np.float32)
    
    def update_state(self, reward: float, done: bool, info: Dict):
        """Update state and training progress"""
        super().update_state(reward, done, info)
        
        self.episode_rewards.append(reward)
        self.training_step += 1
        
        # Check for curriculum progression
        self._check_curriculum_progression()
    
    def _check_curriculum_progression(self):
        """Check if agent should progress to next curriculum level"""
        if self.curriculum_level < 3:
            recent_performance = self.state.get_recent_performance(20)
            threshold = self.curriculum_thresholds[self.curriculum_level]
            
            # Progress if performing well at current level
            if recent_performance > 50:  # Positive performance threshold
                self.curriculum_level += 1
                print(f"Agent {self.agent_id} progressed to curriculum level {self.curriculum_level}")
    
    def train_step(self, env, n_steps: int = 1000):
        """
        Perform training steps on the RL model.
        """
        if not self.model_initialized or self.rl_model is None:
            self.initialize_rl_model(env)
        
        try:
            self.rl_model.learn(n_steps)
            self.training_step += n_steps
        except Exception as e:
            print(f"Training error for agent {self.agent_id}: {e}")
    
    def get_curriculum_config(self) -> Dict[str, float]:
        """Get current curriculum configuration"""
        return self.curriculum_thresholds.get(
            self.curriculum_level,
            self.curriculum_thresholds[3]
        )
    
    def save_model(self, path: str):
        """Save the RL model"""
        if self.model_initialized and self.rl_model is not None:
            self.rl_model.save(path)
    
    def load_model(self, path: str, env):
        """Load the RL model"""
        if self.rl_algorithm == "PPO":
            self.rl_model = PPO.load(path, env=DummyVecEnv([lambda: env]))
        elif self.rl_algorithm == "SAC":
            self.rl_model = SAC.load(path, env=DummyVecEnv([lambda: env]))
        
        self.model_initialized = True
    
    def get_info(self) -> Dict[str, Any]:
        """Get extended agent information including RLAIF-specific data"""
        info = super().get_info()
        info.update({
            'rl_algorithm': self.rl_algorithm,
            'curriculum_level': self.curriculum_level,
            'training_step': self.training_step,
            'model_initialized': self.model_initialized,
            'average_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'preference_weights': self.preference_model.__dict__
        })
        return info


if __name__ == "__main__":
    # Test the RLAIF agent
    agent = RLAIFAgent("test_agent", initial_cash=1000.0)
    
    # Create a dummy observation
    dummy_obs = {
        'price_history': [100.0 + i * 0.1 for i in range(50)],
        'order_book_bids': np.zeros((10, 2)),
        'order_book_asks': np.zeros((10, 2)),
        'agent_cash': 1000.0,
        'agent_holdings': 0,
        'agent_total_value': 1000.0,
        'is_black_box': 0,
        'current_price': 105.0,
        'market_cap': 1000000.0,
        'active_events': np.zeros(10)
    }
    
    # Test action decision
    action = agent.decide_action(dummy_obs, None)
    print(f"Action: {action}")
    print(f"Agent info: {agent.get_info()}")
