from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from collections import deque


class WorldModelType(Enum):
    """Types of world models"""
    DYNAMICS = "dynamics"  # Predicts next state
    REWARD = "reward"  # Predicts immediate reward
    EXPLANATORY = "explanatory"  # Explains causal relationships
    DELAY_AWARE = "delay_aware"  # Handles delayed observations


@dataclass
class WorldModelState:
    """
    State of the world model.
    """
    latent_state: np.ndarray
    belief_state: np.ndarray
    uncertainty: float = 0.0
    prediction_horizon: int = 5
    delay_estimate: int = 0


class DynamicsModel(nn.Module):
    """
    Neural network model for predicting state transitions.
    Based on research: World models for RL from delayed observations.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Encoder: state -> latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Dynamics: latent + action -> next latent
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder: latent -> state
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Uncertainty estimator
        self.uncertainty_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state to latent representation"""
        return self.encoder(state)
    
    def predict_next_latent(
        self,
        latent: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next latent state"""
        combined = torch.cat([latent, action], dim=-1)
        return self.dynamics(combined)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to state"""
        return self.decoder(latent)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict next state and uncertainty.
        Returns (predicted_state, uncertainty)
        """
        latent = self.encode(state)
        next_latent = self.predict_next_latent(latent, action)
        next_state = self.decode(next_latent)
        uncertainty = self.uncertainty_head(next_latent)
        
        return next_state, uncertainty
    
    def predict_trajectory(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trajectory over multiple steps.
        Returns (predicted_states, uncertainties)
        """
        batch_size, seq_len, action_dim = actions.shape
        device = initial_state.device
        
        states = []
        uncertainties = []
        
        current_state = initial_state
        latent = self.encode(current_state)
        
        for t in range(seq_len):
            action = actions[:, t, :]
            next_latent = self.predict_next_latent(latent, action)
            next_state = self.decode(next_latent)
            uncertainty = self.uncertainty_head(next_latent)
            
            states.append(next_state)
            uncertainties.append(uncertainty)
            
            latent = next_latent
        
        states = torch.stack(states, dim=1)
        uncertainties = torch.stack(uncertainties, dim=1)
        
        return states, uncertainties


class RewardModel(nn.Module):
    """
    Neural network model for predicting immediate rewards.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()
        
        layers = []
        input_dim = state_dim + action_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict reward for state-action pair"""
        combined = torch.cat([state, action], dim=-1)
        return self.network(combined)


class DelayAwareModel(nn.Module):
    """
    Model for handling delayed observations and credit assignment.
    Based on research: Delay-aware RL with world models.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_delay: int = 10,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.max_delay = max_delay
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # LSTM for processing delayed observations
        self.lstm = nn.LSTM(
            input_size=state_dim + action_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Delay estimator
        self.delay_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_delay + 1),
            nn.Softmax(dim=-1)
        )
        
        # Credit assignment network
        self.credit_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        state_history: torch.Tensor,
        action_history: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process delayed observations and estimate delay.
        
        Args:
            state_history: (batch, seq_len, state_dim)
            action_history: (batch, seq_len, action_dim)
        
        Returns:
            (hidden_state, delay_distribution, credit_weights)
        """
        combined = torch.cat([state_history, action_history], dim=-1)
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(combined)
        
        # Get final hidden state
        final_hidden = hidden[-1]  # (batch, hidden_dim)
        
        # Estimate delay distribution
        delay_dist = self.delay_estimator(final_hidden)
        
        # Estimate credit weights for historical actions
        credit_weights = self.credit_network(lstm_out)
        
        return final_hidden, delay_dist, credit_weights
    
    def infer_delay(
        self,
        state_history: torch.Tensor,
        action_history: torch.Tensor
    ) -> int:
        """Infer the delay from observation history"""
        _, delay_dist, _ = self.forward(state_history, action_history)
        delay = torch.argmax(delay_dist, dim=-1).item()
        return delay


class ExplanatoryWorldModel:
    """
    Explanatory world model for backward credit assignment.
    Based on research: Explanatory World Models for delayed rewards.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_factors: int = 10
    ):
        self.num_factors = num_factors
        
        # Causal graph (adjacency matrix)
        self.causal_graph = np.random.rand(num_factors, num_factors)
        self.causal_graph = (self.causal_graph > 0.7).astype(float)
        
        # Factor importance weights
        self.factor_importance = np.random.rand(num_factors)
        self.factor_importance /= self.factor_importance.sum()
    
    def explain_action(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float
    ) -> Dict[str, Any]:
        """
        Explain why an action led to a reward.
        Returns attribution of reward to different factors.
        """
        # Simple attribution based on factor importance and reward magnitude
        attribution = self.factor_importance * abs(reward)
        
        # Normalize
        if attribution.sum() > 0:
            attribution /= attribution.sum()
        
        return {
            'reward': reward,
            'attribution': attribution,
            'dominant_factor': int(np.argmax(attribution)),
            'explanation_confidence': np.max(attribution)
        }
    
    def update_causal_beliefs(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float
    ):
        """Update causal graph beliefs based on experience"""
        # Simplified update: strengthen connections that led to positive rewards
        if reward > 0:
            # Strengthen causal connections
            self.causal_graph = np.minimum(1.0, self.causal_graph + 0.01)
        else:
            # Weaken causal connections
            self.causal_graph = np.maximum(0.0, self.causal_graph - 0.01)
    
    def get_causal_explanation(
        self,
        action_sequence: List[np.ndarray],
        reward_sequence: List[float]
    ) -> str:
        """Generate textual explanation of causal relationships"""
        if not action_sequence:
            return "No action sequence to explain."
        
        avg_reward = np.mean(reward_sequence)
        
        if avg_reward > 0:
            explanation = f"Actions led to positive outcomes (avg reward: {avg_reward:.3f}). "
            explanation += "Key contributing factors: "
            top_factors = np.argsort(self.factor_importance)[-3:]
            explanation += ", ".join([f"Factor {i}" for i in top_factors])
        else:
            explanation = f"Actions led to negative outcomes (avg reward: {avg_reward:.3f}). "
            explanation += "Reviewing causal relationships..."
        
        return explanation


class WorldModel:
    """
    Main world model that integrates dynamics, reward, and delay-aware models.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_delay: int = 10,
        use_dynamics: bool = True,
        use_reward_model: bool = True,
        use_delay_aware: bool = True,
        use_explanatory: bool = True
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_delay = max_delay
        
        # Initialize sub-models
        if use_dynamics:
            self.dynamics_model = DynamicsModel(state_dim, action_dim)
        else:
            self.dynamics_model = None
        
        if use_reward_model:
            self.reward_model = RewardModel(state_dim, action_dim)
        else:
            self.reward_model = None
        
        if use_delay_aware:
            self.delay_model = DelayAwareModel(state_dim, action_dim, max_delay)
        else:
            self.delay_model = None
        
        if use_explanatory:
            self.explanatory_model = ExplanatoryWorldModel(state_dim, action_dim)
        else:
            self.explanatory_model = None
        
        # History buffers
        self.state_history = deque(maxlen=max_delay * 2)
        self.action_history = deque(maxlen=max_delay * 2)
        self.reward_history = deque(maxlen=max_delay * 2)
        
        # Training mode
        self.training_mode = False
    
    def predict_next_state(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Predict next state and uncertainty.
        Returns (predicted_state, uncertainty)
        """
        if self.dynamics_model is None:
            return state.copy(), 0.5
        
        self.dynamics_model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            
            next_state, uncertainty = self.dynamics_model(state_tensor, action_tensor)
            
            return next_state.squeeze(0).numpy(), uncertainty.item()
    
    def predict_reward(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> float:
        """Predict immediate reward for state-action pair"""
        if self.reward_model is None:
            return 0.0
        
        self.reward_model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            
            reward = self.reward_model(state_tensor, action_tensor)
            return reward.item()
    
    def estimate_delay(
        self,
        state_history: List[np.ndarray],
        action_history: List[np.ndarray]
    ) -> int:
        """Estimate delay in observations"""
        if self.delay_model is None:
            return 0
        
        if len(state_history) < self.max_delay:
            return 0
        
        self.delay_model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(np.array(state_history)).unsqueeze(0)
            action_tensor = torch.FloatTensor(np.array(action_history)).unsqueeze(0)
            
            delay = self.delay_model.infer_delay(state_tensor, action_tensor)
            return delay
    
    def update_history(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float
    ):
        """Update history buffers"""
        self.state_history.append(state.copy())
        self.action_history.append(action.copy())
        self.reward_history.append(reward)
    
    def get_explanation(
        self,
        action_sequence: List[np.ndarray],
        reward_sequence: List[float]
    ) -> Dict[str, Any]:
        """Get explanation for action-reward sequence"""
        if self.explanatory_model is None:
            return {'explanation': 'Explanatory model not enabled.'}
        
        if not action_sequence:
            return {'explanation': 'No actions to explain.'}
        
        explanation = self.explanatory_model.get_causal_explanation(
            action_sequence,
            reward_sequence
        )
        
        return {'explanation': explanation}
    
    def imagine_trajectory(
        self,
        initial_state: np.ndarray,
        action_sequence: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Imagine future trajectory given action sequence.
        Returns (predicted_states, predicted_rewards)
        """
        if self.dynamics_model is None or self.reward_model is None:
            return [initial_state.copy()], [0.0]
        
        self.dynamics_model.eval()
        self.reward_model.eval()
        
        with torch.no_grad():
            current_state = torch.FloatTensor(initial_state).unsqueeze(0)
            states = [initial_state.copy()]
            rewards = []
            
            for action in action_sequence:
                action_tensor = torch.FloatTensor(action).unsqueeze(0)
                
                # Predict next state
                next_state, _ = self.dynamics_model(current_state, action_tensor)
                
                # Predict reward
                reward = self.reward_model(current_state, action_tensor)
                
                states.append(next_state.squeeze(0).numpy())
                rewards.append(reward.item())
                
                current_state = next_state
            
            return states, rewards
    
    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform one training step on world models.
        Returns loss dictionary.
        """
        losses = {}
        
        if self.dynamics_model is not None and self.training_mode:
            # Train dynamics model
            self.dynamics_model.train()
            state_tensor = torch.FloatTensor(states)
            action_tensor = torch.FloatTensor(actions)
            next_state_tensor = torch.FloatTensor(next_states)
            
            pred_next_state, _ = self.dynamics_model(state_tensor, action_tensor)
            dynamics_loss = nn.MSELoss()(pred_next_state, next_state_tensor)
            
            self.dynamics_model.optimizer.zero_grad()
            dynamics_loss.backward()
            self.dynamics_model.optimizer.step()
            
            losses['dynamics_loss'] = dynamics_loss.item()
        
        if self.reward_model is not None and self.training_mode:
            # Train reward model
            self.reward_model.train()
            reward_tensor = torch.FloatTensor(rewards)
            
            pred_reward = self.reward_model(state_tensor, action_tensor)
            reward_loss = nn.MSELoss()(reward_tensor, pred_reward)
            
            self.reward_model.optimizer.zero_grad()
            reward_loss.backward()
            self.reward_model.optimizer.step()
            
            losses['reward_loss'] = reward_loss.item()
        
        return losses
    
    def set_training_mode(self, mode: bool):
        """Set training mode for all models"""
        self.training_mode = mode
        
        if self.dynamics_model is not None:
            self.dynamics_model.train(mode)
            if mode and not hasattr(self.dynamics_model, 'optimizer'):
                self.dynamics_model.optimizer = torch.optim.Adam(
                    self.dynamics_model.parameters(),
                    lr=1e-3
                )
        
        if self.reward_model is not None:
            self.reward_model.train(mode)
            if mode and not hasattr(self.reward_model, 'optimizer'):
                self.reward_model.optimizer = torch.optim.Adam(
                    self.reward_model.parameters(),
                    lr=1e-3
                )
        
        if self.delay_model is not None:
            self.delay_model.train(mode)
            if mode and not hasattr(self.delay_model, 'optimizer'):
                self.delay_model.optimizer = torch.optim.Adam(
                    self.delay_model.parameters(),
                    lr=1e-3
                )
    
    def reset(self):
        """Reset history buffers"""
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
    
    def save(self, path: str):
        """Save world model to disk"""
        torch.save({
            'dynamics_model': self.dynamics_model.state_dict() if self.dynamics_model else None,
            'reward_model': self.reward_model.state_dict() if self.reward_model else None,
            'delay_model': self.delay_model.state_dict() if self.delay_model else None,
            'explanatory_model': self.explanatory_model.causal_graph if self.explanatory_model else None
        }, path)
    
    def load(self, path: str):
        """Load world model from disk"""
        checkpoint = torch.load(path)
        
        if self.dynamics_model is not None and checkpoint['dynamics_model'] is not None:
            self.dynamics_model.load_state_dict(checkpoint['dynamics_model'])
        
        if self.reward_model is not None and checkpoint['reward_model'] is not None:
            self.reward_model.load_state_dict(checkpoint['reward_model'])
        
        if self.delay_model is not None and checkpoint['delay_model'] is not None:
            self.delay_model.load_state_dict(checkpoint['delay_model'])
        
        if self.explanatory_model is not None and checkpoint['explanatory_model'] is not None:
            self.explanatory_model.causal_graph = checkpoint['explanatory_model']


if __name__ == "__main__":
    # Test world model
    state_dim = 10
    action_dim = 3
    
    world_model = WorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        max_delay=5,
        use_dynamics=True,
        use_reward_model=True,
        use_delay_aware=True,
        use_explanatory=True
    )
    
    # Test prediction
    state = np.random.randn(state_dim)
    action = np.random.randn(action_dim)
    
    next_state, uncertainty = world_model.predict_next_state(state, action)
    print(f"Predicted next state shape: {next_state.shape}")
    print(f"Uncertainty: {uncertainty:.4f}")
    
    reward = world_model.predict_reward(state, action)
    print(f"Predicted reward: {reward:.4f}")
    
    # Test trajectory imagination
    action_sequence = [np.random.randn(action_dim) for _ in range(5)]
    states, rewards = world_model.imagine_trajectory(state, action_sequence)
    print(f"Imagined trajectory length: {len(states)}")
    print(f"Predicted rewards: {rewards}")
    
    # Test explanation
    explanation = world_model.get_explanation(action_sequence, rewards)
    print(f"Explanation: {explanation}")
