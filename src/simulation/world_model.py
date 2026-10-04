from dataclasses import dataclass 
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from collections import deque
from src.utils.logging import LogCategory

class WorldModelType(Enum):
    """Types of world models"""
    DYNAMICS = "dynamics"        # Predicts next state
    REWARD = "reward"            # Predicts immediate reward
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
    
    def __init__(self,state_dim: int,action_dim: int,hidden_dim: int = 256,latent_dim: int = 64,num_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
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
        return self.encoder(state)
    
    def predict_next_latent(self,latent: torch.Tensor,action: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([latent, action], dim=-1)
        return self.dynamics(combined)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
    
    def forward(self,state: torch.Tensor,action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns (predicted_state, uncertainty) """
        next_latent = self.predict_next_latent(self.encode(state), action)

        return self.decode(next_latent), self.uncertainty_head(next_latent)
    
    def predict_trajectory(self,initial_state: torch.Tensor,actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        return torch.stack(states, dim=1), torch.stack(uncertainties, dim=1)


class RewardModel(nn.Module):
    """ NN model for predicting immediate rewards """
    
    def __init__(self,state_dim: int,action_dim: int,hidden_dim: int = 128,num_layers: int = 2):
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self,state: torch.Tensor,action: torch.Tensor) -> torch.Tensor:
        """Predict reward for state-action pair"""
        combined = torch.cat([state, action], dim=-1)
        return self.network(combined)


class DelayAwareModel(nn.Module):
    """ Model for DPOMDP and credit assignment """
    
    def __init__(self,state_dim: int,action_dim: int,max_delay: int = 10,hidden_dim: int = 128):
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
    
    def forward(self,state_history: torch.Tensor,action_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state_history: (batch, seq_len, state_dim)
            action_history: (batch, seq_len, action_dim)
        Returns:
            (hidden_state, delay_distribution, credit_weights)
        """
        combined = torch.cat([state_history, action_history], dim=-1)
        lstm_out, (hidden, cell) = self.lstm(combined)
        final_hidden = hidden[-1]  # (batch, hidden_dim)
        delay_dist = self.delay_estimator(final_hidden)
        credit_weights = self.credit_network(lstm_out)
        return final_hidden, delay_dist, credit_weights
    
    def infer_delay(self,state_history: torch.Tensor,action_history: torch.Tensor) -> int:
        """Infer the delay from observation history"""
        _, delay_dist, _ = self.forward(state_history, action_history)
        delay = torch.argmax(delay_dist, dim=-1).item()
        return delay


class ExplanatoryWorldModel:
    """ Explanatory World Models for delayed rewards. """

    def __init__(self, state_dim: int, action_dim: int, num_factors: int = 10):
        self.state_dim, self.action_dim, self.num_factors = state_dim, action_dim, num_factors
        rng = np.random.default_rng(42)
        self.state_proj = rng.normal(0, 0.1, size=(num_factors, state_dim))
        self.action_proj = rng.normal(0, 0.1, size=(num_factors, action_dim))
        self.causal_graph = (rng.random((num_factors, num_factors)) > 0.75).astype(float)
        self.factor_importance = rng.random(num_factors)
        self.factor_importance /= self.factor_importance.sum() + 1e-8

    def _state_factors(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64).reshape(-1)
        if s.size < self.state_dim:
            s = np.pad(s, (0, self.state_dim - s.size))
        s = s[: self.state_dim]
        fac = np.tanh(self.state_proj @ s)
        sm = np.sum(np.abs(fac))
        return fac / sm if sm > 0 else fac

    def _action_factors(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.size < self.action_dim:
            a = np.pad(a, (0, self.action_dim - a.size))
        a = a[: self.action_dim]
        fac = np.tanh(self.action_proj @ a)
        sm = np.sum(np.abs(fac))
        return fac / sm if sm > 0 else fac

    def explain_action(self, state: np.ndarray, action: np.ndarray, reward: float) -> Dict[str, Any]:
        s_fac, a_fac = self._state_factors(state), self._action_factors(action)
        interaction = s_fac * a_fac
        attribution = self.factor_importance * np.abs(interaction) * abs(reward)
        sm = attribution.sum()
        if sm > 0:
            attribution /= sm
        top2 = np.sort(attribution)[-2:] if attribution.size >= 2 else attribution
        confidence = float(top2[-1] - top2[0]) if top2.size >= 2 else float(np.max(attribution))
        return {'reward': reward, 'attribution': attribution, 'dominant_factor': int(np.argmax(attribution)), 'explanation_confidence': max(0.0, min(1.0, confidence))}

    def update_causal_beliefs(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: float):
        s_fac, a_fac, n_fac = self._state_factors(state), self._action_factors(action), self._state_factors(next_state)
        signed = 1.0 if reward >= 0 else -1.0
        delta = signed * 0.02 * np.outer(s_fac, n_fac) + signed * 0.01 * np.outer(a_fac, n_fac)
        self.causal_graph = np.clip(self.causal_graph + delta, 0.0, 1.0)
        self.factor_importance = np.clip(self.factor_importance + signed * 0.005 * np.abs(s_fac * a_fac), 1e-4, None)
        self.factor_importance /= self.factor_importance.sum()
    
    def get_causal_explanation(self,action_sequence: List[np.ndarray],reward_sequence: List[float]) -> str:
        """Generate explanation of causal relationships"""
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


from src.utils.logging import JSONLogger, get_logger, LogCategory

class WorldModel:
    
    def __init__(self, state_dim: int, action_dim: int, max_delay: int = 10, use_dynamics: bool = True, use_reward_model: bool = True, use_delay_aware: bool = True, use_explanatory: bool = True, logger: Optional[JSONLogger] = None):
        self.logger = logger or get_logger()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_delay = max_delay
        self.training_mode = False
        
        self.dynamics_model = DynamicsModel(state_dim, action_dim) if use_dynamics else None
        self.reward_model = RewardModel(state_dim, action_dim) if use_reward_model else None
        self.delay_model = DelayAwareModel(state_dim, action_dim, max_delay) if use_delay_aware else None
        self.explanatory_model = ExplanatoryWorldModel(state_dim, action_dim) if use_explanatory else None
        
        self.state_history = deque(maxlen=max_delay * 2)
        self.action_history = deque(maxlen=max_delay * 2)
        self.reward_history = deque(maxlen=max_delay * 2)
        self.belief_state = np.zeros(state_dim, dtype=np.float32)
        self.prediction_horizon = 5

    def update_belief(self, state: np.ndarray, uncertainty: float):
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if s.size < self.state_dim:
            s = np.pad(s, (0, self.state_dim - s.size))
        s = s[: self.state_dim]
        self.belief_state = 0.85 * self.belief_state + 0.15 * s
        self.belief_state[-1] = uncertainty

    def predict_next_state(self,state: np.ndarray,action: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.dynamics_model is None:
            return state.copy(), 0.5
        
        self.dynamics_model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            next_state, uncertainty = self.dynamics_model(state_tensor, action_tensor)
            unc = uncertainty.item()
            self.update_belief(next_state.squeeze(0).numpy(), unc)
            return next_state.squeeze(0).numpy(), unc
    
    def predict_reward(self,state: np.ndarray,action: np.ndarray) -> float:
        if self.reward_model is None:
            return 0.0
        
        self.reward_model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            reward = self.reward_model(state_tensor, action_tensor)
            return reward.item()
    
    def estimate_delay(self,state_history: List[np.ndarray],action_history: List[np.ndarray]) -> int:
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
    
    def update_history(self,state: np.ndarray,action: np.ndarray,reward: float):
        self.state_history.append(state.copy())
        self.action_history.append(action.copy())
        self.reward_history.append(reward)
    
    def get_explanation(self,action_sequence: List[np.ndarray],reward_sequence: List[float]) -> Dict[str, Any]:
        if self.explanatory_model is None:
            return {'explanation': 'Explanatory model not enabled.'}
        if not action_sequence:
            return {'explanation': 'No actions to explain.'}
        explanation = self.explanatory_model.get_causal_explanation(action_sequence,reward_sequence)
        return {'explanation': explanation}
    
    def imagine_trajectory(self,initial_state: np.ndarray,action_sequence: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
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
            
                next_state, _ = self.dynamics_model(current_state, action_tensor)
                reward = self.reward_model(current_state, action_tensor)
                
                states.append(next_state.squeeze(0).numpy())
                rewards.append(reward.item())
                current_state = next_state

            return states, rewards
    
    def train_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray, state_histories: Optional[np.ndarray] = None, action_histories: Optional[np.ndarray] = None, true_delays: Optional[np.ndarray] = None) -> Dict[str, float]:
        losses = {}
        if not self.training_mode:
            return losses
        state_tensor = torch.FloatTensor(states)
        action_tensor = torch.FloatTensor(actions)
        next_state_tensor = torch.FloatTensor(next_states)
        reward_tensor = torch.FloatTensor(rewards).reshape(-1, 1)
        if self.dynamics_model is not None:
            self.dynamics_model.train()
            pred_next_state, pred_unc = self.dynamics_model(state_tensor, action_tensor)
            dynamics_loss = nn.MSELoss()(pred_next_state, next_state_tensor)
            target_unc = torch.mean((pred_next_state.detach() - next_state_tensor) ** 2, dim=-1, keepdim=True)
            uncertainty_loss = nn.MSELoss()(pred_unc, torch.clamp(target_unc, 0.0, 1.0))
            dyn_total = dynamics_loss + 0.25 * uncertainty_loss
            self.dynamics_model.optimizer.zero_grad()
            dyn_total.backward()
            self.dynamics_model.optimizer.step()
            losses['dynamics_loss'] = dynamics_loss.item()
            losses['uncertainty_loss'] = uncertainty_loss.item()
        if self.reward_model is not None:
            self.reward_model.train()
            pred_reward = self.reward_model(state_tensor, action_tensor)
            reward_loss = nn.MSELoss()(pred_reward, reward_tensor)
            self.reward_model.optimizer.zero_grad()
            reward_loss.backward()
            self.reward_model.optimizer.step()
            losses['reward_loss'] = reward_loss.item()
        if self.delay_model is not None and state_histories is not None and action_histories is not None and true_delays is not None:
            self.delay_model.train()
            sh = torch.FloatTensor(state_histories)
            ah = torch.FloatTensor(action_histories)
            _, delay_dist, _ = self.delay_model(sh, ah)
            delay_targets = torch.LongTensor(true_delays)
            delay_loss = F.cross_entropy(delay_dist, delay_targets)
            self.delay_model.optimizer.zero_grad()
            delay_loss.backward()
            self.delay_model.optimizer.step()
            losses['delay_loss'] = delay_loss.item()
        if self.explanatory_model is not None and len(states) > 0:
            for i in range(min(8, len(states))):
                self.explanatory_model.update_causal_beliefs(states[i], actions[i], next_states[i], float(rewards[i]))
        return losses

    def train_horizon_rollout(self, initial_state: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        if self.dynamics_model is None or not self.training_mode:
            return 0.0
        self.dynamics_model.train()
        state_t = torch.FloatTensor(initial_state).unsqueeze(0)
        total_loss = 0.0
        for t in range(min(self.prediction_horizon, actions.shape[0], targets.shape[0])):
            act_t = torch.FloatTensor(actions[t]).unsqueeze(0)
            pred, _ = self.dynamics_model(state_t, act_t)
            tgt = torch.FloatTensor(targets[t]).unsqueeze(0)
            total_loss += nn.MSELoss()(pred, tgt)
            state_t = pred.detach()
        self.dynamics_model.optimizer.zero_grad()
        total_loss.backward()
        self.dynamics_model.optimizer.step()
        return float(total_loss.item())
    
    def set_training_mode(self, mode: bool):
        self.training_mode = mode
        
        if self.dynamics_model is not None:
            self.dynamics_model.train(mode)
            if mode and not hasattr(self.dynamics_model, 'optimizer'):
                self.dynamics_model.optimizer = torch.optim.Adam(self.dynamics_model.parameters(),lr=1e-3)
        
        if self.reward_model is not None:
            self.reward_model.train(mode)
            if mode and not hasattr(self.reward_model, 'optimizer'):
                self.reward_model.optimizer = torch.optim.Adam(self.reward_model.parameters(),lr=1e-3)
        
        if self.delay_model is not None:
            self.delay_model.train(mode)
            if mode and not hasattr(self.delay_model, 'optimizer'):
                self.delay_model.optimizer = torch.optim.Adam(self.delay_model.parameters(),lr=1e-3)
    
    def reset(self):
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
    
    def save(self, path: str):
        torch.save({
            'dynamics_model': self.dynamics_model.state_dict() if self.dynamics_model else None,
            'reward_model': self.reward_model.state_dict() if self.reward_model else None,
            'delay_model': self.delay_model.state_dict() if self.delay_model else None,
            'explanatory_model': {'causal_graph': self.explanatory_model.causal_graph, 'factor_importance': self.explanatory_model.factor_importance, 'state_proj': self.explanatory_model.state_proj, 'action_proj': self.explanatory_model.action_proj} if self.explanatory_model else None,
            'belief_state': self.belief_state,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        
        if self.dynamics_model is not None and checkpoint['dynamics_model'] is not None:
            self.dynamics_model.load_state_dict(checkpoint['dynamics_model'])
        
        if self.reward_model is not None and checkpoint['reward_model'] is not None:
            self.reward_model.load_state_dict(checkpoint['reward_model'])
        
        if self.delay_model is not None and checkpoint['delay_model'] is not None:
            self.delay_model.load_state_dict(checkpoint['delay_model'])
        
        if self.explanatory_model is not None and checkpoint.get('explanatory_model') is not None:
            exp = checkpoint['explanatory_model']
            if isinstance(exp, dict):
                self.explanatory_model.causal_graph = exp.get('causal_graph', self.explanatory_model.causal_graph)
                self.explanatory_model.factor_importance = exp.get('factor_importance', self.explanatory_model.factor_importance)
                self.explanatory_model.state_proj = exp.get('state_proj', self.explanatory_model.state_proj)
                self.explanatory_model.action_proj = exp.get('action_proj', self.explanatory_model.action_proj)
            else:
                self.explanatory_model.causal_graph = exp
        if checkpoint.get('belief_state') is not None:
            self.belief_state = checkpoint['belief_state']

    @staticmethod
    def flatten_observation(obs: Dict[str, Any], state_dim: int = 64) -> np.ndarray:
        vec = np.zeros(state_dim, dtype=np.float32)
        cp = obs.get('current_price', [100.0])
        if isinstance(cp, np.ndarray):
            vec[: min(10, state_dim)] = cp[: min(10, state_dim)]
        cash = float(obs.get('agent_cash', 1000.0)) if not isinstance(obs.get('agent_cash'), np.ndarray) else float(obs['agent_cash'])
        vec[10] = cash / 10000.0
        ph = obs.get('price_history', [])
        if isinstance(ph, np.ndarray) and ph.ndim == 2 and ph.shape[0] > 0:
            vec[11:21] = ph[-1, : min(10, ph.shape[1])]
        return vec

    def fit_transitions(self, transitions: List[Dict[str, Any]], epochs: int = 3, batch_size: int = 32) -> Dict[str, float]:
        if not transitions:
            return {}
        self.set_training_mode(True)
        states, actions, next_states, rewards = [], [], [], []
        state_hists, action_hists, delays = [], [], []
        for tr in transitions:
            states.append(tr['state'])
            actions.append(tr['action'])
            next_states.append(tr['next_state'])
            rewards.append(tr['reward'])
            if 'state_history' in tr and 'action_history' in tr and 'delay' in tr:
                state_hists.append(tr['state_history'])
                action_hists.append(tr['action_history'])
                delays.append(tr['delay'])
        states_a, actions_a, next_a, rewards_a = np.array(states), np.array(actions), np.array(next_states), np.array(rewards)
        agg: Dict[str, float] = {}
        n = len(states_a)
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = idx[start : start + batch_size]
                sh = np.array(state_hists)[batch_idx] if state_hists else None
                ah = np.array(action_hists)[batch_idx] if action_hists else None
                td = np.array(delays)[batch_idx] if delays else None
                losses = self.train_step(states_a[batch_idx], actions_a[batch_idx], next_a[batch_idx], rewards_a[batch_idx], sh, ah, td)
                for k, v in losses.items():
                    agg[k] = agg.get(k, 0.0) + v
        return {k: v / max(1, epochs * ((n + batch_size - 1) // batch_size)) for k, v in agg.items()}
