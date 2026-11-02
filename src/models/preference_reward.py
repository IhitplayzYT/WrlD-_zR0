""" Pairwise preference reward comparisons with synthetic teacher utility """
import numpy as np
from typing import Dict, Any


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    p = np.exp(np.clip(x, -60, 0))
    return p / p.sum(axis=-1, keepdims=True)


class PreferenceReward:
    def __init__(self):
        self.weights = np.zeros(4)
        self.updates = 0
    
    def score(self, features: np.ndarray) -> float: return float(np.tanh(np.asarray(features) @ self.weights / 2.))
    
    def fit_pairs(self, preferred: np.ndarray, rejected: np.ndarray, learning_rate: float = 0.03):
        delta = np.asarray(preferred) - np.asarray(rejected)
        delta = np.atleast_2d(delta)
        if delta.shape[1] != 4 or not np.isfinite(delta).all(): raise ValueError('preference pairs need four finite features')
        logits = np.clip(delta @ self.weights, -30, 30)
        grad = ((1. / (1. + np.exp(logits)))[:, None] * delta).mean(0)
        self.weights += learning_rate * (grad - 0.001 * self.weights)
        self.updates += len(delta)
    
    @staticmethod
    def features(weights: np.ndarray, realized_returns: np.ndarray, previous_weights: np.ndarray, cost_rate: float) -> np.ndarray:
        """Extract features from portfolio action and outcome.
        Features: [net_return, downside, turnover, concentration]
        All negative except net_return.
        """
        turnover = float(np.abs(weights - previous_weights).sum())
        net = float(weights @ np.expm1(realized_returns)) - cost_rate * turnover
        concentration = float(weights @ weights)
    # net return (can be positive or negative)
    # downside (negative)
    # turnover penalty (negative)
    # concentration penalty (negative)
        return np.array([np.clip(net / 0.01, -5, 5),-max(0., -net) / 0.01,-turnover,-concentration])
    
    @staticmethod
    def teacher_utility(features: np.ndarray) -> float: return float(np.asarray(features) @ np.array([1., 0.6, 0.08, 0.04]))
    
    def state(self) -> Dict[str, Any]: return {'weights': self.weights.tolist(), 'updates': self.updates}
    
    def restore(self, state: Dict[str, Any]):
        self.weights = np.asarray(state['weights'], float)
        if self.weights.shape != (4,) or not np.isfinite(self.weights).all(): raise ValueError('invalid preference weights')
        self.updates = int(state['updates'])
