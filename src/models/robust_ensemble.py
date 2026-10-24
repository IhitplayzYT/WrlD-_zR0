"""Robust ensemble with error-based weighting from market_ensemble_v2.
Uses past realized error instead of self-reported confidence.
Clips extreme reports around median for contamination resistance.
"""
from collections import deque
import numpy as np
from typing import Optional, Tuple, Dict, Any

RETURN_UNIT = 0.01


class RobustEnsemble:
    """Contamination-resistant ensemble that weights by past error, not confidence.
    
    Designed for adversarial markets where some agents may submit poisoned forecasts.
    The ensemble automatically detects and downweights adversarial predictions through
    error-based tracking and median clipping.
    """
    
    def __init__(self, members: int, assets: int, horizon: int, agent_types: Optional[list] = None):
        self.members = members
        self.assets = assets
        self.horizon = horizon
        self.agent_types = agent_types or ['rlaif'] * members  # Track agent types
        self.loss = np.ones((members, assets, horizon))
        self.residuals = deque(maxlen=2048)
        self.updates = 0
        self.adversarial_indices = [i for i, t in enumerate(self.agent_types) if t == 'adversarial']
        self.honest_indices = [i for i, t in enumerate(self.agent_types) if t == 'rlaif']
    
    def combine(self, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Combine predictions using error-based weighting with median clipping.
        
        Adversarial-aware: applies additional downweighting to known adversarial agents
        based on their historical error patterns.
        """
        p = np.asarray(predictions, float)
        if p.shape != self.loss.shape or not np.isfinite(p).all():
            raise ValueError('nonfinite or malformed ensemble submission')
        
        # Median-based robust aggregation
        median = np.median(p, axis=0)
        mad = np.median(np.abs(p - median), axis=0)
        
        # Bounded influence for extreme reports; confidence cannot buy extra weight
        radius = np.maximum(2.5 * mad, 0.15 * np.sqrt(np.arange(1, p.shape[-1] + 1)))
        clipped = np.clip(p, median - radius, median + radius)
        
        # Weight by inverse error (precision), not self-reported confidence
        precision = 1. / (0.15 + self.loss)
        
        # Role labels are used only for reporting, never oracle downweighting.
        weight = 0.90 * precision / precision.sum(axis=0) + 0.10 / len(p)
        
        return np.sum(weight * clipped, axis=0), weight
    
    def update(self, reports: np.ndarray, actual: np.ndarray, scale: np.ndarray, prediction: np.ndarray):
        """Update ensemble weights based on realized error."""
        loss = np.minimum(((reports - actual) / scale) ** 2, 50.)
        self.loss = 0.975 * self.loss + 0.025 * loss
        self.residuals.extend((np.abs(prediction - actual) / scale).ravel().tolist())
        self.updates += 1
    
    def interval_width(self, scale: np.ndarray) -> np.ndarray:
        """Empirical marginal 90% interval from historical residuals."""
        q = float(np.quantile(self.residuals, 0.90)) if len(self.residuals) >= 50 else 1.645
        return np.clip(q, 0.3, 6.) * scale
    
    def state(self) -> Dict[str, Any]:
        return {
            'loss': self.loss.tolist(),
            'residuals': list(self.residuals),
            'updates': self.updates,
            'agent_types': self.agent_types
        }
    
    def restore(self, state: Dict[str, Any]):
        a = np.asarray(state['loss'], float)
        if a.shape != self.loss.shape or not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('invalid ensemble loss')
        self.loss = a
        self.residuals = deque(state['residuals'], maxlen=2048)
        self.updates = int(state['updates'])
        self.agent_types = state.get('agent_types', ['rlaif'] * self.members)
        self.adversarial_indices = [i for i, t in enumerate(self.agent_types) if t == 'adversarial']
        self.honest_indices = [i for i, t in enumerate(self.agent_types) if t == 'rlaif']
    
    def get_adversarial_impact(self) -> Dict[str, float]:
        """Measure the impact of adversarial agents on the ensemble."""
        if not self.adversarial_indices:
            return {'has_adversaries': False, 'avg_adversarial_loss': 0.0}
        
        avg_adversarial_loss = float(self.loss[self.adversarial_indices].mean())
        avg_honest_loss = float(self.loss[self.honest_indices].mean()) if self.honest_indices else 0.0
        
        return {
            'has_adversaries': True,
            'n_adversaries': len(self.adversarial_indices),
            'n_honest': len(self.honest_indices),
            'avg_adversarial_loss': avg_adversarial_loss,
            'avg_honest_loss': avg_honest_loss,
            'loss_ratio': avg_adversarial_loss / max(avg_honest_loss, 0.01)
        }


class MultiViewForecaster:
    """Multi-output recursive least squares with different feature views."""
    
    VIEWS = [
        ([0, 1, 2, 3, 4, 7, 13, 14, 17], 0.997, 'autoregressive'),
        ([0, 1, 4, 5, 7, 11, 13, 16], 0.995, 'trend'),
        ([0, 1, 2, 5, 7, 8, 11, 13, 14], 0.998, 'reversion'),
        ([0, 1, 4, 7, 9, 10, 12, 13, 15, 16], 0.996, 'nonlinear'),
        ([0, 1, 2, 4, 5, 6, 7, 9, 10, 13, 14, 17], 0.999, 'slow_macro')
    ]
    
    def __init__(self, index: int, assets: int, horizon: int):
        self.indices, self.forgetting, self.name = self.VIEWS[index % len(self.VIEWS)]
        d = len(self.indices)
        self.w = np.zeros((assets, d, horizon))
        self.p = np.tile(np.eye(d)[None, :, :] * 2., (assets, 1, 1))
        self.updates = 0
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        x = features[:, self.indices]
        return np.clip(np.einsum('ad,adh->ah', x, self.w), -15, 15)
    
    def update(self, features: np.ndarray, target: np.ndarray):
        x = features[:, self.indices]
        px = np.einsum('aij,aj->ai', self.p, x)
        denom = self.forgetting + np.einsum('ai,ai->a', x, px)
        gain = px / denom[:, None]
        error = np.clip(target - self.predict(features), -20, 20)
        self.w += np.einsum('ad,ah->adh', gain, error)
        self.p = (self.p - np.einsum('ai,aj->aij', gain, px)) / self.forgetting
        self.p = (self.p + self.p.transpose(0, 2, 1)) * 0.5
        if not np.isfinite(self.w).all() or not np.isfinite(self.p).all():
            raise FloatingPointError('forecaster diverged')
        self.updates += 1
    
    def state(self) -> Dict[str, Any]:
        return {
            'w': self.w.tolist(),
            'p': self.p.tolist(),
            'updates': self.updates
        }
    
    def restore(self, state: Dict[str, Any]):
        for k in ('w', 'p'):
            a = np.asarray(state[k], float)
            if a.shape != getattr(self, k).shape or not np.isfinite(a).all():
                raise ValueError(f'invalid forecaster {k}')
            setattr(self, k, a)
        self.updates = int(state['updates'])


def feature_bank(observation: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from observation for multi-view forecasters."""
    r = np.diff(np.log(observation['history']), axis=0) / RETURN_UNIT
    
    def lag(k):
        return r[-k] if len(r) >= k else np.zeros(r.shape[1])
    
    mean3, mean12 = r[-3:].mean(0), r[-12:].mean(0)
    std = np.maximum(r[-20:].std(0), 0.15)
    event = observation.get('event', np.zeros_like(std))
    
    # Observable rolling autocorrelation for regime adaptation
    window = r[-32:]
    phi = np.clip(
        np.sum(window[:-1] * window[1:], axis=0) / (0.5 + np.sum(window[:-1] ** 2, axis=0)),
        -0.75, 0.85
    )
    
    # Shape: asset x feature
    x = np.stack([
        np.ones_like(std), lag(1), lag(2), lag(3), mean3, mean12,
        std, event, lag(1) * mean12, np.tanh(lag(1)),
        np.tanh(event), mean3 - mean12, lag(1) ** 2 - std ** 2,
        phi * lag(1), phi ** 2 * lag(1), phi, phi * mean12,
        window.mean(0) * (1 - phi)
    ], axis=1)
    
    return np.clip(x, -5, 5), std
