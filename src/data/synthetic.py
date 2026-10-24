"""Seeded regime-switching synthetic prices; one tick is a simulation movement.

These parameters are illustrative, not fitted to a real exchange. No regime
labels or future prices are exposed in agent observations.
"""
import numpy as np


def generate_market_prices(n_steps, seed=0, n_assets=1, volatility=.006):
    if n_steps < 2 or not 1 <= n_assets <= 10 or volatility <= 0:
        raise ValueError('Invalid synthetic data configuration')
    rng = np.random.default_rng(seed)
    returns = np.zeros((n_steps-1, n_assets))
    regimes = np.zeros(n_steps-1, dtype=int)
    # Symmetric bull/bear drift prevents making profit a property of the generator.
    drifts = np.array([0., .0004, -.0004])
    state = 0
    variance = volatility**2
    for t in range(n_steps-1):
        if rng.random() < .025:
            state = int(rng.integers(3))
        regimes[t] = state
        common = rng.normal()
        shock = np.sqrt(.4)*common + np.sqrt(.6)*rng.normal(size=n_assets)
        variance = .94*variance + .06*(volatility**2 if t == 0 else np.mean(returns[t-1]**2))
        sigma = np.sqrt(.7*variance + .3*volatility**2)
        returns[t] = drifts[state] - .5*sigma**2 + sigma*shock
    prices = 100*np.exp(np.vstack([np.zeros(n_assets), np.cumsum(returns, axis=0)]))
    return {f'ASSET{i}': prices[:, i].tolist() for i in range(n_assets)}
