from __future__ import annotations

import numpy as np
import polars as pl


def generate_stock(n: int,start_price: float = 100.0,seed: int | None = None) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    # -------------------------------
    # Hidden market regimes
    # -------------------------------

    regimes = {
        "Bull": {
            "drift": 0.0004,
            "vol": 0.006,
        },
        "Bear": {
            "drift": -0.0004,
            "vol": 0.012,
        },
        "Sideways": {
            "drift": 0.0,
            "vol": 0.003,
        },
        "Crash": {
            "drift": -0.003,
            "vol": 0.05,
        },
        "Recovery": {
            "drift": 0.001,
            "vol": 0.02,
        },
    }

    transition = {
        "Bull": {
            "Bull": 0.992,
            "Sideways": 0.006,
            "Bear": 0.002,
        },
        "Bear": {
            "Bear": 0.992,
            "Recovery": 0.006,
            "Crash": 0.002,
        },
        "Sideways": {
            "Sideways": 0.992,
            "Bull": 0.004,
            "Bear": 0.004,
        },
        "Crash": {
            "Crash": 0.97,
            "Recovery": 0.03,
        },
        "Recovery": {
            "Recovery": 0.985,
            "Bull": 0.015,
        },
    }

    state = "Sideways"

    prices = np.empty(n)
    returns = np.zeros(n)
    volumes = np.empty(n)
    regime_col = []

    prices[0] = start_price

    prev_return = 0.0

    fair_price = start_price

    trend = 0.0

    for t in range(1, n):

        # -----------------------
        # Transition regime
        # -----------------------

        probs = transition[state]

        states = list(probs.keys())
        p = list(probs.values())
        state = rng.choice(states, p=p)
        regime_col.append(state)
        drift = regimes[state]["drift"]
        base_vol = regimes[state]["vol"]

        # -----------------------
        # Slowly changing trend
        # -----------------------

        trend = 0.995 * trend + 0.005 * drift

        # -----------------------
        # Fair value
        # -----------------------

        fair_price *= np.exp(drift)

        # -----------------------
        # Mean reversion
        # -----------------------

        deviation = np.log(prices[t - 1] / fair_price)
        mean_reversion = -0.04 * deviation

        # -----------------------
        # Volatility clustering
        # -----------------------

        current_vol = (0.94 * base_vol+ 0.06 * abs(prev_return))
        volatility_noise = rng.normal(0,current_vol)

        # -----------------------
        # Rare jumps
        # -----------------------

        jump = 0.0
        if rng.random() < 0.002:
            jump = rng.normal(0,4 * current_vol)

        # -----------------------
        # Seasonality
        # -----------------------

        seasonal = (0.0005 * np.sin(2 * np.pi * t / 250))

        # -----------------------
        # Total return
        # -----------------------

        r = trend + mean_reversion + volatility_noise + jump + seasonal

        returns[t] = r
        prices[t] = prices[t - 1] * np.exp(r)
        prev_return = r
        volumes[t] = 1000 + abs(r) * 200000+ rng.normal(0, 100)
    regime_col.insert(0, state)
    volumes[0] = volumes[1]
    return pl.DataFrame(
        {
            "t": np.arange(n),
            "price": prices,
            "return": returns,
            "volume": volumes.astype(int),
            "regime": regime_col,
        }
    )