from __future__ import annotations

import numpy as np
import polars as pl


regimes = {"Bull": {"drift": 0.0004,"vol": 0.006},"Bear": {"drift": -0.0004,"vol": 0.012},"Sideways": {"drift": 0.0,"vol": 0.003},"Crash": {"drift": -0.003,"vol": 0.05},"Recovery": {"drift": 0.001,"vol": 0.02}}
transition = {"Bull": {"Bull": 0.992,"Sideways": 0.006,"Bear": 0.002},"Bear": {"Bear": 0.992,"Recovery": 0.006,"Crash": 0.002},"Sideways": {"Sideways": 0.992,"Bull": 0.004,"Bear": 0.004},"Crash": {"Crash": 0.97,"Recovery": 0.03},"Recovery": {"Recovery": 0.985,"Bull": 0.015}}


def generate_stock(n: int,start_price: float = 100.0,seed: int | None = None) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    # market regimes
    nonlocal regimes,transition
    state = "Sideways"
    prices = np.empty(n)
    returns = np.zeros(n)
    volumes = np.empty(n)
    regime_col = [] # History of transitions 

    # Init the Loop
    prices[0] = start_price
    prev_return = 0.0
    fair_price = start_price
    prev_var = 0.0
    trend = 0.0

    for t in range(1, n):
        probs = transition[state] # Current Markov transition
        states,p = list(probs.keys()),list(probs.values()) # Next Markov Jump
        state = rng.choice(states, p=p) # Random choice
        regime_col.append(state)
        drift,base_vol = regimes[state]["drift"],regimes[state]["vol"]

        trend = 0.995 * trend + 0.005 * drift
        fair_price *= np.exp(drift)

        # Mean reversion
        deviation = np.log(prices[t - 1] / fair_price)
        mean_reversion = -0.04 * deviation

        # Volatility clustering
        current_vol = (0.94 * base_vol+ 0.06 * abs(prev_return))
        volatility_noise = rng.normal(0,current_vol)

        # Random Jump
        jump = 0.0
        if rng.random() < 0.002:
            jump = rng.normal(0,4 * current_vol)

        seasonal = (0.0005 * np.sin(2 * np.pi * t / 250))

        # Total Return
        r = trend + mean_reversion + volatility_noise + jump + seasonal

        returns[t] = prev_return = r
        prices[t] = prices[t - 1] * np.exp(r)
        volumes[t] = max(0,1000 + abs(r) * 200000 + rng.normal(0, 100))
        regime_col.insert(0, state)
        volumes[0] = volumes[1]
    return pl.DataFrame({"t": np.arange(n),"price": prices,"return": returns,"volume": volumes.astype(int),"regime": regime_col,})


def generate_stock(n: int,start_price: float = 100.0,seed: int | None = None,) -> pl.DataFrame:
    if n <= 0:
        raise ValueError("n must be > 0")
    if start_price <= 0:
        raise ValueError("start_price must be > 0")
    rng = np.random.default_rng(seed)
    states = tuple(regimes.keys())
    prices = np.empty(n, dtype=np.float64)
    returns = np.zeros(n, dtype=np.float64)
    volumes = np.empty(n, dtype=np.float64)
    regime_col = np.empty(n, dtype=object)
    state = "Sideways"
    prices[0] = start_price
    volumes[0] = 1000.0
    regime_col[0] = state
    prev_return = 0.0
    # Start variance close to the initial regime volatility
    base_vol = regimes[state]["vol"]
    prev_var = base_vol ** 2
    
    # Dynamic trend and fair value
    trend = 0.0
    fair_price = start_price

    # Volatility persistence
    volatility_decay = 0.94
    shock_weight = 1.0 - volatility_decay

    # Mean-reversion strength
    mean_reversion_strength = 0.04

    # Volume parameters
    base_volume = 1000.0
    volume_return_sensitivity = 20.0
    volume_noise = 0.10

    for t in range(1, n):
        # Markov transition
        probs = transition[state]
        next_states = tuple(probs.keys())
        p = np.asarray(tuple(probs.values()), dtype=np.float64)

        state = rng.choice(next_states, p=p)
        regime_col[t] = state
        drift = regimes[state]["drift"]
        base_vol = regimes[state]["vol"]

        # 2. Persistent trend
        trend_decay = 0.995
        trend = trend_decay * trend + (1.0 - trend_decay) * drift

        # 3. Dynamic fair price
        fair_price *= np.exp(drift)

        # 4. Mean reversion
        deviation = np.log(prices[t - 1] / fair_price)

        mean_reversion = -mean_reversion_strength * deviation

        # 5. Volatility clustering
        # EWMA:
        # sigma²_t = lambda * sigma²_(t-1) + (1-lambda) * r²_(t-1)
        shock_var = prev_return ** 2

        clustered_var = ( volatility_decay * prev_var + shock_weight * shock_var)

        # Regime provides the long-term volatility level.
        current_var = (0.8 * clustered_var+ 0.2 * base_vol ** 2)

        current_vol = np.sqrt(max(current_var, 1e-12))

        volatility_noise = rng.normal(0.0,current_vol)

        # ---------------------------------------------------------
        # 6. Regime-dependent jumps
        # ---------------------------------------------------------


        # Used to decide how hard is it to jump from specific state 
        jump_probability = {
            "Bull": 0.001,
            "Bear": 0.003,
            "Sideways": 0.0005,
            "Crash": 0.015,
            "Recovery": 0.005,
        }.get(state, 0.002)

    
        jump = 0.0
        if rng.random() < jump_probability:
            jump = rng.normal(0.0,4.0 * current_vol)

        # ---------------------------------------------------------
        # 7. Seasonal component
        # ---------------------------------------------------------
        seasonal = 0.0005 * np.sin(2.0 * np.pi * t / 250.0)

        # ---------------------------------------------------------
        # 8. Total log return
        # ---------------------------------------------------------
        r = trend+ mean_reversion+ volatility_noise+ jump+ seasonal

        returns[t] = r
        prev_return = r
        prev_var = current_var
        prices[t] = prices[t - 1] * np.exp(r)

        # ---------------------------------------------------------
        # 10. Volume
        #
        # Lognormal structure keeps volume positive and naturally
        # produces right-skewed volume.
        # ---------------------------------------------------------
        volume_log = np.log(base_volume)+ volume_return_sensitivity * abs(r)+ 5.0 * current_vol+ rng.normal(0.0, volume_noise)
        volumes[t] = np.exp(volume_log)
    return pl.DataFrame({"t": np.arange(n, dtype=np.int64),"price": prices,"return": returns,"volume": volumes.astype(np.int64),"regime": regime_col})
