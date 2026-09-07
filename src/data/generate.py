from __future__ import annotations

import numpy as np
import polars as pl
import yfinance as yf


regimes = {"Bull": {"drift": 0.0004,"vol": 0.006},"Bear": {"drift": -0.0004,"vol": 0.012},"Sideways": {"drift": 0.0,"vol": 0.003},"Crash": {"drift": -0.003,"vol": 0.05},"Recovery": {"drift": 0.001,"vol": 0.02}}
transition = {"Bull": {"Bull": 0.992,"Sideways": 0.006,"Bear": 0.002},"Bear": {"Bear": 0.992,"Recovery": 0.006,"Crash": 0.002},"Sideways": {"Sideways": 0.992,"Bull": 0.004,"Bear": 0.004},"Crash": {"Crash": 0.97,"Recovery": 0.03},"Recovery": {"Recovery": 0.985,"Bull": 0.015}}

def generate_stock(n: int,start_price: float = 100.0,seed: int | None = None,) -> pl.DataFrame:
    if n <= 0 or start_price <= 0:
        raise ValueError(f"{'n' if not n else 'start_price'} must be > 0")

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

    base_vol = regimes[state]["vol"]
    prev_var = base_vol ** 2
    trend = 0.0
    fair_price = start_price

    # Volatility persistence
    volatility_decay = 0.94
    shock_weight = 1.0 - volatility_decay

    # Mean-reversion strength
    mean_reversion_strength = 0.04

    base_volume = 1000.0
    volume_return_sensitivity = 20.0
    volume_noise = 0.10

    for t in range(1, n):
        # MDP
        probs = transition[state]
        n_states = tuple(probs.keys())
        p = np.asarray(tuple(probs.values()), dtype=np.float64)
        state = rng.choice(n_states, p=p)
        regime_col[t] = state
        drift, base_vol = regimes[state]["drift"],regimes[state]["vol"]

        trend_decay = 0.995
        trend = trend_decay * trend + (1.0 - trend_decay) * drift

        fair_price *= np.exp(drift)

        # 4. Mean reversion
        deviation = np.log(prices[t - 1] / fair_price)
        mean_reversion = -mean_reversion_strength * deviation

        # 5. Volatility clustering
        # lambda * sigma²_(t-1) + (1-lambda) * r²_(t-1)
        clustered_var =  volatility_decay * prev_var + shock_weight * (prev_return ** 2)

        current_var = (0.8 * clustered_var+ 0.2 * base_vol ** 2)
        current_vol = np.sqrt(max(current_var, 1e-12))
        volatility_noise = rng.normal(0.0,current_vol)

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


        seasonal = 0.0005 * np.sin(2.0 * np.pi * t / 250.0)

        r = trend + mean_reversion + volatility_noise + jump + seasonal

        returns[t] = r
        prev_return = r
        prev_var = current_var
        prices[t] = prices[t - 1] * np.exp(r)

        volume_log = np.log(base_volume)+ volume_return_sensitivity * abs(r)+ 5.0 * current_vol+ rng.normal(0.0, volume_noise)
        volumes[t] = np.exp(volume_log)
    return pl.DataFrame({"t": np.arange(n, dtype=np.int64),"price": prices,"return": returns,"volume": volumes.astype(np.int64),"regime": regime_col})


def fetch_aapl_data(start_date: str = "2016-01-01", end_date: str = "2022-12-31") -> pl.DataFrame:
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(start=start_date, end=end_date)
        if hist.empty:
            raise ValueError("No data returned from Yahoo Finance for AAPL")
        prices = hist['Close'].values
        df = pl.DataFrame({
            "price": prices,
            "date": hist.index.values
        })
        print(f"Fetched AAPL stock data From: {start_date} To: {end_date}")
        return df
    except Exception as e:
        print(f"Error fetching AAPL data: {e}")
        return generate_stock(1500, start_price=100.0)
