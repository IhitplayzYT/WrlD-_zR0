from __future__ import annotations
import numpy as np
import polars as pl


regimes = {
    "Bull": {"drift": 0.0004, "vol": 0.006},
    "Bear": {"drift": -0.0004, "vol": 0.012},
    "Sideways": {"drift": 0.0, "vol": 0.003},
    "Crash": {"drift": -0.003, "vol": 0.05},
    "Recovery": {"drift": 0.001, "vol": 0.02}
}

transition = {
    "Bull": {"Bull": 0.992, "Sideways": 0.006, "Bear": 0.002},
    "Bear": {"Bear": 0.992, "Recovery": 0.006, "Crash": 0.002},
    "Sideways": {"Sideways": 0.992, "Bull": 0.004, "Bear": 0.004},
    "Crash": {"Crash": 0.97, "Recovery": 0.03},
    "Recovery": {"Recovery": 0.985, "Bull": 0.015}
}

# Generate realistic synthetic market data following market stats
def generate_stock(n: int, start_price: float = 100.0, seed: int | None = None, vol: float = 1000.0) -> pl.DataFrame:
    if n <= 0 or start_price <= 0:
        raise ValueError("n and start_price must be > 0")
    rng = np.random.default_rng(seed)
    prices = np.empty(n, dtype=np.float64)
    returns = np.empty(n, dtype=np.float64)
    volumes = np.empty(n, dtype=np.float64)
    regime_col = np.empty(n, dtype=object)

    prices[0] = fair_price = start_price
    returns[0] = 0.0
    state = "Sideways"
    regime_col[0] = state
    base_volume = vol if vol else 1000.0
    volumes[0] = base_volume
    prev_return = trend = 0.0
    base_vol = regimes[state]["vol"]
    prev_var = base_vol ** 2

    volatility_decay = 0.94
    shock_weight = 1.0 - volatility_decay

    mean_reversion_strength = 0.04
    volume_return_sensitivity = 20.0
    volume_noise = 0.10

    for t in range(1, n):
        probs = transition[state]
        n_states = tuple(probs.keys())
        p = np.asarray(tuple(probs.values()), dtype=np.float64)
        state = str(rng.choice(n_states, p=p))
        regime_col[t] = state
        drift, base_vol = regimes[state]["drift"], regimes[state]["vol"]
        trend_decay = 0.995
        trend = trend_decay * trend + (1.0 - trend_decay) * drift
        fair_price *= np.exp(drift)

        deviation = np.log(max(prices[t - 1], 1e-6) / max(fair_price, 1e-6))
        mean_reversion = -mean_reversion_strength * deviation

        clustered_var = volatility_decay * prev_var + shock_weight * (prev_return ** 2)

        current_var = (0.8 * clustered_var + 0.2 * base_vol ** 2)
        current_vol = np.sqrt(max(current_var, 1e-12))
        volatility_noise = float(rng.normal(0.0, current_vol))
        jump_probability = {"Bull": 0.001, "Bear": 0.003, "Sideways": 0.0005, "Crash": 0.015, "Recovery": 0.005}.get(state, 0.002)

        jump = 0.0
        if rng.random() < jump_probability:
            jump = float(rng.normal(0.0, 4.0 * current_vol))

        seasonal = 0.0005 * np.sin(2.0 * np.pi * t / 250.0)

        r = trend + mean_reversion + volatility_noise + jump + seasonal

        returns[t] = prev_return = r
        prev_var = current_var
        prices[t] = max(1.0, prices[t - 1] * np.exp(r))
        volumes[t] = np.exp(np.log(base_volume) + volume_return_sensitivity * abs(r) + 5.0 * current_vol + rng.normal(0.0, volume_noise))

    return pl.DataFrame({"t": np.arange(n, dtype=np.int64),"price": prices,"return": returns,"volume": volumes.astype(np.int64),"regime": regime_col})

def fetch_aapl_data(start_date: str = "2016-01-01", end_date: str = "2022-12-31") -> pl.DataFrame:
    import yfinance as yf
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(start=start_date, end=end_date)
        if hist.empty:
            raise ValueError("No data in yfinance for AAPL")
        df = pl.DataFrame({"price": hist['Close'].values, "date": hist.index.values})
        return df
    except Exception as e:
        raise RuntimeError("AAPL download failed; choose synthetic data explicitly") from e
