"""Chronological market windows: preserve ordering and observed return scales.

Never shuffle price levels, compound augmentation between episodes, or silently
invent rows to make a short dataset fit an experiment.
"""
import numpy as np
import polars as pl


class Augmenter:
    def __init__(self, df, n_episodes, episode_day_len, each_day_movements):
        self.df = df
        self.n_episodes = n_episodes
        self.min_data_len = episode_day_len * each_day_movements + 1
        if n_episodes < 1 or self.min_data_len < 2:
            raise ValueError('Positive episode count and length required')

    def extract_price_data(self):
        if 'price' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'price' column")
        values = np.asarray(self.df['price'].to_list(), dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Prices must be finite, positive and nonmissing')
        return pl.Series('price', values)

    def compress_data(self, prices):
        if len(prices) < self.min_data_len:
            raise ValueError(f'Need {self.min_data_len} observed prices; got {len(prices)}')
        return prices.head(self.min_data_len)

    def generate_all_episodes(self):
        prices = self.extract_price_data().to_list()
        self.compress_data(pl.Series(prices))
        starts = np.linspace(0, len(prices)-self.min_data_len, self.n_episodes).astype(int)
        return [prices[i:i+self.min_data_len] for i in starts]


def augment_stock_data(df, n_episodes, episode_day_len, each_day_movements):
    return Augmenter(df, n_episodes, episode_day_len, each_day_movements).generate_all_episodes()
