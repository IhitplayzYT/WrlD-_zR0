import polars as pl
import random
from typing import List


class Augmenter:
    """Augments stock price data for RLAIF simulation episodes"""
    
    def __init__(self, df: pl.DataFrame, n_episodes: int, episode_day_len: int, each_day_movements: int):
        """ Note : df: DataFrame from ingestors containing stock data with compulsory 'price' column """
        self.df = df
        self.n_episodes = max(1, n_episodes)
        self.episode_day_len = max(1, episode_day_len)
        self.each_day_movements = max(1, each_day_movements)
        self.min_data_len = self.episode_day_len * self.each_day_movements
        
    def extract_price_data(self) -> pl.Series:
        if "price" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'price' column")
        
        prices = self.df["price"]
        prices = prices.fill_null(strategy="forward")
        prices = prices.fill_null(strategy="backward")
        return prices
    
    def compress_data(self, prices: pl.Series) -> pl.Series:
        """
        - If too much data: compress by sampling
        - If too little data: extrapolate by random sampling and random volatility change
        """
        data_list = prices.to_list()
        cur_l = len(data_list)
        if cur_l == 0:
            data_list = [100.0] * self.min_data_len
            return pl.Series(data_list)
        if cur_l == self.min_data_len:
            return prices
        
        # Compress by sampling
        if cur_l > self.min_data_len:
            compr_rate = max(1, cur_l // self.min_data_len)
            ret = data_list[::compr_rate][:self.min_data_len]
            if len(ret) < self.min_data_len:
                ret.extend([ret[-1]] * (self.min_data_len - len(ret)))
            return pl.Series(ret)
        
        # Too little data, extrapolate via rand sampling
        while len(data_list) < self.min_data_len:
            cur_l = len(data_list)
            mn_perc, mx_perc = random.uniform(0.05, 0.10), random.uniform(0.15, 0.30)
            mn_slice, mx_slice = max(5, int(cur_l * mn_perc)), max(10, int(cur_l * mx_perc))    
            rand_l = random.randint(mn_slice, max(mn_slice, mx_slice))
            strt = random.randint(0, max(0, cur_l - rand_l))
            ed = min(cur_l, strt + rand_l)
            augmented_snippet = self.apply_volatility_transform(data_list[strt:ed])
            data_list.extend(augmented_snippet)
            if len(data_list) == cur_l:
                data_list.append(data_list[-1])

        if len(data_list) > self.min_data_len:
            data_list = data_list[:self.min_data_len]
        
        return pl.Series(data_list)
    
    def apply_volatility_transform(self, data: List[float]) -> List[float]:
        transformed = []
        current_volatility = random.uniform(0.01, 0.05)
        jump_active = False
        jump_duration = 0        
        for value in data:
            if not jump_active and random.random() < 0.05:
                jump_active = True
                current_volatility = random.uniform(0.08, 0.15)
                jump_duration = random.randint(5, 15) 
            
            change_percent = random.uniform(-current_volatility, current_volatility)
            new_value = max(1.0, value * (1 + change_percent))
            transformed.append(new_value)
            if jump_active:
                jump_duration -= 1
                if jump_duration <= 0:
                    jump_active = False
                else:
                    current_volatility = max(0.01, current_volatility - 0.01)
            else:
                current_volatility = random.uniform(0.01, 0.05)
        return transformed
    
    def generate_episode_data(self, base_data: List[float]) -> List[float]:
        swapped = self.random_day_swap(base_data)
        transformed = self.apply_volatility_transform(swapped)
        return transformed
    
    def generate_all_episodes(self) -> List[List[float]]:
        prices = self.extract_price_data()
        prices = self.compress_data(prices)
        base_data = prices.to_list()
        episodes = []
        current_data = base_data
        for episode_num in range(self.n_episodes):
            episode_data = self.generate_episode_data(current_data)
            episodes.append(episode_data)
            current_data = episode_data
        return episodes

    def random_day_swap(self, data: List[float]) -> List[float]:
        swapped = data.copy()
        n = len(swapped)
        if n <= 1:
            return swapped
        num_swaps = random.randint(1, max(1, n // 4))
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            swapped[i], swapped[j] = swapped[j], swapped[i]
        for i in range(n):
            if random.random() < 0.1:
                multiplier = random.uniform(0.95, 1.05)
                swapped[i] = max(1.0, swapped[i] * multiplier)
        return swapped

    def random_episode_swap(self, df: List[List[float]]) -> List[List[float]]:
        l = len(df)
        if l <= 1:
            return df
        n_swap = random.randint(1, max(1, l // 4))
        ret = [ep.copy() for ep in df]
        for _ in range(n_swap):
            i = random.randint(0, l - 1)
            j = random.randint(0, l - 1)
            ret[i], ret[j] = ret[j], ret[i]
        return ret


def augment_stock_data(df: pl.DataFrame, n_episodes: int, episode_day_len: int, each_day_movements: int) -> List[List[float]]:
    """ Top-level function API to augment stock data """
    augmenter = Augmenter(df, n_episodes, episode_day_len, each_day_movements)
    return augmenter.random_episode_swap(augmenter.generate_all_episodes())
