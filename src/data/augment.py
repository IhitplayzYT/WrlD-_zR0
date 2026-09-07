import polars as pl
import random
from typing import List


class Augmenter:
    """Augments stock price data for RLAIF simulation episodes"""
    
    def __init__(self, df: pl.DataFrame, n_episodes: int, episode_day_len: int, each_day_movements: int):
        """ Note : df: DataFrame from ingestors containing stock data with [VIMP] 'price' column """
        self.df = df
        self.n_episodes = n_episodes
        self.episode_day_len = episode_day_len
        self.each_day_movements = each_day_movements
        self.min_data_len = episode_day_len * each_day_movements
        
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
        if cur_l == self.min_data_len:
            return prices
        
        # Compress by sampling
        if cur_l > self.min_data_len:
            compr_rate = cur_l // self.min_data_len
            ret = data_list[::compr_rate][:self.min_data_len]
            return pl.Series(ret)
        
        # Case 2: Too little data - extrapolate via random self-sampling
        while len(data_list) < self.min_data_len:
            # Pick random locality length (between 5 and 20% of current data, varies each iteration)
            mn_perc,mx_perc = random.uniform(0.05, 0.10),random.uniform(0.15, 0.30)
            mn_slice,mx_slice = max(5, int(cur_l * mn_perc)),max(10, int(cur_l * mx_perc))    
            rand_l = random.randint(mn_slice, mx_slice)
            strt,ed = random.randint(0, max(0, cur_l - rand_l)),min(cur_l, strt + rand_l)
            augmented_snippet = self.apply_volatility_transform(data_list[strt:ed])
            data_list.extend(augmented_snippet)
            cur_l = len(data_list)

        if len(data_list) > self.min_data_len:
            data_list = data_list[:self.min_data_len]
        
        return pl.Series(data_list)
    
    def random_day_swap(self, data: List[float]) -> List[float]:
        """ Apply random day swap with random multiplier to each value """
        swapped = data.copy()
        n = len(swapped)
        num_swaps = random.randint(1, n // 4)  # 25% of elems are gonna be swapped
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            swapped[i], swapped[j] = swapped[j], swapped[i]
        for i in range(n):
            if random.random() < 0.1:  # 10% chance for multiplier
                multiplier = random.uniform(0.8, 1.2)
                swapped[i] *= multiplier
        return swapped
    
    def apply_volatility_transform(self, data: List[float]) -> List[float]:
        """
        Apply random percentage add/sub with varying volatility and has rare high volatility spikes
        - rand is 0.1-0.3 and rare rand jumps to 0.5-0.6 then slowly back to 0.1-0.3
        """
        transformed = []
        current_volatility = random.uniform(0.1, 0.3)
        jump_active = False
        jump_duration = 0        
        for value in data:
            if not jump_active and random.random() < 0.05:
                jump_active = True
                current_volatility = random.uniform(0.5, 0.6)
                jump_duration = random.randint(5, 15) 
            
            change_percent = random.uniform(-current_volatility, current_volatility)
            new_value = value * (1 + change_percent)
            transformed.append(new_value)
            if jump_active:
                jump_duration -= 1
                if jump_duration <= 0:
                    jump_active = False
                else:
                    current_volatility = max(0.1, current_volatility - 0.02)
            else:
                current_volatility = random.uniform(0.1, 0.3)
        return transformed
    
    def generate_episode_data(self, base_data: List[float]) -> List[float]:
        """ Generate data for a single episode by applying transforms """
        swapped = self.random_day_swap(base_data)
        transformed = self.apply_volatility_transform(swapped)
        return transformed
    
    def generate_all_episodes(self) -> List[List[float]]:
        """ Generate data chunks for all episodes,each uses the previous one for being built """
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

    def random_episode_swap(self,df: List[List[float]]) -> List[List[float]]:
        l = len(df)
        n_swap = random.randint(1,l//4)
        ret = df.copy()
        for _ in range(n_swap):
            i = random.randint(0,l)
            j = random.randint(0,l)
            ret[i],ret[j] = ret[j],ret[i]
        return ret

        
    def augment_stock_data(self,df: pl.DataFrame, n_episodes: int, episode_day_len: int, each_day_movements: int) -> List[List[float]]:
        """ Main function to augment stock data """
        augmenter = Augmenter(df, n_episodes, episode_day_len, each_day_movements)
        return self.random_episode_swap(augmenter.generate_all_episodes())
