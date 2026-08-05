import polars as pl
import random
from typing import List


class Augmenter:
    """Augments stock price data for RLAIF simulation episodes"""
    
    def __init__(self, df: pl.DataFrame, n_episodes: int, episode_day_len: int, each_day_movements: int):
        """
        Initialize the augmenter
        
        Args:
            df: DataFrame from ingestors containing stock data with 'price' column
            n_episodes: Number of episodes to generate
            episode_day_len: Number of days per episode
            each_day_movements: Number of price movements per day
        """
        self.df = df
        self.n_episodes = n_episodes
        self.episode_day_len = episode_day_len
        self.each_day_movements = each_day_movements
        self.min_data_len = episode_day_len * each_day_movements
        
    def extract_price_data(self) -> pl.Series:
        """
        Extract price column from DataFrame
        Handles NaN values by filling with forward fill then backward fill
        """
        if "price" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'price' column")
        
        prices = self.df["price"]
        prices = prices.fill_null(strategy="forward")
        prices = prices.fill_null(strategy="backward")
        return prices
    
    def compress_data(self, prices: pl.Series) -> pl.Series:
        """
        Adjust data to minimum required length
        - If too much data: compress by sampling
        - If too little data: extrapolate via random self-sampling with volatility transform
        """
        data_list = prices.to_list()
        current_len = len(data_list)
        
        if current_len == self.min_data_len:
            return prices
        
        # Case 1: Too much data - compress by sampling
        if current_len > self.min_data_len:
            sample_rate = current_len // self.min_data_len
            compressed = data_list[::sample_rate][:self.min_data_len]
            return pl.Series(compressed)
        
        # Case 2: Too little data - extrapolate via random self-sampling
        while len(data_list) < self.min_data_len:
            # Pick random locality length (between 5 and 20% of current data, varies each iteration)
            min_percent = random.uniform(0.05, 0.10)
            max_percent = random.uniform(0.15, 0.30)
            min_snippet = max(5, int(current_len * min_percent))
            max_snippet = max(10, int(current_len * max_percent))
            snippet_len = random.randint(min_snippet, max_snippet)
            start_pos = random.randint(0, max(0, current_len - snippet_len))
            end_pos = min(current_len, start_pos + snippet_len)
            snippet = data_list[start_pos:end_pos]
            augmented_snippet = self.apply_volatility_transform(snippet)
            data_list.extend(augmented_snippet)
            current_len = len(data_list)

        # Trim to exact length if we overshot
        if len(data_list) > self.min_data_len:
            data_list = data_list[:self.min_data_len]
        
        return pl.Series(data_list)
    
    def random_day_swap(self, data: List[float]) -> List[float]:
        """
        Apply random day swap with random multiplier
        Example: 1 2 3 4 5 6 7 8 -> 2 1 5 4 3 6 7 8
        """
        swapped = data.copy()
        n = len(swapped)
        
        num_swaps = random.randint(1, n // 4)  # Up to 25% of elements
        
        for _ in range(num_swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            swapped[i], swapped[j] = swapped[j], swapped[i]
        
        for i in range(n):
            if random.random() < 0.1:  # 10% chance to apply multiplier
                multiplier = random.uniform(0.8, 1.2)
                swapped[i] *= multiplier
        
        return swapped
    
    def apply_volatility_transform(self, data: List[float]) -> List[float]:
        """
        Apply random percentage add/sub with volatility jumps
        - Normal: rand% * a where rand is 0.1-0.3
        - Rare jumps: rand jumps to 0.5-0.6 then slowly reduces back to 0.1-0.3
        """
        transformed = []
        current_volatility = random.uniform(0.1, 0.3)
        jump_active = False
        jump_duration = 0
        
        for value in data:
            # Check for rare volatility jump (5% chance)
            if not jump_active and random.random() < 0.05:
                jump_active = True
                current_volatility = random.uniform(0.5, 0.6)
                jump_duration = random.randint(5, 15)  # Jump lasts 5-15 iterations
            
            change_percent = random.uniform(-current_volatility, current_volatility)
            new_value = value * (1 + change_percent)
            transformed.append(new_value)
            if jump_active:
                jump_duration -= 1
                if jump_duration <= 0:
                    jump_active = False
                else:
                    # Slowly reduce back to 0.1-0.3
                    current_volatility = max(0.1, current_volatility - 0.02)
            else:
                # Normal fluctuation in 0.1-0.3 range
                current_volatility = random.uniform(0.1, 0.3)
        return transformed
    
    def generate_episode_data(self, base_data: List[float]) -> List[float]:
        """
        Generate data for a single episode by applying transforms
        """
        swapped = self.random_day_swap(base_data)
        transformed = self.apply_volatility_transform(swapped)
        return transformed
    
    def generate_all_episodes(self) -> List[List[float]]:
        """
        Generate data chunks for all n_episodes
        Each episode builds on the previous one
        """
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


def augment_stock_data(df: pl.DataFrame, n_episodes: int, episode_day_len: int, each_day_movements: int) -> List[List[float]]:
    """
    Main function to augment stock data
    
    Args:
        df: DataFrame from ingestors containing 'price' column
        n_episodes: Number of episodes to generate
        episode_day_len: Number of days per episode
        each_day_movements: Number of price movements per day
        
    Returns:
        List of episode data chunks, each containing price data
    """
    augmenter = Augmenter(df, n_episodes, episode_day_len, each_day_movements)
    return augmenter.generate_all_episodes()
