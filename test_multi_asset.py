#!/usr/bin/env python3
"""Test script for multi-asset integration"""

import numpy as np
from src.market.multi_asset_order_book import MultiAssetOrderBook
from src.market.order_book import OrderSide
from src.environment.market_env import MarketEnvironment

def test_multi_asset_order_book():
    """Test MultiAssetOrderBook functionality"""
    print("Testing MultiAssetOrderBook...")
    
    assets = ["AAPL", "MSFT", "GOOGL"]
    maob = MultiAssetOrderBook(assets)
    
    # Add orders for different assets
    maob.add_order("agent_0", "AAPL", OrderSide.BUY, 150.0, 10)
    maob.add_order("agent_1", "MSFT", OrderSide.SELL, 300.0, 5)
    maob.add_order("agent_2", "GOOGL", OrderSide.BUY, 2500.0, 2)
    
    # Test by index
    maob.add_order_by_idx("agent_3", 0, OrderSide.SELL, 151.0, 5)
    
    # Get states
    aapl_state = maob.get_order_book_state("AAPL")
    msft_state = maob.get_order_book_state("MSFT")
    googl_state = maob.get_order_book_state("GOOGL")
    
    print(f"AAPL state: {aapl_state}")
    print(f"MSFT state: {msft_state}")
    print(f"GOOGL state: {googl_state}")
    
    # Get aggregated state
    agg_state = maob.get_aggregated_market_state()
    print(f"Aggregated state: {agg_state}")
    
    print("MultiAssetOrderBook test passed!\n")

def test_multi_asset_environment():
    """Test MarketEnvironment with multiple assets"""
    print("Testing MarketEnvironment with multiple assets...")
    
    # Create price data for multiple assets
    price_data = {
        "AAPL": [100.0 + i * 0.1 for i in range(500)],
        "MSFT": [200.0 + i * 0.15 for i in range(500)],
        "GOOGL": [150.0 + i * 0.12 for i in range(500)]
    }
    
    env = MarketEnvironment(
        n_agents=2,
        max_bad_agents=1,
        initial_cash=1000.0,
        episode_length=50,
        price_data=price_data,
        black_box_duration=5,
        black_box_frequency=20
    )
    
    print(f"Available assets: {env.available_assets}")
    print(f"Order book type: {type(env.order_book)}")
    
    # Reset environment
    obs, info = env.reset()
    
    print(f"Observation keys: {obs['agent_0'].keys()}")
    print(f"Price history shape: {obs['agent_0']['price_history'].shape}")
    print(f"Order book bids shape: {obs['agent_0']['order_book_bids'].shape}")
    print(f"Order book asks shape: {obs['agent_0']['order_book_asks'].shape}")
    print(f"Current price shape: {obs['agent_0']['current_price'].shape}")
    
    # Take a few steps
    for step in range(5):
        action = np.array([1, 0, 10, 100.0], dtype=np.float32)  # Buy AAPL
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Reward={reward:.2f}")
        
        if terminated or truncated:
            break
    
    print("MarketEnvironment multi-asset test passed!\n")

def test_adversarial_agent_multi_asset():
    """Test AdversarialAgent with multi-asset actions"""
    print("Testing AdversarialAgent with multi-asset actions...")
    
    from src.agents.adversarial_agent import AdversarialAgent
    
    agent = AdversarialAgent("test_adv", initial_cash=1000.0)
    
    # Create multi-asset observation
    dummy_obs = {
        'price_history': np.random.rand(50, 10).astype(np.float32),
        'order_book_bids': np.random.rand(10, 10, 2).astype(np.float32),
        'order_book_asks': np.random.rand(10, 10, 2).astype(np.float32),
        'agent_cash': 1000.0,
        'agent_holdings': np.zeros(10, dtype=np.int32),
        'agent_total_value': 1000.0,
        'is_black_box': 0,
        'current_price': np.array([100.0, 200.0, 150.0] + [100.0] * 7, dtype=np.float32),
        'market_cap': np.array([1e6, 2e6, 1.5e6] + [1e6] * 7, dtype=np.float32),
        'active_events': np.zeros(10, dtype=np.float32),
        'available_assets': np.array([1.0, 1.0, 1.0] + [0.0] * 7, dtype=np.float32),
    }
    
    action = agent.decide_action(dummy_obs, None)
    print(f"Action shape: {action.shape}")
    print(f"Action: {action}")
    print(f"Action type: {int(action[0])}, Asset idx: {int(action[1])}, Quantity: {int(action[2])}, Price: {action[3]:.2f}")
    
    print("AdversarialAgent multi-asset test passed!\n")

if __name__ == "__main__":
    test_multi_asset_order_book()
    test_multi_asset_environment()
    test_adversarial_agent_multi_asset()
    print("All multi-asset integration tests passed!")
