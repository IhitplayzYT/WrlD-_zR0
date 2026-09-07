from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from sortedcontainers import SortedDict, SortedList
import time
from src.market.order_book import OrderBook, Order, OrderSide, Trade


class MultiAssetOrderBook:
    """
    Manages order books for multiple assets.
    Each asset has its own independent order book.
    """
    
    def __init__(self, assets: List[str], tick_size: float = 0.01):
        self.assets = assets
        self.tick_size = tick_size
        self.order_books: Dict[str, OrderBook] = {}
        self.asset_to_idx: Dict[str, int] = {asset: idx for idx, asset in enumerate(assets)}
        
        # Initialize order book for each asset
        for asset in assets:
            self.order_books[asset] = OrderBook(tick_size=tick_size)
    
    def get_order_book(self, asset: str) -> OrderBook:
        """Get order book for a specific asset"""
        if asset not in self.order_books:
            raise ValueError(f"Asset {asset} not found in order books")
        return self.order_books[asset]
    
    def get_order_book_by_idx(self, asset_idx: int) -> Optional[OrderBook]:
        """Get order book by asset index"""
        for asset, idx in self.asset_to_idx.items():
            if idx == asset_idx:
                return self.order_books[asset]
        return None
    
    def add_order(
        self, 
        agent_id: str, 
        asset: str, 
        side: OrderSide, 
        price: float, 
        quantity: int
    ) -> str:
        """Add order to specific asset's order book"""
        if asset not in self.order_books:
            raise ValueError(f"Asset {asset} not found")
        return self.order_books[asset].add_order(agent_id, side, price, quantity)
    
    def add_order_by_idx(
        self, 
        agent_id: str, 
        asset_idx: int, 
        side: OrderSide, 
        price: float, 
        quantity: int
    ) -> str:
        """Add order using asset index"""
        order_book = self.get_order_book_by_idx(asset_idx)
        if order_book is None:
            raise ValueError(f"Invalid asset index: {asset_idx}")
        return order_book.add_order(agent_id, side, price, quantity)
    
    def cancel_order(self, asset: str, order_id: str) -> bool:
        """Cancel order in specific asset's order book"""
        if asset not in self.order_books:
            return False
        return self.order_books[asset].cancel_order(order_id)
    
    def cancel_order_by_idx(self, asset_idx: int, order_id: str) -> bool:
        """Cancel order using asset index"""
        order_book = self.get_order_book_by_idx(asset_idx)
        if order_book is None:
            return False
        return order_book.cancel_order(order_id)
    
    def get_best_bid(self, asset: str) -> Optional[Tuple[float, int]]:
        """Get best bid for specific asset"""
        if asset not in self.order_books:
            return None
        return self.order_books[asset].get_best_bid()
    
    def get_best_ask(self, asset: str) -> Optional[Tuple[float, int]]:
        """Get best ask for specific asset"""
        if asset not in self.order_books:
            return None
        return self.order_books[asset].get_best_ask()
    
    def get_spread(self, asset: str) -> Optional[float]:
        """Get spread for specific asset"""
        if asset not in self.order_books:
            return None
        return self.order_books[asset].get_spread()
    
    def get_mid_price(self, asset: str) -> Optional[float]:
        """Get mid price for specific asset"""
        if asset not in self.order_books:
            return None
        return self.order_books[asset].get_mid_price()
    
    def get_market_depth(self, asset: str, depth: int = 5) -> Dict:
        """Get market depth for specific asset"""
        if asset not in self.order_books:
            return {}
        return self.order_books[asset].get_market_depth(depth)
    
    def calculate_price_impact(
        self, 
        asset: str, 
        quantity: int, 
        side: OrderSide
    ) -> float:
        """Calculate price impact for specific asset"""
        if asset not in self.order_books:
            return 0.0
        return self.order_books[asset].calculate_price_impact(quantity, side)
    
    def get_order_book_state(self, asset: str) -> Dict:
        """Get order book state for specific asset"""
        if asset not in self.order_books:
            return {}
        return self.order_books[asset].get_order_book_state()
    
    def get_all_order_book_states(self) -> Dict[str, Dict]:
        """Get order book states for all assets"""
        return {
            asset: self.order_books[asset].get_order_book_state()
            for asset in self.assets
        }
    
    def get_recent_trades(self, asset: str, limit: int = 10) -> List[Dict]:
        """Get recent trades for specific asset"""
        if asset not in self.order_books:
            return []
        return self.order_books[asset].get_recent_trades(limit)
    
    def get_all_recent_trades(self, limit: int = 10) -> Dict[str, List[Dict]]:
        """Get recent trades for all assets"""
        return {
            asset: self.order_books[asset].get_recent_trades(limit)
            for asset in self.assets
        }
    
    def get_aggregated_market_state(self) -> Dict:
        """Get aggregated market state across all assets"""
        state = {
            'assets': self.assets,
            'asset_indices': self.asset_to_idx,
            'order_books': {},
            'mid_prices': {},
            'spreads': {},
            'total_bid_volume': 0,
            'total_ask_volume': 0,
            'total_trades': 0
        }
        
        for asset in self.assets:
            book_state = self.order_books[asset].get_order_book_state()
            state['order_books'][asset] = book_state
            state['mid_prices'][asset] = book_state.get('mid_price')
            state['spreads'][asset] = book_state.get('spread')
            state['total_bid_volume'] += book_state.get('total_bid_volume', 0)
            state['total_ask_volume'] += book_state.get('total_ask_volume', 0)
            state['total_trades'] += book_state.get('num_trades', 0)
        
        return state
    
    def reset(self):
        """Reset all order books"""
        for order_book in self.order_books.values():
            order_book.reset()
    
    def get_available_assets(self) -> List[str]:
        """Get list of available assets"""
        return self.assets.copy()
    
    def get_asset_count(self) -> int:
        """Get number of assets"""
        return len(self.assets)
    
    def add_asset(self, asset: str):
        """Add a new asset with its own order book"""
        if asset in self.order_books:
            raise ValueError(f"Asset {asset} already exists")
        self.order_books[asset] = OrderBook(tick_size=self.tick_size)
        self.asset_to_idx[asset] = len(self.assets)
        self.assets.append(asset)
    
    def remove_asset(self, asset: str):
        """Remove an asset and its order book"""
        if asset not in self.order_books:
            raise ValueError(f"Asset {asset} not found")
        del self.order_books[asset]
        del self.asset_to_idx[asset]
        self.assets.remove(asset)
        # Rebuild indices
        self.asset_to_idx = {asset: idx for idx, asset in enumerate(self.assets)}


if __name__ == "__main__":
    # Test multi-asset order book
    assets = ["AAPL", "MSFT", "GOOGL"]
    maob = MultiAssetOrderBook(assets)
    
    # Add orders for different assets
    maob.add_order("agent_0", "AAPL", OrderSide.BUY, 150.0, 10)
    maob.add_order("agent_1", "MSFT", OrderSide.SELL, 300.0, 5)
    maob.add_order("agent_2", "GOOGL", OrderSide.BUY, 2500.0, 2)
    
    # Get states
    print("AAPL state:", maob.get_order_book_state("AAPL"))
    print("MSFT state:", maob.get_order_book_state("MSFT"))
    print("GOOGL state:", maob.get_order_book_state("GOOGL"))
    
    # Get aggregated state
    print("\nAggregated state:", maob.get_aggregated_market_state())
    
    # Test by index
    maob.add_order_by_idx("agent_3", 0, OrderSide.SELL, 151.0, 5)
    print("\nAfter adding to AAPL by index:", maob.get_order_book_state("AAPL"))
