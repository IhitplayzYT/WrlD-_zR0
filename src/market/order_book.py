from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from sortedcontainers import SortedDict, SortedList
import time


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    id: str
    agent_id: str
    side: OrderSide
    price: float
    quantity: int
    timestamp: float = field(default_factory=time.time)
    
    def __lt__(self, other: 'Order') -> bool:
        """For sorting orders"""
        if self.side == OrderSide.BUY:
            # Buy orders: higher price first (price-time priority)
            if self.price != other.price:
                return self.price > other.price
            return self.timestamp < other.timestamp
        else:
            # Sell orders: lower price first (price-time priority)
            if self.price != other.price:
                return self.price < other.price
            return self.timestamp < other.timestamp
    
    def __eq__(self, other: 'Order') -> bool:
        return self.id == other.id


@dataclass
class Trade:
    buy_order_id: str
    sell_order_id: str
    price: float
    quantity: int
    timestamp: float = field(default_factory=time.time)


class OrderBook:
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        # SortedDict: price -> SortedList of orders at that price
        # Bids: descending order (highest price first)
        self.bids: SortedDict[float, SortedList] = SortedDict(lambda k: -k)
        # Asks: ascending order (lowest price first)
        self.asks: SortedDict[float, SortedList] = SortedDict()
        self.order_map: Dict[str, Order] = {}  # Order ID -> Order
        self.trades: List[Trade] = []
        self._order_counter = 0
        
    def _generate_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD_{self._order_counter}"
    
    def add_order(self, agent_id: str, side: OrderSide, price: float, quantity: int) -> str:
        """Add a new order to the book"""
        order_id = self._generate_order_id()
        order = Order(
            id=order_id,
            agent_id=agent_id,
            side=side,
            price=price,
            quantity=quantity
        )
        
        self.order_map[order_id] = order
        
        if side == OrderSide.BUY:
            if price not in self.bids:
                self.bids[price] = SortedList(key=lambda o: o.timestamp)
            self.bids[price].add(order)
        else:
            if price not in self.asks:
                self.asks[price] = SortedList(key=lambda o: o.timestamp)
            self.asks[price].add(order)
        
        # Try to match immediately
        self._match_orders()
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        if order_id not in self.order_map:
            return False
        
        order = self.order_map[order_id]
        
        if order.side == OrderSide.BUY:
            if order.price in self.bids and order in self.bids[order.price]:
                self.bids[order.price].remove(order)
                # Remove price level if empty
                if not self.bids[order.price]:
                    del self.bids[order.price]
        else:
            if order.price in self.asks and order in self.asks[order.price]:
                self.asks[order.price].remove(order)
                # Remove price level if empty
                if not self.asks[order.price]:
                    del self.asks[order.price]
        
        del self.order_map[order_id]
        return True
    
    def _match_orders(self) -> List[Trade]:
        """Match orders and return trades"""
        new_trades = []
        
        while self.bids and self.asks:
            # Get best bid (highest price) and best ask (lowest price)
            best_bid_price = self.bids.keys()[0]  # First key in descending order
            best_ask_price = self.asks.keys()[0]   # First key in ascending order
            
            # Check if orders can match
            if best_bid_price >= best_ask_price:
                # Get the first order at each price level
                best_bid = self.bids[best_bid_price][0]
                best_ask = self.asks[best_ask_price][0]
                
                # Calculate trade price (midpoint)
                trade_price = (best_bid_price + best_ask_price) / 2.0
                
                # Calculate trade quantity
                trade_quantity = min(best_bid.quantity, best_ask.quantity)
                
                # Create trade
                trade = Trade(
                    buy_order_id=best_bid.id,
                    sell_order_id=best_ask.id,
                    price=trade_price,
                    quantity=trade_quantity
                )
                new_trades.append(trade)
                self.trades.append(trade)
                
                # Update order quantities
                best_bid.quantity -= trade_quantity
                best_ask.quantity -= trade_quantity
                
                # Remove or update filled orders
                if best_bid.quantity == 0:
                    self.bids[best_bid_price].pop(0)
                    del self.order_map[best_bid.id]
                    if not self.bids[best_bid_price]:
                        del self.bids[best_bid_price]
                else:
                    # Update the order in the map
                    self.order_map[best_bid.id] = best_bid
                
                if best_ask.quantity == 0:
                    self.asks[best_ask_price].pop(0)
                    del self.order_map[best_ask.id]
                    if not self.asks[best_ask_price]:
                        del self.asks[best_ask_price]
                else:
                    # Update the order in the map
                    self.order_map[best_ask.id] = best_ask
            else:
                # No more matches possible
                break
        
        return new_trades
    
    def get_best_bid(self) -> Optional[Tuple[float, int]]:
        """Get best bid price and total quantity at that price"""
        if not self.bids:
            return None
        best_price = self.bids.keys()[0]
        total_qty = sum(o.quantity for o in self.bids[best_price])
        return (best_price, total_qty)
    
    def get_best_ask(self) -> Optional[Tuple[float, int]]:
        """Get best ask price and total quantity at that price"""
        if not self.asks:
            return None
        best_price = self.asks.keys()[0]
        total_qty = sum(o.quantity for o in self.asks[best_price])
        return (best_price, total_qty)
    
    def get_spread(self) -> Optional[float]:
        """Get current bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return best_ask[0] - best_bid[0]
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid[0] + best_ask[0]) / 2.0
    
    def get_market_depth(self, depth: int = 5) -> Dict:
        """Get market depth at multiple price levels"""
        bid_levels = []
        ask_levels = []
        
        # Aggregate bids at each price level
        bid_prices = {}
        for order in self.bids[:depth * 10]:  # Get more orders to aggregate
            if order.price not in bid_prices:
                bid_prices[order.price] = 0
            bid_prices[order.price] += order.quantity
        
        sorted_bid_prices = sorted(bid_prices.keys(), reverse=True)[:depth]
        for price in sorted_bid_prices:
            bid_levels.append({
                'price': price,
                'quantity': bid_prices[price]
            })
        
        # Aggregate asks at each price level
        ask_prices = {}
        for order in self.asks[:depth * 10]:
            if order.price not in ask_prices:
                ask_prices[order.price] = 0
            ask_prices[order.price] += order.quantity
        
        sorted_ask_prices = sorted(ask_prices.keys())[:depth]
        for price in sorted_ask_prices:
            ask_levels.append({
                'price': price,
                'quantity': ask_prices[price]
            })
        
        return {
            'bids': bid_levels,
            'asks': ask_levels,
            'spread': self.get_spread(),
            'mid_price': self.get_mid_price()
        }
    
    def calculate_price_impact(self, quantity: int, side: OrderSide) -> float:
        """
        Calculate estimated price impact for a given order quantity
        Returns the expected slippage in price units
        """
        if side == OrderSide.BUY:
            # For buying, calculate how much we'd move up the ask book
            remaining_qty = quantity
            total_cost = 0.0
            filled_qty = 0
            
            for price in self.asks.keys():
                if remaining_qty <= 0:
                    break
                
                for order in self.asks[price]:
                    if remaining_qty <= 0:
                        break
                    
                    fill_qty = min(remaining_qty, order.quantity)
                    total_cost += order.price * fill_qty
                    filled_qty += fill_qty
                    remaining_qty -= fill_qty
            
            if filled_qty == 0:
                return 0.0
            
            avg_price = total_cost / filled_qty
            current_mid = self.get_mid_price()
            if current_mid is None:
                return 0.0
            
            return avg_price - current_mid
        else:
            # For selling, calculate how much we'd move down the bid book
            remaining_qty = quantity
            total_revenue = 0.0
            filled_qty = 0
            
            for price in self.bids.keys():
                if remaining_qty <= 0:
                    break
                
                for order in self.bids[price]:
                    if remaining_qty <= 0:
                        break
                    
                    fill_qty = min(remaining_qty, order.quantity)
                    total_revenue += order.price * fill_qty
                    filled_qty += fill_qty
                    remaining_qty -= fill_qty
            
            if filled_qty == 0:
                return 0.0
            
            avg_price = total_revenue / filled_qty
            current_mid = self.get_mid_price()
            if current_mid is None:
                return 0.0
            
            return current_mid - avg_price
    
    def get_order_book_state(self) -> Dict:
        """Get current state of the order book"""
        bid_list = []
        ask_list = []
        
        # Get top 10 bid levels
        for i, price in enumerate(self.bids.keys()):
            if i >= 10:
                break
            for order in self.bids[price]:
                bid_list.append((order.price, order.quantity))
                if len(bid_list) >= 10:
                    break
            if len(bid_list) >= 10:
                break
        
        # Get top 10 ask levels
        for i, price in enumerate(self.asks.keys()):
            if i >= 10:
                break
            for order in self.asks[price]:
                ask_list.append((order.price, order.quantity))
                if len(ask_list) >= 10:
                    break
            if len(ask_list) >= 10:
                break
        
        return {
            'bids': bid_list,
            'asks': ask_list,
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'spread': self.get_spread(),
            'mid_price': self.get_mid_price(),
            'total_bid_volume': sum(sum(o.quantity for o in orders) for orders in self.bids.values()),
            'total_ask_volume': sum(sum(o.quantity for o in orders) for orders in self.asks.values()),
            'num_trades': len(self.trades)
        }
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades"""
        recent = self.trades[-limit:] if self.trades else []
        return [
            {
                'price': t.price,
                'quantity': t.quantity,
                'timestamp': t.timestamp,
                'buy_order_id': t.buy_order_id,
                'sell_order_id': t.sell_order_id
            }
            for t in recent
        ]
    
    def reset(self):
        """Reset the order book"""
        self.bids.clear()
        self.asks.clear()
        self.order_map.clear()
        self.trades.clear()
        self._order_counter = 0
