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
        if self.side == OrderSide.BUY:
            return self.price > other.price if self.price != other.price else self.timestamp < other.timestamp
        else:
            return self.price < other.price if self.price != other.price else self.timestamp < other.timestamp
    
    def __eq__(self, other: 'Order') -> bool:
        return isinstance(other, Order) and self.id == other.id

@dataclass
class Trade:
    buy_order_id: str
    sell_order_id: str
    price: float
    quantity: int
    timestamp: float = field(default_factory=time.time)

class OrderBook:
    def __init__(self):
        self._order_counter = 0
        self.bids: SortedDict = SortedDict(lambda k: -k)
        self.asks: SortedDict = SortedDict()
        self.order_map: Dict[str, Order] = {} 
        self.trades: List[Trade] = []
        
    def _generate_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD_{self._order_counter}"
    
    def add_order(self, agent_id: str, side: OrderSide, price: float, quantity: int) -> str:
        order_id = self._generate_order_id()
        order = Order(order_id, agent_id, side, price, quantity)
        self.order_map[order_id] = order

        if side == OrderSide.BUY:
            if price not in self.bids:
                self.bids[price] = SortedList(key=lambda o: o.timestamp)
            self.bids[price].add(order)
        else:
            if price not in self.asks:
                self.asks[price] = SortedList(key=lambda o: o.timestamp)
            self.asks[price].add(order)
        
        self._match_orders()
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.order_map:
            return False
        order = self.order_map[order_id]
        if order.side == OrderSide.BUY:
            if order.price in self.bids and order in self.bids[order.price]:
                self.bids[order.price].remove(order)
                if not self.bids[order.price]:
                    del self.bids[order.price]
        else:
            if order.price in self.asks and order in self.asks[order.price]:
                self.asks[order.price].remove(order)
                if not self.asks[order.price]:
                    del self.asks[order.price]
        
        del self.order_map[order_id]
        return True
    
    def _match_orders(self) -> List[Trade]:
        new_trades = []
        while self.bids and self.asks:
            best_bid_price, best_ask_price = self.bids.keys()[0], self.asks.keys()[0] 
            if best_bid_price >= best_ask_price:
                best_bid = self.bids[best_bid_price][0]
                best_ask = self.asks[best_ask_price][0]
                trade_price = (best_bid_price + best_ask_price) * 0.5
                trade_quantity = min(best_bid.quantity, best_ask.quantity)
                trade = Trade(buy_order_id=best_bid.id, sell_order_id=best_ask.id, price=trade_price, quantity=trade_quantity)
                new_trades.append(trade)
                self.trades.append(trade)                
                best_bid.quantity -= trade_quantity
                best_ask.quantity -= trade_quantity
                
                if best_bid.quantity == 0:
                    self.bids[best_bid_price].pop(0)
                    del self.order_map[best_bid.id]
                    if not self.bids[best_bid_price]:
                        del self.bids[best_bid_price]
                else:
                    self.order_map[best_bid.id] = best_bid
                
                if best_ask.quantity == 0:
                    self.asks[best_ask_price].pop(0)
                    del self.order_map[best_ask.id]
                    if not self.asks[best_ask_price]:
                        del self.asks[best_ask_price]
                else:
                    self.order_map[best_ask.id] = best_ask
            else:
                break
        return new_trades
    
    def get_best_bid(self) -> Optional[Tuple[float, int]]:
        if not self.bids:
            return None
        best_price = self.bids.keys()[0]
        return (best_price, sum(o.quantity for o in self.bids[best_price]))
    
    def get_best_ask(self) -> Optional[Tuple[float, int]]:
        if not self.asks:
            return None
        best_price = self.asks.keys()[0]
        return (best_price, sum(o.quantity for o in self.asks[best_price]))
    
    def get_spread(self) -> Optional[float]:
        best_bid, best_ask = self.get_best_bid(), self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask[0] - best_bid[0]
    
    def get_mid_price(self) -> Optional[float]:
        best_bid, best_ask = self.get_best_bid(), self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid[0] + best_ask[0]) * 0.5
    
    def get_market_depth(self, depth: int = 5) -> Dict:
        bid_levels = []
        ask_levels = []
        bid_prices: Dict[float, int] = {}
        ask_prices: Dict[float, int] = {}

        for price in list(self.bids.keys())[:depth * 10]:
            for order in self.bids[price]:
                bid_prices[order.price] = bid_prices.get(order.price, 0) + order.quantity
        for price in list(self.asks.keys())[:depth * 10]:
            for order in self.asks[price]:
                ask_prices[order.price] = ask_prices.get(order.price, 0) + order.quantity

        for price in sorted(bid_prices.keys(), reverse=True)[:depth]:
            bid_levels.append({'price': price, 'quantity': bid_prices[price]})
        for price in sorted(ask_prices.keys())[:depth]:
            ask_levels.append({'price': price, 'quantity': ask_prices[price]})
        
        return {'bids': bid_levels, 'asks': ask_levels, 'spread': self.get_spread(), 'mid_price': self.get_mid_price()}
    
    def calculate_price_impact(self, quantity: int, side: OrderSide) -> float:
        if side == OrderSide.BUY:
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
        bid_list, ask_list = [], []
        for i, price in enumerate(self.bids.keys()):
            if i >= 10:
                break
            for order in self.bids[price]:
                bid_list.append((order.price, order.quantity))
                if len(bid_list) >= 10:
                    break
            if len(bid_list) >= 10:
                break
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
        return [{'price': t.price, 'quantity': t.quantity, 'timestamp': t.timestamp, 'buy_order_id': t.buy_order_id, 'sell_order_id': t.sell_order_id} for t in (self.trades[-limit:] if self.trades else [])]
    
    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self.order_map.clear()
        self.trades.clear()
        self._order_counter = 0
