from typing import Optional
from typing import Dict, List

class MultiStockEnvironment:
    def __init__(self, assets: List[str], stock_data: Dict[str, List[float]]):
        self.assets = assets
        self.stock_data = stock_data
        self.current_prices: Dict[str, float] = {}
        self.reset()

    def reset(self):
        for asset in self.assets:
            prices = self.stock_data.get(asset, None)
            if not prices:
                continue
            self.current_prices[asset] = prices[0]

    def update_prices(self, new_prices: Dict[str, float]):
        for asset, price in new_prices.items():
            if asset in self.current_prices:
                self.current_prices[asset] = max(1.0, float(price))

    def get_stock_price(self, asset: str, step: int) -> Optional[float]:
        if asset not in self.stock_data:
            return None
        prices = self.stock_data[asset]
        if not prices:
            return None
        idx = min(step, len(prices) - 1)
        return prices[idx]

    def get_all_prices(self, step: int) -> Dict[str, float]:
        ret = {}
        for asset in self.assets:
            price = self.get_stock_price(asset,step)
            if price:
                ret[asset] = price
        return ret