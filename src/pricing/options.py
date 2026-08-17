from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta


class OptionType(Enum):
    """Types of options"""
    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    """Exercise styles"""
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class OptionContract:
    """
    Option contract specification.
    """
    contract_id: str
    underlying: str
    option_type: OptionType
    strike_price: float
    expiry_date: datetime
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    multiplier: float = 100.0
    is_active: bool = True
    
    def time_to_expiry(self, current_date: datetime) -> float:
        """Calculate time to expiry in years"""
        delta = self.expiry_date - current_date
        return max(0.0, delta.days / 365.0)
    
    def is_expired(self, current_date: datetime) -> bool:
        """Check if option is expired"""
        return current_date >= self.expiry_date


class BlackScholesModel:
    """Black-Scholes option pricing model."""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return BlackScholesModel.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(0.0, S - K)
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma)
        d2_val = BlackScholesModel.d2(S, K, T, r, sigma)
        price = S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
        return max(0.0, price)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(0.0, K - S)
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma)
        d2_val = BlackScholesModel.d2(S, K, T, r, sigma)
        price = K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
        return max(0.0, price)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
        if T <= 0:
            return 1.0 if (option_type == OptionType.CALL and S > K) else (-1.0 if option_type == OptionType.PUT and S < K else 0.0)
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma)
        return norm.cdf(d1_val) if option_type == OptionType.CALL else norm.cdf(d1_val) - 1.0
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma)
        return norm.pdf(d1_val) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return 0.0
        d1_val = BlackScholesModel.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1_val) * np.sqrt(T)
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: OptionType) -> float:
        sigma = 0.5
        for _ in range(100):
            model_price = BlackScholesModel.call_price(S, K, T, r, sigma) if option_type == OptionType.CALL else BlackScholesModel.put_price(S, K, T, r, sigma)
            price_diff = model_price - market_price
            if abs(price_diff) < 1e-6:
                break
            vega = BlackScholesModel.vega(S, K, T, r, sigma)
            if vega < 1e-6:
                break
            sigma = sigma - price_diff / vega
            sigma = max(0.01, min(5.0, sigma))
        return sigma


class OptionsPricingSystem:
    """Comprehensive options pricing system."""
    
    def __init__(self, risk_free_rate: float = 0.02, default_volatility: float = 0.2):
        self.risk_free_rate = risk_free_rate
        self.default_volatility = default_volatility
        self.contracts: Dict[str, OptionContract] = {}
        self.underlying_prices: Dict[str, float] = {}
    
    def add_contract(self, contract: OptionContract):
        self.contracts[contract.contract_id] = contract
    
    def update_underlying_price(self, underlying: str, price: float):
        self.underlying_prices[underlying] = price
    
    def price_option(self, contract_id: str, current_date: datetime, volatility: Optional[float] = None) -> float:
        contract = self.contracts.get(contract_id)
        if not contract:
            raise ValueError(f"Contract {contract_id} not found")
        
        S = self.underlying_prices.get(contract.underlying, 100.0)
        K = contract.strike_price
        T = contract.time_to_expiry(current_date)
        r = self.risk_free_rate
        sigma = volatility or self.default_volatility
        
        if T <= 0:
            return 0.0
        
        if contract.option_type == OptionType.CALL:
            return BlackScholesModel.call_price(S, K, T, r, sigma)
        else:
            return BlackScholesModel.put_price(S, K, T, r, sigma)
    
    def calculate_greeks(self, contract_id: str, current_date: datetime) -> Dict[str, float]:
        contract = self.contracts.get(contract_id)
        if not contract:
            raise ValueError(f"Contract {contract_id} not found")
        
        S = self.underlying_prices.get(contract.underlying, 100.0)
        K = contract.strike_price
        T = contract.time_to_expiry(current_date)
        r = self.risk_free_rate
        sigma = self.default_volatility
        
        if T <= 0:
            return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        
        return {
            'delta': BlackScholesModel.delta(S, K, T, r, sigma, contract.option_type),
            'gamma': BlackScholesModel.gamma(S, K, T, r, sigma),
            'vega': BlackScholesModel.vega(S, K, T, r, sigma),
            'theta': 0.0,
            'rho': 0.0
        }


if __name__ == "__main__":
    system = OptionsPricingSystem()
    
    contract = OptionContract(
        contract_id="AAPL_CALL_100",
        underlying="AAPL",
        option_type=OptionType.CALL,
        strike_price=100.0,
        expiry_date=datetime.now() + timedelta(days=30)
    )
    
    system.add_contract(contract)
    system.update_underlying_price("AAPL", 105.0)
    
    price = system.price_option("AAPL_CALL_100", datetime.now())
    greeks = system.calculate_greeks("AAPL_CALL_100", datetime.now())
    
    print(f"Option Price: {price:.2f}")
    print(f"Greeks: {greeks}")
