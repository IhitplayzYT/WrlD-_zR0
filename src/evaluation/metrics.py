from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from enum import Enum
from src.utils.logging import JSONLogger, get_logger

TRADING_DAYS = 252

class MetricType(Enum):
    PROFITABILITY = "profitability"
    RISK = "risk"
    CONSISTENCY = "consistency"
    MARKET_IMPACT = "market_impact"
    PREDICTION_ACCURACY = "prediction_accuracy"

@dataclass
class PerformanceMetrics:
    tot_ret = annualize_ret = 0.0
    calmar_r = sortino_r = sharpe_r = 0.0
    mx_drwdwn,mx_drwdwn_dur = 0.0,0

    # Risk metrics
    volatility = down_dev = 0.0
    var_95 = cvar_95 = 0.0 
    alpha = beta = 0.0
    
    win_rate = profit_factor = 0.0
    avg_win = avg_loss = 0.0
    largest_win = largest_loss = 0.0
    tot_trades,avg_trade_dur,turnover_r = 0,0.0,0.0
    avg_slippage = market_impact = liquidity_usage = 0.0
    predict_acc = predict_calib = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {'total_return': self.tot_ret,'annualized_return': self.annualize_ret,'sharpe_ratio': self.sharpe_r,'sortino_ratio': self.sortino_r,'calmar_ratio': self.calmar_r,'max_drawdown': self.mx_drwdwn,'max_drawdown_duration': self.mx_drwdwn_dur,'volatility': self.volatility,'downside_deviation': self.down_dev,'var_95': self.var_95,'cvar_95': self.cvar_95,'beta': self.beta,'alpha': self.alpha,'win_rate': self.win_rate,'profit_factor': self.profit_factor,'avg_win': self.avg_win,'avg_loss': self.avg_loss,'largest_win': self.largest_win,'largest_loss': self.largest_loss,'total_trades': self.tot_trades,'avg_trade_duration': self.avg_trade_dur,'turnover_rate': self.turnover_r,'avg_slippage': self.avg_slippage,'market_impact': self.market_impact,'liquidity_consumption': self.liquidity_usage,'prediction_accuracy': self.predict_acc,'prediction_calibration': self.predict_calib}


class MetricsCalculator:
    def __init__(self, risk_free_rate: float = 0.02, logger: Optional[JSONLogger] = None):
        self.rfr = risk_free_rate
        self.logger = logger or get_logger()
    
    def calculate_metrics(self, returns: List[float], portfolio_values: List[float], market_returns: Optional[List[float]] = None, trades: Optional[List[Dict]] = None, predictions: Optional[List[Dict]] = None) -> PerformanceMetrics:
        metrics = PerformanceMetrics()
        if portfolio_values and len(portfolio_values) > 1 and portfolio_values[0] != 0:
            metrics.tot_ret = (portfolio_values[-1] / portfolio_values[0]) - 1
        metrics.annualize_ret = self._calculate_annualized_return(returns)
        metrics.volatility = np.std(returns) if returns else 0.0
        metrics.sharpe_r = self._calculate_sharpe_ratio(returns)
        metrics.sortino_r = self._calculate_sortino_ratio(returns)
        metrics.calmar_r = self._calculate_calmar_ratio(portfolio_values)
        metrics.mx_drwdwn, metrics.mx_drwdwn_dur = self._calculate_max_drawdown(portfolio_values)
        metrics.down_dev = self._calculate_downside_deviation(returns)
        metrics.var_95 = self._calculate_var(returns, 0.95)
        metrics.cvar_95 = self._calculate_cvar(returns, 0.95)
        if market_returns:
            metrics.beta, metrics.alpha = self._calculate_beta_alpha(returns, market_returns)
        if returns:
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r < 0]
            metrics.win_rate = len(wins) / len(returns) if returns else 0.0
            metrics.avg_win = np.mean(wins) if wins else 0.0
            metrics.avg_loss = np.mean(losses) if losses else 0.0
            metrics.largest_win = max(wins) if wins else 0.0
            metrics.largest_loss = min(losses) if losses else 0.0
            metrics.profit_factor = abs(sum(wins) / sum(losses)) if losses else float('inf')
        if trades:
            metrics.tot_trades = len(trades)
            trade_durations = [t.get('duration', 0) for t in trades]
            metrics.avg_trade_dur = np.mean(trade_durations) if trade_durations else 0.0
            slippages = [t.get('slippage', 0) for t in trades]
            metrics.avg_slippage = np.mean(slippages) if slippages else 0.0
            metrics.market_impact = np.mean([t.get('market_impact', 0) for t in trades])
            metrics.liquidity_usage = np.mean([t.get('liquidity_consumed', 0) for t in trades])
        if predictions:
            accuracies = [p.get('accuracy', 0) for p in predictions]
            calibrations = [p.get('calibration', 0) for p in predictions]
            metrics.predict_acc = np.mean(accuracies) if accuracies else 0.0
            metrics.predict_calib = np.mean(calibrations) if calibrations else 0.0
        return metrics
    
    def _calculate_annualized_return(self, returns: List[float]) -> float:
        return 0.0 if not returns else np.mean(returns) * TRADING_DAYS
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        if not returns or np.std(returns) == 0:
            return 0.0
        excess_returns = np.array(returns) - (self.rfr / TRADING_DAYS)
        return np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        excess_returns = np.array(returns) - (self.rfr / TRADING_DAYS)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        std_downside = np.std(downside_returns)
        if std_downside == 0:
            return 0.0
        return float(np.mean(excess_returns) / std_downside)
    
    def _calculate_calmar_ratio(self, portfolio_values: List[float]) -> float:
        if not portfolio_values:
            return 0.0
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        max_dd, _ = self._calculate_max_drawdown(portfolio_values)
        return float('inf') if max_dd == 0 else (total_return / abs(max_dd))
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> Tuple[float, int]:
        if not portfolio_values:
            return 0.0, 0
        peak = portfolio_values[0]
        max_drawdown = 0.0
        max_duration = current_duration = 0
        for i, value in enumerate(portfolio_values):
            if value > peak:
                peak = value
                current_duration = 0
            else:
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                current_duration += 1
                if current_duration > max_duration:
                    max_duration = current_duration
        return max_drawdown, max_duration
    
    def _calculate_downside_deviation(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        negative_returns = [r for r in returns if r < 0]
        return 0.0 if not negative_returns else np.std(negative_returns) 
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        return 0.0 if not returns else np.percentile(returns, (1 - confidence) * 100)

    def _calculate_cvar(self, returns: List[float], confidence: float) -> float:
        if not returns:
            return 0.0
        var = self._calculate_var(returns, confidence)
        tail_returns = [r for r in returns if r <= var]
        return var if not tail_returns else np.mean(tail_returns)
    
    def _calculate_beta_alpha(self,returns: List[float],market_returns: List[float]) -> Tuple[float, float]:
        if len(returns) != len(market_returns) or len(returns) < 2:
            return 0.0, 0.0
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        if market_variance == 0:
            return 0.0, 0.0
        beta = covariance / market_variance
        alpha = np.mean(returns) - beta * np.mean(market_returns)
        return beta, alpha

class StatisticalSignificanceTester:
    def __init__(self, n_bootstrap: int = 1000, random_seed: Optional[int] = None):
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def bootstrap_test(self,sample1: List[float],sample2: Optional[List[float]] = None,metric: str = "mean") -> Dict[str, Any]:
        if sample2 is None:
            bootstrap_stats = []
            for _ in range(self.n_bootstrap):
                resample = np.random.choice(sample1, size=len(sample1), replace=True)
                if metric == "mean":
                    bootstrap_stats.append(np.mean(resample))
                elif metric == "median":
                    bootstrap_stats.append(np.median(resample))
                elif metric == "std":
                    bootstrap_stats.append(np.std(resample))
            return {'metric': metric,'original_value': np.mean(sample1) if metric == "mean" else np.median(sample1) if metric == "median" else np.std(sample1),'bootstrap_mean': np.mean(bootstrap_stats),'bootstrap_std': np.std(bootstrap_stats),'confidence_interval': (np.percentile(bootstrap_stats, 2.5),np.percentile(bootstrap_stats, 97.5))}
        else:
            bootstrap_diffs = []
            for _ in range(self.n_bootstrap):
                resample1 = np.random.choice(sample1, size=len(sample1), replace=True)
                resample2 = np.random.choice(sample2, size=len(sample2), replace=True)
                if metric == "mean":
                    diff = np.mean(resample1) - np.mean(resample2)
                elif metric == "median":
                    diff = np.median(resample1) - np.median(resample2)
                elif metric == "std":
                    diff = np.std(resample1) - np.std(resample2)
                else:
                    diff = 0
                bootstrap_diffs.append(diff)
            original_diff = (np.mean(sample1) - np.mean(sample2) if metric == "mean" else np.median(sample1) - np.median(sample2) if metric == "median" else np.std(sample1) - np.std(sample2))
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(original_diff))
            return {'metric': metric,'original_difference': original_diff,'p_value': p_value,'is_significant': p_value < 0.05,'confidence_interval': (np.percentile(bootstrap_diffs, 2.5),np.percentile(bootstrap_diffs, 97.5))}
    
    def permutation_test(self,sample1: List[float],sample2: List[float],metric: str = "mean") -> Dict[str, Any]:
        combined = np.concatenate([sample1, sample2])
        n1 = len(sample1)
        observed_diff = (np.mean(sample1) - np.mean(sample2) if metric == "mean" else np.median(sample1) - np.median(sample2) if metric == "median" else np.std(sample1) - np.std(sample2))
        permuted_diffs = []
        for _ in range(self.n_bootstrap):
            np.random.shuffle(combined)
            perm1 = combined[:n1]
            perm2 = combined[n1:]
            
            if metric == "mean":
                diff = np.mean(perm1) - np.mean(perm2)
            elif metric == "median":
                diff = np.median(perm1) - np.median(perm2)
            elif metric == "std":
                diff = np.std(perm1) - np.std(perm2)
            else:
                diff = 0
            
            permuted_diffs.append(diff)
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        return {'metric': metric,'observed_difference': observed_diff,'p_value': p_value,'is_significant': p_value < 0.05}


class RegimeAnalyzer:
    def __init__(self):
        self.regime_performances: Dict[str, List[float]] = {}
    
    def analyze_regime_performance(self,returns: List[float],regimes: List[str]) -> Dict[str, Dict[str, float]]:
        regime_stats = {}    
        uniq_regimes = set(regimes)
        for regime in uniq_regimes:
            regime_returns = [r for r, reg in zip(returns, regimes) if reg == regime]
            if regime_returns:
                regime_stats[regime] = {'mean': np.mean(regime_returns),'std': np.std(regime_returns),'count': len(regime_returns),'sharpe': np.mean(regime_returns) / np.std(regime_returns) if np.std(regime_returns) > 0 else 0}
        return regime_stats


class MarketHealthMetrics:
    
    def calculate_market_health(self,price_history: List[float],order_book_depths: List[float],spreads: List[float]) -> Dict[str, float]:
        if not price_history:
            return {}
        returns = np.diff(price_history) / price_history[:-1]
        return {'volatility': np.std(returns) if len(returns) > 0 else 0.0,'liquidity': np.mean(order_book_depths) if order_book_depths else 0.0,'spread': np.mean(spreads) if spreads else 0.0,'efficiency': self._calculate_market_efficiency(returns),'stability': self._calculate_market_stability(returns)}
    
    def _calculate_market_efficiency(self, returns: List[float]) -> float:
        if len(returns) < 2:
            return 0.0
        return abs(np.corrcoef(returns[:-1], returns[1:])[0, 1]) 
    
    def _calculate_market_stability(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        return 1.0 / (1.0 +  np.std(returns))
