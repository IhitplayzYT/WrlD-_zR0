from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats
from enum import Enum
TRADING_DAYS = 252

class MetricType(Enum):
    PROFITABILITY = "profitability"
    RISK = "risk"
    CONSISTENCY = "consistency"
    MARKET_IMPACT = "market_impact"
    PREDICTION_ACCURACY = "prediction_accuracy"


@dataclass
class PerformanceMetrics:
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    # Risk metrics
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0  # Value at Risk at 95% confidence
    cvar_95: float = 0.0  # Conditional VaR at 95% confidence
    beta: float = 0.0
    alpha: float = 0.0
    
    # Consistency metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    avg_trade_duration: float = 0.0
    turnover_rate: float = 0.0
    
    # Market impact metrics
    avg_slippage: float = 0.0
    market_impact: float = 0.0
    liquidity_consumption: float = 0.0
    
    # Prediction metrics (for black box)
    prediction_accuracy: float = 0.0
    prediction_calibration: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {'total_return': self.total_return,'annualized_return': self.annualized_return,'sharpe_ratio': self.sharpe_ratio,'sortino_ratio': self.sortino_ratio,'calmar_ratio': self.calmar_ratio,'max_drawdown': self.max_drawdown,'max_drawdown_duration': self.max_drawdown_duration,'volatility': self.volatility,'downside_deviation': self.downside_deviation,'var_95': self.var_95,'cvar_95': self.cvar_95,'beta': self.beta,'alpha': self.alpha,'win_rate': self.win_rate,'profit_factor': self.profit_factor,'avg_win': self.avg_win,'avg_loss': self.avg_loss,'largest_win': self.largest_win,'largest_loss': self.largest_loss,'total_trades': self.total_trades,'avg_trade_duration': self.avg_trade_duration,'turnover_rate': self.turnover_rate,'avg_slippage': self.avg_slippage,'market_impact': self.market_impact,'liquidity_consumption': self.liquidity_consumption,'prediction_accuracy': self.prediction_accuracy,'prediction_calibration': self.prediction_calibration}

    def from_dict(self, dict: Dict[str, float]):
        self.total_return = dict['total_return']
        self.annualized_return = dict['annualized_return']
        self.sharpe_ratio = dict['sharpe_ratio']
        self.sortino_ratio = dict['sortino_ratio']
        self.calmar_ratio = dict['calmar_ratio']
        self.max_drawdown = dict['max_drawdown']
        self.max_drawdown_duration = dict['max_drawdown_duration']
        self.volatility = dict['volatility']
        self.downside_deviation = dict['downside_deviation']
        self.var_95 = dict['var_95']
        self.cvar_95 = dict['cvar_95']
        self.beta = dict['beta']
        self.alpha = dict['alpha']
        self.win_rate = dict['win_rate']
        self.profit_factor = dict['profit_factor']
        self.avg_win = dict['avg_win']
        self.avg_loss = dict['avg_loss']
        self.largest_win = dict['largest_win']
        self.largest_loss = dict['largest_loss']
        self.total_trades = dict['total_trades']
        self.avg_trade_duration = dict['avg_trade_duration']
        self.turnover_rate = dict['turnover_rate']
        self.avg_slippage = dict['avg_slippage']
        self.market_impact = dict['market_impact']
        self.liquidity_consumption = dict['liquidity_consumption']
        self.prediction_accuracy = dict['prediction_accuracy']
        self.prediction_calibration = dict['prediction_calibration']


class MetricsCalculator:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(self,returns: List[float],portfolio_values: List[float],market_returns: Optional[List[float]] = None,trades: Optional[List[Dict]] = None,predictions: Optional[List[Dict]] = None) -> PerformanceMetrics:
        metrics = PerformanceMetrics()
        metrics.total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        metrics.annualized_return = self._calculate_annualized_return(returns)
        metrics.volatility = np.std(returns) if returns else 0.0
        metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
        metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
        metrics.calmar_ratio = self._calculate_calmar_ratio(portfolio_values)
        metrics.max_drawdown, metrics.max_drawdown_duration = self._calculate_max_drawdown(portfolio_values)
        metrics.downside_deviation = self._calculate_downside_deviation(returns)
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
            metrics.total_trades = len(trades)
            trade_durations = [t.get('duration', 0) for t in trades]
            metrics.avg_trade_duration = np.mean(trade_durations) if trade_durations else 0.0
            slippages = [t.get('slippage', 0) for t in trades]
            metrics.avg_slippage = np.mean(slippages) if slippages else 0.0
            metrics.market_impact = np.mean([t.get('market_impact', 0) for t in trades])
            metrics.liquidity_consumption = np.mean([t.get('liquidity_consumed', 0) for t in trades])
        if predictions:
            accuracies = [p.get('accuracy', 0) for p in predictions]
            calibrations = [p.get('calibration', 0) for p in predictions]
            metrics.prediction_accuracy = np.mean(accuracies) if accuracies else 0.0
            metrics.prediction_calibration = np.mean(calibrations) if calibrations else 0.0
        return metrics
    
    def _calculate_annualized_return(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        return np.mean(returns) * TRADING_DAYS
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        if not returns or np.std(returns) == 0:
            return 0.0
        excess_returns = np.array(returns) - (self.risk_free_rate / 252)
        return np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        excess_returns = np.array(returns) - (self.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        return np.mean(excess_returns) / np.std(downside_returns)
    
    def _calculate_calmar_ratio(self, portfolio_values: List[float]) -> float:
        if not portfolio_values:
            return 0.0
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        max_dd, _ = self._calculate_max_drawdown(portfolio_values)
        if max_dd == 0:
            return float('inf')
        return total_return / abs(max_dd)
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> Tuple[float, int]:
        if not portfolio_values:
            return 0.0, 0
        peak = portfolio_values[0]
        max_drawdown = 0.0
        max_duration = 0
        current_duration = 0
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
        if not negative_returns:
            return 0.0
        return np.std(negative_returns)
    
    def _calculate_var(self, returns: List[float], confidence: float) -> float:
        if not returns:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: List[float], confidence: float) -> float:
        if not returns:
            return 0.0
        var = self._calculate_var(returns, confidence)
        tail_returns = [r for r in returns if r <= var]
        if not tail_returns:
            return var
        return np.mean(tail_returns)
    
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
    
    def bootstrap_test(
        self,
        sample1: List[float],
        sample2: Optional[List[float]] = None,
        metric: str = "mean"
    ) -> Dict[str, Any]:
        """
        Perform bootstrap test for statistical significance.
        
        Args:
            sample1: First sample (e.g., agent returns)
            sample2: Optional second sample (e.g., baseline returns)
            metric: Metric to test ("mean", "median", "std")
        """
        if sample2 is None:
            # Bootstrap single sample
            bootstrap_stats = []
            for _ in range(self.n_bootstrap):
                resample = np.random.choice(sample1, size=len(sample1), replace=True)
                if metric == "mean":
                    bootstrap_stats.append(np.mean(resample))
                elif metric == "median":
                    bootstrap_stats.append(np.median(resample))
                elif metric == "std":
                    bootstrap_stats.append(np.std(resample))
            
            return {
                'metric': metric,
                'original_value': np.mean(sample1) if metric == "mean" else np.median(sample1) if metric == "median" else np.std(sample1),
                'bootstrap_mean': np.mean(bootstrap_stats),
                'bootstrap_std': np.std(bootstrap_stats),
                'confidence_interval': (
                    np.percentile(bootstrap_stats, 2.5),
                    np.percentile(bootstrap_stats, 97.5)
                )
            }
        else:
            # Bootstrap difference between samples
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
            
            original_diff = (np.mean(sample1) - np.mean(sample2) if metric == "mean" else
                           np.median(sample1) - np.median(sample2) if metric == "median" else
                           np.std(sample1) - np.std(sample2))
            
            # Calculate p-value (two-tailed)
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(original_diff))
            
            return {
                'metric': metric,
                'original_difference': original_diff,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'confidence_interval': (
                    np.percentile(bootstrap_diffs, 2.5),
                    np.percentile(bootstrap_diffs, 97.5)
                )
            }
    
    def permutation_test(
        self,
        sample1: List[float],
        sample2: List[float],
        metric: str = "mean"
    ) -> Dict[str, Any]:
        """
        Perform permutation test for statistical significance.
        """
        combined = np.concatenate([sample1, sample2])
        n1 = len(sample1)
        
        observed_diff = (np.mean(sample1) - np.mean(sample2) if metric == "mean" else
                       np.median(sample1) - np.median(sample2) if metric == "median" else
                       np.std(sample1) - np.std(sample2))
        
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
        
        return {
            'metric': metric,
            'observed_difference': observed_diff,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }


class RegimeAnalyzer:
    """
    Analyze performance across different market regimes.
    """
    
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
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        return abs(autocorr) 
    
    def _calculate_market_stability(self, returns: List[float]) -> float:
        if not returns:
            return 0.0
        volatility = np.std(returns)
        return 1.0 / (1.0 + volatility)  # Higher = more stable


class BaselineComparator:

    
    def compare_to_baselines(self,agent_returns: List[float],baseline_returns: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
        comparisons = {}
        for baseline_name, baseline_ret in baseline_returns.items():
            if not baseline_ret:
                continue
            agent_mean = np.mean(agent_returns)
            baseline_mean = np.mean(baseline_ret)
            agent_std = np.std(agent_returns)
            baseline_std = np.std(baseline_ret)
            comparisons[baseline_name] = {'agent_return': agent_mean,'baseline_return': baseline_mean,'outperformance': agent_mean - baseline_mean,'agent_sharpe': agent_mean / agent_std if agent_std > 0 else 0,'baseline_sharpe': baseline_mean / baseline_std if baseline_std > 0 else 0,'sharpe_improvement': (agent_mean / agent_std if agent_std > 0 else 0) - (baseline_mean / baseline_std if baseline_std > 0 else 0)}
        return comparisons


if __name__ == "__main__":
    calculator = MetricsCalculator(risk_free_rate=0.02)
    
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252).tolist()
    portfolio_values = [100.0]
    for r in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + r))
    
    market_returns = np.random.normal(0.0005, 0.015, 252).tolist()
    
    metrics = calculator.calculate_metrics(returns=returns,portfolio_values=portfolio_values,market_returns=market_returns)
    
    print("Performance Metrics:")
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value:.4f}")
    
    tester = StatisticalSignificanceTester(n_bootstrap=1000)
    baseline_returns = np.random.normal(0.0005, 0.015, 252).tolist()
    
    bootstrap_result = tester.bootstrap_test(returns, baseline_returns, metric="mean")
    print(f"\nBootstrap Test: {bootstrap_result}")
    
    permutation_result = tester.permutation_test(returns, baseline_returns, metric="mean")
    print(f"Permutation Test: {permutation_result}")
