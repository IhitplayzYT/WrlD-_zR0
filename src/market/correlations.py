from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from enum import Enum
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import eigh
import polars as pl
from src.utils.logging import get_logger, LogCategory

logger = get_logger()


class CorrelationMethod(Enum):
    """Methods for calculating correlation"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


@dataclass
class CorrelationMatrix:
    """
    Correlation matrix for multiple assets.
    """
    assets: List[str]
    matrix: np.ndarray
    method: CorrelationMethod
    timestamp: str
    
    def get_correlation(self, asset1: str, asset2: str) -> float:
        """Get correlation between two assets"""
        try:
            idx1 = self.assets.index(asset1)
            idx2 = self.assets.index(asset2)
            return self.matrix[idx1, idx2]
        except ValueError:
            return 0.0
    
    def get_highly_correlated_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Get pairs of assets with correlation above threshold"""
        pairs = []
        n = len(self.assets)
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.matrix[i, j]
                if abs(corr) >= threshold:
                    pairs.append((self.assets[i], self.assets[j], corr))
        
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)


class CorrelationAnalyzer:
    """
    Analyze correlations between multiple assets.
    """
    
    def __init__(self, method: CorrelationMethod = CorrelationMethod.PEARSON):
        self.method = method
        self.price_history: Dict[str, List[float]] = {}
        self.returns_history: Dict[str, List[float]] = {}
        self.correlation_matrix: Optional[CorrelationMatrix] = None
    
    def add_price_data(self, asset: str, price: float):
        """Add price data for an asset"""
        if asset not in self.price_history:
            self.price_history[asset] = []
            self.returns_history[asset] = []
        
        # Calculate return if we have previous price
        if len(self.price_history[asset]) > 0:
            prev_price = self.price_history[asset][-1]
            if prev_price > 0:
                return_val = (price - prev_price) / prev_price
                self.returns_history[asset].append(return_val)
        
        self.price_history[asset].append(price)
    
    def add_price_history(self, asset: str, prices: List[float]):
        """Add historical price data for an asset"""
        if asset not in self.price_history:
            self.price_history[asset] = []
            self.returns_history[asset] = []
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        self.price_history[asset].extend(prices)
        self.returns_history[asset].extend(returns)
    
    def calculate_correlation_matrix(self) -> CorrelationMatrix:
        """Calculate correlation matrix for all assets"""
        assets = list(self.returns_history.keys())
        n = len(assets)
        
        if n < 2:
            raise ValueError("Need at least 2 assets to calculate correlations")
        
        # Build return matrix
        min_length = min(len(self.returns_history[asset]) for asset in assets)
        if min_length < 2:
            raise ValueError("Need at least 2 data points per asset")
        
        return_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    return_matrix[i, j] = 1.0
                else:
                    asset1_returns = self.returns_history[assets[i]][:min_length]
                    asset2_returns = self.returns_history[assets[j]][:min_length]
                    
                    if self.method == CorrelationMethod.PEARSON:
                        corr, _ = pearsonr(asset1_returns, asset2_returns)
                    elif self.method == CorrelationMethod.SPEARMAN:
                        corr, _ = spearmanr(asset1_returns, asset2_returns)
                    else:
                        corr, _ = pearsonr(asset1_returns, asset2_returns)  # Default to Pearson
                    
                    return_matrix[i, j] = corr if not np.isnan(corr) else 0.0
        
        self.correlation_matrix = CorrelationMatrix(
            assets=assets,
            matrix=return_matrix,
            method=self.method,
            timestamp=str(np.datetime64('now'))
        )
        
        return self.correlation_matrix
    
    def get_eigenvalues(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate eigenvalues and eigenvectors of correlation matrix.
        Used for principal component analysis.
        """
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
        
        eigenvalues, eigenvectors = eigh(self.correlation_matrix.matrix)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def get_principal_components(self, n_components: int = 3) -> np.ndarray:
        """
        Get principal components of the correlation matrix.
        """
        eigenvalues, eigenvectors = self.get_eigenvalues()
        
        # Return top n components
        return eigenvectors[:, :n_components]
    
    def calculate_systemic_risk(self) -> Dict[str, float]:
        """
        Calculate systemic risk metrics based on correlations.
        """
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
        
        matrix = self.correlation_matrix.matrix
        n = len(self.correlation_matrix.assets)
        
        # Average correlation
        avg_correlation = (np.sum(matrix) - n) / (n * (n - 1))
        
        # Maximum correlation
        max_correlation = np.max(matrix - np.eye(n))
        
        # Eigenvalue-based systemic risk (first eigenvalue)
        eigenvalues, _ = self.get_eigenvalues()
        first_eigenvalue = eigenvalues[0]
        eigenvalue_ratio = first_eigenvalue / n
        
        return {
            'average_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'first_eigenvalue': first_eigenvalue,
            'eigenvalue_ratio': eigenvalue_ratio,
            'systemic_risk_score': eigenvalue_ratio  # Higher = more systemic risk
        }
    
    def get_diversification_benefit(self, portfolio_weights: Dict[str, float]) -> float:
        """
        Calculate diversification benefit of a portfolio.
        """
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()
        
        assets = self.correlation_matrix.assets
        matrix = self.correlation_matrix.matrix
        
        # Convert weights to array
        weights = np.array([portfolio_weights.get(asset, 0.0) for asset in assets])
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(weights, np.dot(matrix, weights))
        
        # Calculate average variance (assuming unit variance for each asset)
        avg_variance = np.mean(np.diag(matrix))
        
        # Diversification benefit
        diversification_benefit = 1.0 - (portfolio_variance / avg_variance)
        
        return max(0.0, diversification_benefit)
    
    def reset(self):
        """Reset correlation analyzer"""
        self.price_history.clear()
        self.returns_history.clear()
        self.correlation_matrix = None


class FactorModel:
    """
    Factor model for explaining asset returns through common factors.
    """
    
    def __init__(self, n_factors: int = 3):
        self.n_factors = n_factors
        self.factor_loadings: Optional[np.ndarray] = None
        self.factor_returns: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
    
    def fit(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Fit factor model using PCA.
        Returns factor loadings and explained variance.
        """
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=self.n_factors)
        factor_loadings = pca.fit_transform(returns)
        
        self.factor_loadings = factor_loadings
        self.factor_returns = pca.components_
        
        explained_variance_ratio = pca.explained_variance_ratio_
        
        return {
            'factor_loadings': factor_loadings,
            'explained_variance_ratio': explained_variance_ratio,
            'total_explained_variance': np.sum(explained_variance_ratio)
        }
    
    def predict_returns(self, factor_returns: np.ndarray) -> np.ndarray:
        """Predict asset returns from factor returns"""
        if self.factor_loadings is None:
            raise ValueError("Model not fitted")
        
        predicted = np.dot(factor_returns, self.factor_loadings.T)
        return predicted


class MultiStockEnvironment:
    """
    Multi-stock environment with correlation modeling.
    """
    
    def __init__(
        self,
        assets: List[str],
        correlation_analyzer: Optional[CorrelationAnalyzer] = None
    ):
        self.assets = assets
        self.correlation_analyzer = correlation_analyzer or CorrelationAnalyzer()
        self.current_prices: Dict[str, float] = {asset: 100.0 for asset in assets}
        self.price_history: Dict[str, List[float]] = {asset: [100.0] for asset in assets}
    
    def update_prices(self, price_updates: Dict[str, float]):
        """Update prices for multiple assets"""
        for asset, price in price_updates.items():
            if asset in self.assets:
                self.current_prices[asset] = price
                self.price_history[asset].append(price)
                self.correlation_analyzer.add_price_data(asset, price)
    
    def generate_correlated_returns(
        self,
        n_steps: int,
        mean_returns: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[float]]:
        """
        Generate correlated returns using Cholesky decomposition.
        """
        if self.correlation_analyzer.correlation_matrix is None:
            self.correlation_analyzer.calculate_correlation_matrix()
        
        corr_matrix = self.correlation_analyzer.correlation_matrix.matrix
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # Add small diagonal perturbation if not positive definite
            corr_matrix += np.eye(len(self.assets)) * 1e-6
            L = np.linalg.cholesky(corr_matrix)
        
        # Generate uncorrelated random returns
        n_assets = len(self.assets)
        uncorrelated_returns = np.random.randn(n_steps, n_assets)
        
        # Apply correlation structure
        correlated_returns = np.dot(uncorrelated_returns, L.T)
        
        # Apply mean returns and volatilities
        if mean_returns is None:
            mean_returns = {asset: 0.0 for asset in self.assets}
        if volatilities is None:
            volatilities = {asset: 0.02 for asset in self.assets}
        
        result = {}
        for i, asset in enumerate(self.assets):
            mu = mean_returns.get(asset, 0.0)
            sigma = volatilities.get(asset, 0.02)
            result[asset] = (correlated_returns[:, i] * sigma + mu).tolist()
        
        return result
    
    def get_market_regime(self) -> str:
        """
        Determine current market regime based on correlations.
        """
        if self.correlation_analyzer.correlation_matrix is None:
            self.correlation_analyzer.calculate_correlation_matrix()
        
        systemic_risk = self.correlation_analyzer.calculate_systemic_risk()
        
        if systemic_risk['systemic_risk_score'] > 0.6:
            return "high_correlation"
        elif systemic_risk['systemic_risk_score'] > 0.4:
            return "moderate_correlation"
        else:
            return "low_correlation"
    
    def reset(self):
        """Reset multi-stock environment"""
        self.current_prices = {asset: 100.0 for asset in self.assets}
        self.price_history = {asset: [100.0] for asset in self.assets}
        self.correlation_analyzer.reset()


if __name__ == "__main__":
    # Test correlation analyzer
    analyzer = CorrelationAnalyzer()
    
    # Add sample data for 3 assets
    np.random.seed(42)
    for asset in ["AAPL", "MSFT", "GOOGL"]:
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        analyzer.add_price_history(asset, prices.tolist())
    
    # Calculate correlations
    corr_matrix = analyzer.calculate_correlation_matrix()
    logger.debug(LogCategory.MARKET, "Correlation Matrix", data={'matrix': corr_matrix.matrix.tolist()})
    
    # Get highly correlated pairs
    pairs = corr_matrix.get_highly_correlated_pairs(threshold=0.5)
    logger.info(LogCategory.MARKET, "Highly correlated pairs", data={'pairs': pairs})
    
    # Calculate systemic risk
    systemic_risk = analyzer.calculate_systemic_risk()
    logger.info(LogCategory.MARKET, "Systemic risk metrics", data=systemic_risk)
    
    # Test multi-stock environment
    env = MultiStockEnvironment(["AAPL", "MSFT", "GOOGL"], analyzer)
    regime = env.get_market_regime()
    logger.info(LogCategory.MARKET, "Market regime", data={'regime': regime})
    
    # Generate correlated returns
    correlated_returns = env.generate_correlated_returns(10)
    logger.debug(LogCategory.MARKET, "Generated correlated returns", data={'asset': env.assets[0], 'returns': correlated_returns[env.assets[0]][:5]})
