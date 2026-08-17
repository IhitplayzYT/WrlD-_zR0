from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from enum import Enum


class Regime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRASH = "crash"
    RECOVERY = "recovery"


class Trend(Enum):
    """Trend classifications"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    NEUTRAL = "neutral"


@dataclass
class RegimeDetection:
    """
    Detect market regimes using statistical methods and ML models.
    Based on research for partial information handling in trading.
    """
    lookback_window: int = 20
    volatility_threshold: float = 0.02
    trend_threshold: float = 0.01
    
    def detect_regime(self, price_history: List[float]) -> Tuple[Regime, float]:
        """
        Detect current market regime from price history.
        Returns (regime, confidence_score)
        """
        if len(price_history) < self.lookback_window:
            return Regime.SIDEWAYS, 0.5
        
        recent_prices = np.array(price_history[-self.lookback_window:])
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Calculate statistics
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Determine regime based on statistics
        if volatility > self.volatility_threshold * 3:
            if mean_return < -self.trend_threshold * 2:
                return Regime.CRASH, 0.8
            else:
                return Regime.SIDEWAYS, 0.6
        elif mean_return > self.trend_threshold:
            if volatility < self.volatility_threshold:
                return Regime.BULL, 0.7
            else:
                return Regime.RECOVERY, 0.6
        elif mean_return < -self.trend_threshold:
            return Regime.BEAR, 0.7
        else:
            return Regime.SIDEWAYS, 0.5
    
    def detect_trend(self, price_history: List[float]) -> Tuple[Trend, float]:
        """
        Detect current trend using dual moving average crossover.
        Returns (trend, confidence_score)
        """
        if len(price_history) < 20:
            return Trend.NEUTRAL, 0.5
        
        prices = np.array(price_history)
        
        # Calculate moving averages
        ma_short = np.mean(prices[-5:])
        ma_long = np.mean(prices[-20:])
        
        # Determine trend
        if ma_short > ma_long * 1.01:  # 1% above
            return Trend.UPTREND, 0.7
        elif ma_short < ma_long * 0.99:  # 1% below
            return Trend.DOWNTREND, 0.7
        else:
            return Trend.NEUTRAL, 0.5


class TemporalFeatureExtractor(nn.Module):
    """
    GRU/LSTM-based temporal feature extraction for price sequences.
    Based on research: LSTM-augmented DQN for quantitative trading in partially observable markets.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gru: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Choose between GRU and LSTM
        if use_gru:
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Output layers for different predictions
        self.price_predictor = nn.Linear(hidden_size, 1)
        self.regime_classifier = nn.Linear(hidden_size, 5)  # 5 regimes
        self.volatility_predictor = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through temporal feature extractor.
        x shape: (batch_size, sequence_length, input_size)
        """
        # RNN forward pass
        rnn_out, hidden = self.rnn(x)
        
        # Use the last output for predictions
        last_output = rnn_out[:, -1, :]
        
        # Make predictions
        price_pred = self.price_predictor(last_output)
        regime_logits = self.regime_classifier(last_output)
        volatility_pred = self.volatility_predictor(last_output)
        
        return {
            'hidden_state': last_output,
            'price_prediction': price_pred,
            'regime_logits': regime_logits,
            'volatility_prediction': volatility_pred
        }
    
    def extract_features(self, price_sequence: List[float]) -> np.ndarray:
        """
        Extract temporal features from price sequence.
        Returns hidden state representation.
        """
        self.eval()
        with torch.no_grad():
            # Prepare input
            x = torch.tensor(price_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            # Forward pass
            outputs = self.forward(x)
            
            # Return hidden state as features
            return outputs['hidden_state'].squeeze(0).cpu().numpy()


@dataclass
class ImplementationShortfall:
    """
    Implementation shortfall metrics for observation space.
    Based on research: Implementation Shortfall as observation variable.
    """
    arrival_price: float = 0.0
    executed_price: float = 0.0
    total_shares: int = 0
    executed_shares: int = 0
    remaining_shares: int = 0
    
    def calculate_shortfall(self) -> float:
        """
        Calculate implementation shortfall.
        Positive = worse than arrival price, Negative = better than arrival price
        """
        if self.executed_shares == 0:
            return 0.0
        
        # Average executed price
        avg_executed_price = self.executed_price / self.executed_shares
        
        # Shortfall calculation
        if avg_executed_price > self.arrival_price:
            # Buying: paid more than arrival price
            shortfall = (avg_executed_price - self.arrival_price) / self.arrival_price
        else:
            # Selling: received less than arrival price
            shortfall = (self.arrival_price - avg_executed_price) / self.arrival_price
        
        return shortfall
    
    def calculate_price_divergence(self, current_best_bid: float, current_best_ask: float, is_buy: bool) -> float:
        """
        Calculate price divergence from arrival price.
        Based on research: Price divergence as market condition indicator.
        """
        if is_buy:
            # For buy: Arrival Price - Average Ask Prices
            avg_ask = current_best_ask
            divergence = self.arrival_price - avg_ask
        else:
            # For sell: Average Bid Prices - Arrival Price
            avg_bid = current_best_bid
            divergence = avg_bid - self.arrival_price
        
        return divergence


@dataclass
class BeliefState:
    """
    Discrete belief state to enhance Markov property in POMDPs.
    Based on research: Discrete belief states for partially observable markets.
    """
    regime_beliefs: Dict[str, float] = field(default_factory=dict)
    trend_belief: float = 0.5
    volatility_belief: float = 0.5
    liquidity_belief: float = 0.5
    
    def update_regime_beliefs(self, regime: Regime, confidence: float):
        """Update regime beliefs using Bayesian updating"""
        regime_name = regime.value
        old_belief = self.regime_beliefs.get(regime_name, 0.5)
        
        # Bayesian update
        new_belief = old_belief * confidence + (1 - old_belief) * (1 - confidence)
        self.regime_beliefs[regime_name] = new_belief
        
        # Normalize beliefs
        total = sum(self.regime_beliefs.values())
        if total > 0:
            for key in self.regime_beliefs:
                self.regime_beliefs[key] /= total
    
    def get_discrete_belief_vector(self) -> np.ndarray:
        """
        Get discrete belief vector for observation.
        Returns normalized vector of beliefs.
        """
        # Ensure all regimes have beliefs
        all_regimes = ['bull', 'bear', 'sideways', 'crash', 'recovery']
        belief_vector = []
        
        for regime in all_regimes:
            belief_vector.append(self.regime_beliefs.get(regime, 0.2))
        
        # Add other beliefs
        belief_vector.extend([
            self.trend_belief,
            self.volatility_belief,
            self.liquidity_belief
        ])
        
        return np.array(belief_vector, dtype=np.float32)


class EnhancedObservationSpace:
    """
    Enhanced observation space with research-driven features.
    Implements advanced observation space design based on research insights.
    """
    
    def __init__(self, use_temporal_features: bool = True, use_belief_states: bool = True):
        self.use_temporal_features = use_temporal_features
        self.use_belief_states = use_belief_states
        
        # Initialize components
        self.regime_detector = RegimeDetection()
        self.belief_state = BeliefState()
        self.implementation_shortfall = ImplementationShortfall()
        
        # Initialize temporal feature extractor if enabled
        if use_temporal_features:
            self.temporal_extractor = TemporalFeatureExtractor(
                input_size=1,
                hidden_size=64,
                num_layers=2,
                use_gru=True
            )
        else:
            self.temporal_extractor = None
        
        # Observation history for belief updating
        self.observation_history = []
    
    def create_observation(
        self,
        base_observation: Dict[str, Any],
        agent_state: Dict[str, Any],
        market_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create enhanced observation with research-driven features.
        """
        enhanced_obs = base_observation.copy()
        
        # Extract price history
        price_history = base_observation.get('price_history', [])
        
        # Regime detection
        regime, regime_confidence = self.regime_detector.detect_regime(price_history)
        trend, trend_confidence = self.regime_detector.detect_trend(price_history)
        
        # Update belief state
        if self.use_belief_states:
            self.belief_state.update_regime_beliefs(regime, regime_confidence)
            self.belief_state.trend_belief = trend_confidence
            
            # Add discrete belief vector to observation
            enhanced_obs['belief_state'] = self.belief_state.get_discrete_belief_vector()
        
        # Add regime and trend information
        enhanced_obs['current_regime'] = regime.value
        enhanced_obs['regime_confidence'] = regime_confidence
        enhanced_obs['current_trend'] = trend.value
        enhanced_obs['trend_confidence'] = trend_confidence
        
        # Temporal feature extraction
        if self.use_temporal_features and self.temporal_extractor is not None:
            if len(price_history) >= 10:
                temporal_features = self.temporal_extractor.extract_features(price_history[-10:])
                enhanced_obs['temporal_features'] = temporal_features
            else:
                enhanced_obs['temporal_features'] = np.zeros(64, dtype=np.float32)
        
        # Implementation shortfall metrics
        current_price = market_state.get('current_price', 100.0)
        best_bid = market_state.get('best_bid', current_price * 0.99)
        best_ask = market_state.get('best_ask', current_price * 1.01)
        
        # Update arrival price if not set
        if self.implementation_shortfall.arrival_price == 0.0:
            self.implementation_shortfall.arrival_price = current_price
        
        # Calculate price divergence
        is_buying = agent_state.get('holdings', 0) > 0
        price_divergence = self.implementation_shortfall.calculate_price_divergence(
            best_bid, best_ask, is_buying
        )
        enhanced_obs['price_divergence'] = price_divergence
        
        # Calculate implementation shortfall
        shortfall = self.implementation_shortfall.calculate_shortfall()
        enhanced_obs['implementation_shortfall'] = shortfall
        
        # Add market condition indicators
        enhanced_obs['market_condition'] = self._calculate_market_condition(
            price_history, market_state
        )
        
        # Add agent-specific private variables
        enhanced_obs['remaining_shares'] = agent_state.get('holdings', 0)
        enhanced_obs['agent_cash_ratio'] = agent_state.get('cash', 1000.0) / \
                                          max(1.0, agent_state.get('total_value', 1000.0))
        
        # Store observation for history
        self.observation_history.append(enhanced_obs.copy())
        if len(self.observation_history) > 100:
            self.observation_history.pop(0)
        
        return enhanced_obs
    
    def _calculate_market_condition(
        self,
        price_history: List[float],
        market_state: Dict[str, Any]
    ) -> float:
        """
        Calculate overall market condition score.
        Returns value between 0 (bad) and 1 (good).
        """
        if len(price_history) < 10:
            return 0.5
        
        recent_prices = np.array(price_history[-10:])
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Volatility component (lower is better)
        volatility = np.std(returns)
        volatility_score = max(0.0, min(1.0, 1.0 - volatility / 0.05))
        
        # Trend component (positive trend is better)
        mean_return = np.mean(returns)
        trend_score = max(0.0, min(1.0, 0.5 + mean_return * 10))
        
        # Liquidity component (based on spread)
        spread = market_state.get('spread')
        if spread is None:
            liquidity_score = 0.5
        else:
            liquidity_score = max(0.0, min(1.0, 1.0 - spread / 0.05))
        
        # Combine scores
        condition_score = (volatility_score * 0.4 + 
                          trend_score * 0.4 + 
                          liquidity_score * 0.2)
        
        return condition_score
    
    def get_observation_space_size(self) -> Dict[str, Tuple]:
        """
        Get the size of each observation component.
        Returns dictionary mapping component names to their shapes.
        """
        sizes = {
            'price_history': (50,),
            'order_book_bids': (10, 2),
            'order_book_asks': (10, 2),
            'agent_cash': (),
            'agent_holdings': (),
            'agent_total_value': (),
            'is_black_box': (),
            'current_price': (),
            'market_cap': (),
            'active_events': (10,),
        }
        
        if self.use_belief_states:
            sizes['belief_state'] = (8,)  # 5 regimes + 3 other beliefs
        
        if self.use_temporal_features:
            sizes['temporal_features'] = (64,)  # Hidden state size
        
        # Additional enhanced features
        sizes.update({
            'current_regime': (),  # Could be one-hot encoded
            'regime_confidence': (),
            'current_trend': (),
            'trend_confidence': (),
            'price_divergence': (),
            'implementation_shortfall': (),
            'market_condition': (),
            'remaining_shares': (),
            'agent_cash_ratio': (),
        })
        
        return sizes
    
    def reset(self):
        """Reset observation space state for new episode"""
        self.belief_state = BeliefState()
        self.implementation_shortfall = ImplementationShortfall()
        self.observation_history = []


# Factory function for creating enhanced observation spaces
def create_enhanced_observation_space(
    use_temporal_features: bool = True,
    use_belief_states: bool = True,
    pretrained_model_path: Optional[str] = None
) -> EnhancedObservationSpace:
    """
    Create enhanced observation space with optional pretrained model.
    """
    obs_space = EnhancedObservationSpace(
        use_temporal_features=use_temporal_features,
        use_belief_states=use_belief_states
    )
    
    # Load pretrained model if provided
    if pretrained_model_path and obs_space.temporal_extractor is not None:
        try:
            obs_space.temporal_extractor.load_state_dict(
                torch.load(pretrained_model_path)
            )
            obs_space.temporal_extractor.eval()
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
    
    return obs_space


if __name__ == "__main__":
    # Test the enhanced observation space
    obs_space = EnhancedObservationSpace(
        use_temporal_features=True,
        use_belief_states=True
    )
    
    # Create dummy observations
    base_obs = {
        'price_history': [100.0 + i * 0.1 for i in range(50)],
        'order_book_bids': np.zeros((10, 2)),
        'order_book_asks': np.zeros((10, 2)),
        'agent_cash': 1000.0,
        'agent_holdings': 10,
        'agent_total_value': 1100.0,
        'is_black_box': 0,
        'current_price': 105.0,
        'market_cap': 1000000.0,
        'active_events': np.zeros(10)
    }
    
    agent_state = {
        'cash': 1000.0,
        'holdings': 10,
        'total_value': 1100.0
    }
    
    market_state = {
        'current_price': 105.0,
        'best_bid': 104.5,
        'best_ask': 105.5,
        'spread': 0.01
    }
    
    # Create enhanced observation
    enhanced_obs = obs_space.create_observation(base_obs, agent_state, market_state)
    
    print("Enhanced Observation Components:")
    for key, value in enhanced_obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {value}")
    
    print("\nObservation Space Sizes:")
    sizes = obs_space.get_observation_space_size()
    for key, shape in sizes.items():
        print(f"  {key}: {shape}")
