from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from enum import Enum

class Regime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRASH = "crash"
    RECOVERY = "recovery"

class Trend(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    NEUTRAL = "neutral"


@dataclass
class RegimeDetection:
    lookback_window: int = 20
    volatility_threshold: float = 0.02
    trend_threshold: float = 0.01
    
    def detect_regime(self, price_history: List[float]) -> Tuple[Regime, float]:
        """ Returns (regime, confidence_score) """
        if len(price_history) < self.lookback_window:
            return Regime.SIDEWAYS, 0.5
        recent_prices = np.array(price_history[-self.lookback_window:])
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
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
        """ Detect current trend using dual moving average crossover, Returns (trend, confidence_score) """
        if len(price_history) < 20:
            return Trend.NEUTRAL, 0.5
        prices = np.array(price_history)
        # Calculate moving averages
        ma_short = np.mean(prices[-5:])
        ma_long = np.mean(prices[-20:])
        
        if ma_short > ma_long * 1.01:
            return Trend.UPTREND, 0.7
        elif ma_short < ma_long * 0.99:
            return Trend.DOWNTREND, 0.7
        else:
            return Trend.NEUTRAL, 0.5


class TemporalFeatureExtractor(nn.Module):
    """
    GRU/LSTM-based temporal feature extraction for price sequences.
    Based on research: LSTM-augmented DQN for quantitative trading in partially observable markets.
    """
    
    def __init__(self,input_size: int = 1,hidden_size: int = 64,num_layers: int = 2,dropout: float = 0.1,use_gru: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if use_gru:
            self.rnn = nn.GRU(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout if num_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=dropout if num_layers > 1 else 0)

       # Decoders 
        self.price_predictor = nn.Linear(hidden_size, 1)
        self.regime_classifier = nn.Linear(hidden_size, 5)
        self.volatility_predictor = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ x shape: (batch_size, sequence_length, input_size) """
        rnn_out, hidden = self.rnn(x)
        # Use the last output for predictions
        last_output = rnn_out[:, -1, :]
        return {'hidden_state': last_output,'price_prediction': self.price_predictor(last_output),'regime_logits': self.regime_classifier(last_output),'volatility_prediction': self.volatility_predictor(last_output)}
    
    def extract_features(self, price_sequence: List[float]) -> np.ndarray:
        """
        Extract temporal features from price sequence.
        Returns hidden state representation.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(price_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            outputs = self.forward(x)
            return outputs['hidden_state'].squeeze(0).cpu().numpy()


@dataclass
class ImplementationShortfall:
    """ Implementation shortfall metrics for observation space """
    arrival_price: float = 0.0
    executed_price: float = 0.0
    total_shares: int = 0
    executed_shares: int = 0
    remaining_shares: int = 0
    
    def calculate_shortfall(self) -> float:
        """ Positive = worse than arrival price, Negative = better than arrival price """
        if self.executed_shares == 0:
            return 0.0
        avg_executed_price = self.executed_price / self.executed_shares

        if avg_executed_price > self.arrival_price:
            shortfall = (avg_executed_price - self.arrival_price) / self.arrival_price
        else:
            shortfall = (self.arrival_price - avg_executed_price) / self.arrival_price
        return shortfall
    
    def calculate_price_divergence(self, current_best_bid: float, current_best_ask: float, is_buy: bool) -> float:
        return  self.arrival_price - current_best_ask if is_buy else current_best_bid - self.arrival_price


@dataclass
class BeliefState:
    """ Discrete belief state to enhance Markov property in POMDPs """
    regime_beliefs: Dict[str, float] = field(default_factory=dict)
    trend_belief: float = 0.5
    volatility_belief: float = 0.5
    liquidity_belief: float = 0.5
    
    def update_regime_beliefs(self, regime: Regime, confidence: float):
        regime_name = regime.value
        old_belief = self.regime_beliefs.get(regime_name, 0.5)
        new_belief = old_belief * confidence + (1 - old_belief) * (1 - confidence)
        self.regime_beliefs[regime_name] = new_belief
        
        total = sum(self.regime_beliefs.values())
        if total > 0:
            for key in self.regime_beliefs:
                self.regime_beliefs[key] /= total
    
    def get_discrete_belief_vector(self) -> np.ndarray:
        all_regimes = ['bull', 'bear', 'sideways', 'crash', 'recovery']
        belief_vector = []
        for regime in all_regimes:
            belief_vector.append(self.regime_beliefs.get(regime, 0.2))

        belief_vector.extend([self.trend_belief,self.volatility_belief,self.liquidity_belief])
        return np.array(belief_vector, dtype=np.float32)


class EnhancedObservationSpace:
    """ Implements advanced observation space design based on research insights """
    
    def __init__(self, use_temporal_features: bool = True, use_belief_states: bool = True):
        self.use_temporal_features = use_temporal_features
        self.use_belief_states = use_belief_states
        
        self.regime_detector = RegimeDetection()
        self.belief_state = BeliefState()
        self.implementation_shortfall = ImplementationShortfall()
        self.temporal_extractor = TemporalFeatureExtractor(input_size=1,hidden_size=64,num_layers=2,use_gru=True) if use_temporal_features else None
        
        self.observation_history = []
    
    def create_observation(self,base_observation: Dict[str, Any],agent_state: Dict[str, Any],market_state: Dict[str, Any]) -> Dict[str, Any]:
        enhanced_obs = base_observation.copy()
        price_history = base_observation.get('price_history', [])
        regime, regime_confidence = self.regime_detector.detect_regime(price_history)
        trend, trend_confidence = self.regime_detector.detect_trend(price_history)
        
        if self.use_belief_states:
            self.belief_state.update_regime_beliefs(regime, regime_confidence)
            self.belief_state.trend_belief = trend_confidence
            enhanced_obs['belief_state'] = self.belief_state.get_discrete_belief_vector()
        
        enhanced_obs['current_regime'] = regime.value
        enhanced_obs['regime_confidence'] = regime_confidence
        enhanced_obs['current_trend'] = trend.value
        enhanced_obs['trend_confidence'] = trend_confidence
        
        if self.use_temporal_features and self.temporal_extractor is not None:
            if len(price_history) >= 10:
                temporal_features = self.temporal_extractor.extract_features(price_history[-10:])
                enhanced_obs['temporal_features'] = temporal_features
            else:
                enhanced_obs['temporal_features'] = np.zeros(64, dtype=np.float32)
        
        current_price = market_state.get('current_price', 100.0)
        best_bid = market_state.get('best_bid', current_price * 0.99)
        best_ask = market_state.get('best_ask', current_price * 1.01)
        
        if self.implementation_shortfall.arrival_price == 0.0:
            self.implementation_shortfall.arrival_price = current_price
        
        is_buying = agent_state.get('holdings', 0) > 0
        price_divergence = self.implementation_shortfall.calculate_price_divergence(best_bid, best_ask, is_buying)
        enhanced_obs['price_divergence'] = price_divergence
        shortfall = self.implementation_shortfall.calculate_shortfall()
        enhanced_obs['implementation_shortfall'] = shortfall
        enhanced_obs['market_condition'] = self._calculate_market_condition(price_history, market_state)
        enhanced_obs['remaining_shares'] = agent_state.get('holdings', 0)
        enhanced_obs['agent_cash_ratio'] = agent_state.get('cash', 1000.0) / max(1.0, agent_state.get('total_value', 1000.0))
        
        self.observation_history.append(enhanced_obs.copy())
        if len(self.observation_history) > 100:
            self.observation_history.pop(0)
        
        return enhanced_obs
    
    def _calculate_market_condition(self,price_history: List[float],market_state: Dict[str, Any]) -> float:
        if len(price_history) < 10:
            return 0.5
        
        recent_prices = np.array(price_history[-10:])
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        volatility = np.std(returns)
        volatility_score = max(0.0, min(1.0, 1.0 - volatility / 0.05))
        
        mean_return = np.mean(returns)
        trend_score = max(0.0, min(1.0, 0.5 + mean_return * 10))
        
        spread = market_state.get('spread')
        if spread is None:
            liquidity_score = 0.5
        else:
            liquidity_score = max(0.0, min(1.0, 1.0 - spread / 0.05))
        
        return  (volatility_score * 0.4 + trend_score * 0.4 + liquidity_score * 0.2)
    
    def get_observation_space_size(self) -> Dict[str, Tuple]:
        sizes = {'price_history': (50,),'order_book_bids': (10, 2),'order_book_asks': (10, 2),'agent_cash': (),'agent_holdings': (),'agent_total_value': (),'is_black_box': (),'current_price': (),'market_cap': (),'active_events': (10,)}
        
        if self.use_belief_states:
            sizes['belief_state'] = (8,)  # 5 regimes + 3 other beliefs
        
        if self.use_temporal_features:
            sizes['temporal_features'] = (64,)  # Hidden state size
        
        sizes.update({'current_regime': (),'regime_confidence': (),'current_trend': (),'trend_confidence': (),'price_divergence': (),'implementation_shortfall': (),'market_condition': (),'remaining_shares': (),'agent_cash_ratio': ()})
        return sizes
    
    def reset(self):
        self.belief_state = BeliefState()
        self.implementation_shortfall = ImplementationShortfall()
        self.observation_history = []


def create_enhanced_observation_space(use_temporal_features: bool = True,use_belief_states: bool = True,pretrained_model_path: Optional[str] = None) -> EnhancedObservationSpace:
    obs_space = EnhancedObservationSpace(use_temporal_features=use_temporal_features,use_belief_states=use_belief_states)
    
    if pretrained_model_path and obs_space.temporal_extractor is not None:
        try:
            obs_space.temporal_extractor.load_state_dict(torch.load(pretrained_model_path))
            obs_space.temporal_extractor.eval()
        except Exception as e:
            pass
    return obs_space
