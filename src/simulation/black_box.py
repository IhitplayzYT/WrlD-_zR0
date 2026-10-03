from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from enum import Enum
from datetime import datetime
from src.utils.logging import LogCategory

class PredictionType(Enum):
    PRICE_LEVEL = "price_level"
    REGIME_TRANSITION = "regime_transition"
    TREND_DIRECTION = "trend_direction"
    VOLATILITY_LEVEL = "volatility_level"


@dataclass
class Prediction:
    """ Single prediction by agent during bbox period. """
    agent_id: str
    prediction_type: PredictionType
    predicted_values: List[float]  # Predicted price path for k days
    confidence_interval: Tuple[float, float]  # (lower_bound, upper_bound)
    confidence_level: float  # 0.0 to 1.0
    explanation: str  # Text explanation of reasoning
    timestamp: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> bool:
        """ Validate prediction """
        if not self.predicted_values:
            return False
        
        if len(self.confidence_interval) != 2:
            return False
        
        if self.confidence_interval[0] >= self.confidence_interval[1]:
            return False
        
        if not 0.0 <= self.confidence_level <= 1.0:
            return False
        
        return True

@dataclass
class PredictionResult:
    prediction: Prediction
    actual_values: List[float]
    accuracy_score: float
    confidence_calibration: float  # How well confidence matched accuracy
    reward_multiplier: float = 1.0
    villain_points_earned: int = 0
    is_king: bool = False

@dataclass
class PredictionMarket:
    """
    Prediction market where agents can bet on predictions.
    Implements market mechanism for prediction aggregation.
    """
    market_maker: str = "house"
    liquidity_pool: float = 10000.0
    active_bets: Dict[str, Dict] = field(default_factory=dict)
    
    def place_bet(self,agent_id: str,prediction_id: str,amount: float,outcome: str) -> bool:
        if amount <= 0 or amount > self.liquidity_pool * 0.1:
            return False
        if agent_id in self.active_bets:
            return False 

        self.active_bets[agent_id] = {'prediction_id': prediction_id,'amount': amount,'outcome': outcome}
        return True
    
    def settle_bets(self, correct_predictions: List[str]) -> Dict[str, float]:
        payouts = {}
        for agent_id, bet in self.active_bets.items():
            prediction_id = bet['prediction_id']
            outcome = bet['outcome']
            
            if (outcome == "correct" and prediction_id in correct_predictions) or (outcome == "incorrect" and prediction_id not in correct_predictions):
                payouts[agent_id] = bet['amount'] * 2.0
            else:
                payouts[agent_id] = 0.0
        self.active_bets.clear()
        return payouts


@dataclass
class LeaderboardEntry:
    agent_id: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    average_confidence: float
    confidence_calibration: float
    total_villain_points: int
    king_count: int
    reputation_score: float


from src.utils.logging import JSONLogger, get_logger, LogCategory

class BlackBoxPredictionSystem:
    """
    Implements sophisticated prediction contests with market mechanisms.
    """
    
    def __init__(self, duration: int = 5, frequency: int = 20, min_confidence: float = 0.3, require_explanation: bool = True, enable_prediction_market: bool = True, logger: Optional[JSONLogger] = None):
        self.logger = logger or get_logger()
        self.duration = duration
        self.frequency = frequency
        self.min_confidence = min_confidence
        self.require_explanation = require_explanation
        self.enable_prediction_market = enable_prediction_market
        
        self.is_active = False
        self.remaining_duration = 0
        self.step_counter = 0
        self.current_predictions: Dict[str, Prediction] = {}
        self.prediction_results: List[PredictionResult] = []
        
        self.prediction_market = PredictionMarket() if enable_prediction_market else None
        self.leaderboard: Dict[str, LeaderboardEntry] = {}
        
        self.current_king: Optional[str] = None
        self.king_reward_multiplier: float = 2.0
        self.king_history: List[Tuple[str, int]] = []  # (agent_id, duration)
        
        self.villain_points_threshold = 0.8  # Accuracy threshold for villain points
        self.villain_points_per_prediction = 2
        
        self.use_ensemble = True
        self.ensemble_method = "weighted_average"  # "weighted_average", "voting", "stacking"
    
    def should_trigger(self) -> bool:
        self.step_counter += 1
        if not self.is_active and self.step_counter >= self.frequency:
            self.is_active = True
            self.remaining_duration = self.duration
            self.step_counter = 0
            return True
        return False
    
    def update(self):
        if self.is_active:
            self.remaining_duration -= 1
            if self.remaining_duration <= 0:
                self.is_active = False
    
    def submit_prediction(self,agent_id: str,prediction: Prediction) -> bool:
        if not self.is_active:
            return False
        
        if not prediction.validate():
            return False
        
        if prediction.confidence_level < self.min_confidence:
            return False
        
        if self.require_explanation and not prediction.explanation:
            return False
        
        if agent_id in self.current_predictions:
            return False
        
        self.current_predictions[agent_id] = prediction
        
        return True
    
    def evaluate_predictions(self, actual_values: List[float]) -> List[PredictionResult]:
        """ Returns list of prediction results """
        results = []
        
        for agent_id, prediction in self.current_predictions.items():

            accuracy = self._calculate_accuracy(prediction.predicted_values,actual_values)
            confidence_calibration = self._calculate_confidence_calibration(prediction,actual_values)
            reward_multiplier = self._calculate_reward_multiplier(accuracy,confidence_calibration)
            
            villain_points = 0
            if accuracy >= self.villain_points_threshold:
                villain_points = self.villain_points_per_prediction
            
            is_king = False
            if accuracy == max(self._calculate_accuracy(p.predicted_values, actual_values)for p in self.current_predictions.values()):
                is_king = True
                self.current_king = agent_id
                reward_multiplier *= self.king_reward_multiplier
            
            result = PredictionResult(prediction,actual_values,accuracy,confidence_calibration,reward_multiplier,villain_points,is_king)
            results.append(result)
            self._update_leaderboard(agent_id, result)
        
        self.prediction_results.extend(results)
        self.current_predictions.clear()

        return results
    
    def _calculate_accuracy(self,predicted_values: List[float],actual_values: List[float]) -> float:
        """Calculate accuracy score using mean absolute percentage error"""
        if len(predicted_values) != len(actual_values):
            return 0.0
        total_error = 0.0
        for pred, actual in zip(predicted_values, actual_values):
            if actual != 0:
                error = abs(pred - actual) / abs(actual)
                total_error += error

        accuracy = max(0.0, 1.0 - total_error / len(predicted_values))
        return accuracy
    
    def _calculate_confidence_calibration(self,prediction: Prediction,actual_values: List[float]) -> float:
        """
        Calculate how well confidence interval matched actual values
        Returns calibration score (0.0 to 1.0)
        """
        lower_bound, upper_bound = prediction.confidence_interval
        
        within_interval = 0
        for actual in actual_values:
            if lower_bound <= actual <= upper_bound:
                within_interval += 1
        
        expected_proportion = prediction.confidence_level
        actual_proportion = within_interval / len(actual_values)
        return max(0.0, 1.0 - abs(actual_proportion - expected_proportion))
    
    def _calculate_reward_multiplier(self,accuracy: float,confidence_calibration: float) -> float:

        base_multiplier = 1.0
        if accuracy > 0.9:
            base_multiplier *= 1.5
        elif accuracy > 0.8:
            base_multiplier *= 1.3
        elif accuracy > 0.7:
            base_multiplier *= 1.1
        
        if confidence_calibration > 0.9:
            base_multiplier *= 1.2
        elif confidence_calibration > 0.8:
            base_multiplier *= 1.1
        
        return base_multiplier
    
    def _update_leaderboard(self, agent_id: str, result: PredictionResult):
        if agent_id not in self.leaderboard:
            self.leaderboard[agent_id] = LeaderboardEntry(agent_id,0,0,0.0,0.0,0.0,0,0,0.0)
        
        entry = self.leaderboard[agent_id]
        entry.total_predictions += 1
        if result.accuracy_score >= 0.7:  # Consider 70%+ as "correct"
            entry.correct_predictions += 1
        
        entry.accuracy = entry.correct_predictions / entry.total_predictions
        entry.average_confidence = (entry.average_confidence * (entry.total_predictions - 1) +result.prediction.confidence_level) / entry.total_predictions
        entry.confidence_calibration = (entry.confidence_calibration * (entry.total_predictions - 1) +result.confidence_calibration) / entry.total_predictions
        
        entry.total_villain_points += result.villain_points_earned
        
        if result.is_king:
            entry.king_count += 1
        
        entry.reputation_score = self._calculate_reputation_score(entry)
    
    def _calculate_reputation_score(self, entry: LeaderboardEntry) -> float:
        accuracy_weight = 0.4
        calibration_weight = 0.2
        villain_points_weight = 0.2
        king_weight = 0.2
        score = (entry.accuracy * accuracy_weight +entry.confidence_calibration * calibration_weight +min(1.0, entry.total_villain_points / 50.0) * villain_points_weight +min(1.0, entry.king_count / 10.0) * king_weight)
        return score
    
    def get_ensemble_prediction(self) -> Optional[List[float]]:
        """
        Get ensemble prediction from all submitted predictions.
        Returns aggregated prediction if ensemble is enabled.
        """
        if not self.use_ensemble or not self.current_predictions:
            return None
        predictions = list(self.current_predictions.values())
        if self.ensemble_method == "weighted_average":
            return self._weighted_average_ensemble(predictions)
        elif self.ensemble_method == "voting":
            return self._voting_ensemble(predictions)
        elif self.ensemble_method == "stacking":
            return self._stacking_ensemble(predictions)
        else:
            return self._simple_average_ensemble(predictions)
    
    def _simple_average_ensemble(self, predictions: List[Prediction]) -> List[float]:
        if not predictions:
            return []
        num_predictions = len(predictions)
        prediction_length = len(predictions[0].predicted_values)
        ensemble = [0.0] * prediction_length
        for pred in predictions:
            for i, value in enumerate(pred.predicted_values):
                ensemble[i] += value
        return [value / num_predictions for value in ensemble]
    
    def _weighted_average_ensemble(self, predictions: List[Prediction]) -> List[float]:
        """Weighted average based on confidence levels"""
        if not predictions:
            return []
        total_weight = sum(pred.confidence_level for pred in predictions)
        if total_weight == 0:
            return self._simple_average_ensemble(predictions)
        prediction_length = len(predictions[0].predicted_values)
        ensemble = [0.0] * prediction_length
        for pred in predictions:
            weight = pred.confidence_level / total_weight
            for i, value in enumerate(pred.predicted_values):
                ensemble[i] += value * weight
        return ensemble
    
    def _voting_ensemble(self, predictions: List[Prediction]) -> List[float]:
        """Voting-based ensemble using median"""
        if not predictions:
            return []
        prediction_length = len(predictions[0].predicted_values)
        ensemble = []
        for i in range(prediction_length):
            values_at_step = [pred.predicted_values[i] for pred in predictions]
            ensemble.append(np.median(values_at_step))
        return ensemble
    
    def _stacking_ensemble(self, predictions: List[Prediction]) -> List[float]:
        """
        Stacking ensemble using reputation-based weighting.
        More sophisticated method using leaderboard reputation.
        """
        if not predictions:
            return []
        total_reputation = 0.0
        for pred in predictions:
            entry = self.leaderboard.get(pred.agent_id)
            if entry:
                total_reputation += entry.reputation_score
            else:
                total_reputation += 0.5  # Default for new agents
        if total_reputation == 0:
            return self._simple_average_ensemble(predictions)
        prediction_length = len(predictions[0].predicted_values)
        ensemble = [0.0] * prediction_length
        for pred in predictions:
            entry = self.leaderboard.get(pred.agent_id)
            weight = entry.reputation_score / total_reputation if entry else 0.5 / total_reputation
            for i, value in enumerate(pred.predicted_values):
                ensemble[i] += value * weight
        return ensemble
    
    def get_leaderboard(self, top_n: int = 10) -> List[LeaderboardEntry]:
        return sorted(self.leaderboard.values(),key=lambda x: x.reputation_score,reverse=True)[:top_n]
    
    def get_market_consensus(self) -> Optional[Dict[str, Any]]:
        """ Returns aggregated market view """
        if not self.current_predictions:
            return None
        ensemble = self.get_ensemble_prediction()
        if not ensemble:
            return None
        predictions = list(self.current_predictions.values())
        all_values = [value for pred in predictions for value in pred.predicted_values]
        
        return {'ensemble_prediction': ensemble,'mean_prediction': np.mean(all_values),'std_prediction': np.std(all_values),'num_predictions': len(predictions),'average_confidence': np.mean([pred.confidence_level for pred in predictions]),'consensus_strength': 1.0 - (np.std(all_values) / np.mean(all_values)) if np.mean(all_values) > 0 else 0.0}
    
    def get_non_participants_penalty(self, all_agent_ids: List[str]) -> Dict[str, float]:
        penalties = {}
        for agent_id in all_agent_ids:
            if agent_id not in self.current_predictions:
                penalties[agent_id] = -0.5
        return penalties
    
    def reset(self):
        self.is_active = False
        self.remaining_duration = 0
        self.step_counter = 0
        self.current_predictions.clear()
        self.prediction_results.clear()
        self.current_king = None
    
    def get_state(self) -> Dict[str, Any]:
        return {'is_active': self.is_active,'remaining_duration': self.remaining_duration,'step_counter': self.step_counter,'num_predictions': len(self.current_predictions),'current_king': self.current_king,'leaderboard_size': len(self.leaderboard),'market_consensus': self.get_market_consensus()}
