from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from enum import Enum
from datetime import datetime
from src.utils.logging import JSONLogger, get_logger, LogCategory

class PredictionType(Enum):
    PRICE_LEVEL = "price_level"
    REGIME_TRANSITION = "regime_transition"
    TREND_DIRECTION = "trend_direction"
    VOLATILITY_LEVEL = "volatility_level"

@dataclass
class Prediction:
    agent_id: str
    pred_t: PredictionType
    pred_val: List[float]  # Predicted price path for k days
    conf_interv: Tuple[float, float]  # (lower_bound, upper_bound)
    conf_lvl: float 
    timestamp: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> bool:
        if not self.pred_val or len(self.conf_interv) != 2 or self.conf_interv[0] >= self.conf_interv[1] or  not 0.0 <= self.confidence_level <= 1.0:
            return False
        return True

@dataclass
class PredictionResult:
    prediction: Prediction
    actual: List[float]
    accuracy: float
    conf_accuracy: float  # How well confidence matched accuracy
    rwrd_factor: float = 1.0
    villion_pt_earned: int = 0
    is_king: bool = False

@dataclass
class PredictionMarket:
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
            payouts[agent_id] = bet['amount'] * 2.0 if (outcome == "correct" and prediction_id in correct_predictions) or (outcome == "incorrect" and prediction_id not in correct_predictions) else  0.0
        self.active_bets.clear()
        return payouts

@dataclass
class LeaderboardEntry:
    agent_id: str
    total_preds: int
    correct_preds: int
    accuracy: float
    avg_conf: float
    conf_accuracy: float
    total_vp: int
    n_king: int
    reputation: float


class BlackBoxPredictionSystem:
    def __init__(self, duration: int = 5, frequency: int = 20, min_confidence: float = 0.3, enable_prediction_market: bool = True, logger: Optional[JSONLogger] = None):
        self.logger = logger or get_logger()
        self.dur = duration
        self.freq = frequency
        self.min_conf = min_confidence
        self.enable_prediction_market = enable_prediction_market
        self.is_active = False
        self.rem_dur = 0
        self.n_steps = 0
        self.curr_pred: Dict[str, Prediction] = {}
        self.pred_results: List[PredictionResult] = []
        self.pred_market = PredictionMarket() if enable_prediction_market else None
        self.leaderboard: Dict[str, LeaderboardEntry] = {}
        self.curr_king: Optional[str] = None
        self.king_rwrd_factor: float = 2.0
        self.king_hist: List[Tuple[str, int]] = []  # (agent_id, duration)
        self.vp_thresh = 0.8  # Accuracy threshold for villain points
        self.vp_per_pred = 2
        self.use_ensemble = True
        self.ensemble_method = "weighted_average"  # "weighted_average", "voting", "stacking"
    
    def should_trigger(self) -> bool:
        self.n_steps += 1
        if not self.is_active and self.n_steps >= self.freq:
            self.is_active = True
            self.rem_dur = self.dur
            self.n_steps = 0
            return True
        return False
    
    def update(self):
        if self.is_active:
            self.rem_dur -= 1
            if self.rem_dur <= 0:
                self.is_active = False
    
    def submit_prediction(self,agent_id: str,prediction: Prediction) -> bool:
        if not self.is_active:
            return False
        if not prediction.validate():
            return False
        if prediction.conf_lvl < self.min_conf:
            return False
        if agent_id in self.current_predictions:
            return False
        self.current_predictions[agent_id] = prediction
        return True
    
    def evaluate_predictions(self, actual_values: List[float]) -> List[PredictionResult]:
        results = []
        for agent_id, prediction in self.curr_pred.items():
            accuracy = self._calculate_accuracy(prediction.pred_val,actual_values)
            conf_acc = self._calculate_confidence_calibration(prediction,actual_values)
            reward_multiplier = self._calculate_reward_multiplier(accuracy,conf_acc)
            vp = 0
            if accuracy >= self.vp_thresh:
                vp = self.vp_per_pred
            is_king = False
            
            if accuracy == max(self._calculate_accuracy(prediction.pred_val, actual_values) for p in self.curr_pred.values()):
                is_king = True
                self.curr_king = agent_id
                reward_multiplier *= self.king_rwrd_factor
            
            result = PredictionResult(prediction,actual_values,accuracy,conf_acc,reward_multiplier,vp,is_king)
            results.append(result)
            self._update_leaderboard(agent_id, result)
        self.prediction_results.extend(results)
        self.current_predictions.clear()
        return results
    
    def _calculate_accuracy(self,predicted_values: List[float],actual_values: List[float]) -> float:
        if len(predicted_values) != len(actual_values):
            return 0.0
        tot_err = 0.0
        for pred, actual in zip(predicted_values, actual_values):
            if actual != 0:
                tot_err += abs(pred - actual) / abs(actual)
        return max(0.0, 1.0 - tot_err / len(predicted_values))
    
    def _calculate_confidence_calibration(self,prediction: Prediction,actual_values: List[float]) -> float:
        lb, ub = prediction.confidence_interval
        in_interv = 0
        for actual in actual_values:
            if lb <= actual <= ub:
                in_interv += 1
        expected_ratio = prediction.conf_lvl
        actual_ratio = in_interv / len(actual_values)
        return max(0.0, 1.0 - abs(actual_ratio - expected_ratio))
    
    def _calculate_reward_multiplier(self,accuracy: float,confidence_calibration: float) -> float:
        ret = 1.0
        if accuracy > 0.9:
            ret *= 1.5
        elif accuracy > 0.8:
            ret *= 1.3
        elif accuracy > 0.7:
            ret *= 1.1
        
        if confidence_calibration > 0.9:
            ret *= 1.2
        elif confidence_calibration > 0.8:
            ret *= 1.1
        return ret 
    
    def _update_leaderboard(self, agent_id: str, result: PredictionResult):
        if agent_id not in self.leaderboard:
            self.leaderboard[agent_id] = LeaderboardEntry(agent_id,0,0,0.0,0.0,0.0,0,0,0.0)
        entry = self.leaderboard[agent_id]
        entry.total_preds += 1
        if result.accuracy >= 0.7:
            entry.correct_preds += 1
        entry.accuracy = entry.correct_preds / entry.total_preds
        entry.avg_conf = (entry.avg_conf * (entry.total_preds - 1) + result.prediction.conf_lvl) / entry.total_preds
        entry.conf_accuracy = (entry.conf_accuracy * (entry.total_preds - 1) + result.conf_accuracy) / entry.total_preds
        entry.total_vp += result.villion_pt_earned
        if result.is_king:
            entry.king_count += 1
        entry.reputation = self._calculate_reputation_score(entry)
    
    def _calculate_reputation_score(self, entry: LeaderboardEntry) -> float:
        accuracy_weight = 0.4
        calibration_weight = 0.2
        villain_points_weight = 0.2
        king_weight = 0.2
        return (entry.accuracy * accuracy_weight + entry.conf_accuracy * calibration_weight + min(1.0, entry.total_vp / 50.0) * villain_points_weight + min(1.0, entry.n_king / 10.0) * king_weight)
    
    def get_ensemble_prediction(self) -> Optional[List[float]]:
        """ Returns aggregated prediction """
        if not self.use_ensemble or not self.curr_pred:
            return None
        predictions = list(self.curr_pred.values())
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
        num_preds = len(predictions)
        pred_l = len(predictions[0].pred_val)
        ensemble = [0.0] * pred_l 
        for pred in predictions:
            for i, value in enumerate(pred.predicted_values):
                ensemble[i] += value
        return [value / num_preds for value in ensemble]
    
    def _weighted_average_ensemble(self, predictions: List[Prediction]) -> List[float]:
        if not predictions:
            return []
        tot_weight = sum(pred.conf_lvl for pred in predictions)
        if tot_weight == 0:
            return self._simple_average_ensemble(predictions)
        pred_l = len(predictions[0].pred_val)
        ensemble = [0.0] * pred_l 
        for pred in predictions:
            weight = pred.conf_lvl / tot_weight 
            for i, value in enumerate(pred.pred_val):
                ensemble[i] += value * weight
        return ensemble
    
    def _voting_ensemble(self, predictions: List[Prediction]) -> List[float]:
        if not predictions:
            return []
        pred_l = len(predictions[0].pred_val)
        ensemble = []
        for i in range(pred_l):
            values_at_step = [pred.pred_val[i] for pred in predictions]
            ensemble.append(np.median(values_at_step))
        return ensemble
    
    def _stacking_ensemble(self, predictions: List[Prediction]) -> List[float]:
        """ Use reputation based weighting """
        if not predictions:
            return []
        total_rep = 0.0
        for pred in predictions:
            entry = self.leaderboard.get(pred.agent_id)
            total_reputation += entry.reputation if entry else 0.5
        pred_l = len(predictions[0].pred_val)
        ensemble = [0.0] * pred_l 
        for pred in predictions:
            entry = self.leaderboard.get(pred.agent_id)
            weight = (entry.reputation / total_reputation) if entry else (0.5 / total_reputation)
            for i, value in enumerate(pred.pred_val):
                ensemble[i] += value * weight
        return ensemble
    
    def get_leaderboard(self, top_n: int = 10) -> List[LeaderboardEntry]:
        return sorted(self.leaderboard.values(),key=lambda x: x.reputation,reverse=True)[:top_n]
    
    def get_market_consensus(self) -> Optional[Dict[str, Any]]:
        if not self.curr_pred:
            return None
        ensemble = self.get_ensemble_prediction()
        if not ensemble:
            return None
        predictions = list(self.curr_pred.values())
        all_values = [value for pred in predictions for value in pred.pred_val]
        return {'ensemble_prediction': ensemble,'mean_prediction': np.mean(all_values),'std_prediction': np.std(all_values),'num_predictions': len(predictions),'average_confidence': np.mean([pred.conf_lvl for pred in predictions]),'consensus_strength': 1.0 - (np.std(all_values) / np.mean(all_values)) if np.mean(all_values) > 0 else 0.0}
    
    def get_non_participants_penalty(self, all_agent_ids: List[str]) -> Dict[str, float]:
        penalties = {}
        for agent_id in all_agent_ids:
            if agent_id not in self.curr_pred:
                penalties[agent_id] = -0.5
        return penalties
    
    def reset(self):
        self.is_active = False
        self.remaining_duration = self.step_counter = 0
        self.current_predictions.clear()
        self.prediction_results.clear()
        self.current_king = None
    
    def get_state(self) -> Dict[str, Any]:
        return {'is_active': self.is_active,'remaining_duration': self.rem_dur,'step_counter': self.n_steps,'num_predictions': len(self.curr_pred),'current_king': self.curr_king,'leaderboard_size': len(self.leaderboard),'market_consensus': self.get_market_consensus()}