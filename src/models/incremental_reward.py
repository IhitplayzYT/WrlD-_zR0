"""Incremental reward design from market_ensemble_v2.
Uses log returns per step with explicit component logging.
"""
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Explicit reward component tracking."""
    pnl: float = 0.0
    forecast: float = 0.0
    preference: float = 0.0
    drawdown: float = 0.0
    damage: float = 0.0  # Market damage (for adversarial agents)


class IncrementalRewardCalculator:
    """Calculates incremental rewards with component logging.
    
    Adversarial-aware: applies different reward structures for adversarial agents
    to account for their market manipulation goals.
    """
    
    def __init__(self, 
                 pnl_weight: float = 1.0,
                 forecast_weight: float = 0.5,
                 preference_weight: float = 0.3,
                 drawdown_penalty: float = 0.5,
                 adversarial_damage_weight: float = 0.4):
        self.pnl_weight = pnl_weight
        self.forecast_weight = forecast_weight
        self.preference_weight = preference_weight
        self.drawdown_penalty = drawdown_penalty
        self.adversarial_damage_weight = adversarial_damage_weight
        self.return_unit = 0.01
    
    def calculate_step_reward(self,
                             old_value: float,
                             new_value: float,
                             old_drawdown: float,
                             new_drawdown: float,
                             forecast_skill: float = 0.0,
                             preference_score: float = 0.0,
                             is_adversarial: bool = False,
                             market_damage: float = 0.0) -> tuple[float, RewardComponents]:
        """Calculate incremental step reward with component breakdown.
        
        Args:
            old_value: Portfolio value before step
            new_value: Portfolio value after step
            old_drawdown: Drawdown before step
            new_drawdown: Drawdown after step
            forecast_skill: Contest skill score (tanh of MSE improvement)
            preference_score: Preference model score difference
            is_adversarial: Whether the agent is adversarial
            market_damage: Damage caused to market (for adversarial agents)
        
        Returns:
            total_reward: Combined reward
            components: Individual reward components
        """
        # Incremental log return (not cumulative)
        pnl = np.log(new_value / old_value) / self.return_unit
        
        # Drawdown penalty (only for increases in drawdown)
        drawdown = -self.drawdown_penalty * np.maximum(new_drawdown - old_drawdown, 0.)
        
        if is_adversarial:
            # Adversarial agents: reward for market damage + reduced emphasis on PnL
            # They want to destabilize the market, not necessarily profit
            damage_reward = self.adversarial_damage_weight * market_damage
            total = (0.3 * self.pnl_weight * pnl +  # Reduced PnL weight
                     0.5 * self.forecast_weight * forecast_skill +  # Forecast skill still matters
                     0.2 * self.preference_weight * preference_score +  # Reduced preference
                     drawdown +  # Drawdown penalty still applies
                     damage_reward)  # Market damage reward
            
            components = RewardComponents(
                pnl=0.3 * self.pnl_weight * pnl,
                forecast=0.5 * self.forecast_weight * forecast_skill,
                preference=0.2 * self.preference_weight * preference_score,
                drawdown=drawdown,
                damage=damage_reward
            )
        else:
            # RLAIF agents: standard reward structure
            total = (self.pnl_weight * pnl + 
                     self.forecast_weight * forecast_skill + 
                     self.preference_weight * preference_score + 
                     drawdown)
            
            components = RewardComponents(
                pnl=self.pnl_weight * pnl,
                forecast=self.forecast_weight * forecast_skill,
                preference=self.preference_weight * preference_score,
                drawdown=drawdown,
                damage=0.0
            )
        
        return total, components
    
    def calculate_forecast_skill(self,
                                 individual_mse: float,
                                 baseline_mse: float) -> float:
        """Calculate forecast skill as tanh of MSE improvement over baseline.
        
        Positive scores require beating persistence on that realized window.
        """
        return np.tanh(baseline_mse - individual_mse)
    
    def aggregate_components(self, components_list: List[RewardComponents]) -> RewardComponents:
        """Aggregate reward components over multiple steps."""
        total = RewardComponents()
        for comp in components_list:
            total.pnl += comp.pnl
            total.forecast += comp.forecast
            total.preference += comp.preference
            total.drawdown += comp.drawdown
            total.damage += comp.damage
        return total
    
    def state(self) -> Dict[str, Any]:
        return {
            'pnl_weight': self.pnl_weight,
            'forecast_weight': self.forecast_weight,
            'preference_weight': self.preference_weight,
            'drawdown_penalty': self.drawdown_penalty
        }
    
    def restore(self, state: Dict[str, Any]):
        self.pnl_weight = state['pnl_weight']
        self.forecast_weight = state['forecast_weight']
        self.preference_weight = state['preference_weight']
        self.drawdown_penalty = state['drawdown_penalty']
