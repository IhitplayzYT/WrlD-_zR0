""" Incremental reward uses log returns per step with explicit component logging """
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class RewardComponents:
    pnl: float = 0.0
    forecast: float = 0.0
    preference: float = 0.0
    drawdown: float = 0.0
    damage: float = 0.0  # Disturbance
    kaiser_damage: float = 0.0  # Kaiser damage part of the capped damage reward (adversarial only)


class IncrementalRewardCalculator:
    """Calculates incremental rewards with component logging.
    Adversarial-aware: applies different reward structures for adversarial agents
    to account for their market manipulation goals.
    """
    
    def __init__(self, pnl_weight: float = 1.0,forecast_weight: float = 0.5,preference_weight: float = 0.3,drawdown_penalty: float = 0.5,adversarial_damage_weight: float = 0.4):
        self.pnl_weight = pnl_weight
        self.forecast_weight = forecast_weight
        self.preference_weight = preference_weight
        self.drawdown_penalty = drawdown_penalty
        self.adversarial_damage_weight = adversarial_damage_weight
        self.return_unit = 0.01
    
    def calculate_step_reward(self,old_value: float,new_value: float,old_drawdown: float,new_drawdown: float,forecast_skill: float = 0.0,preference_score: float = 0.0,is_adversarial: bool = False,market_damage: float = 0.0,kaiser_damage_bonus: float = 0.0,agent_characteristics: dict = None) -> tuple[float, RewardComponents]:
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
            kaiser_damage_bonus: Causal bonus for the Kaiser's loss while the agent's own shock is in force (adversarial only)
            agent_characteristics: Dict with heterogeneous agent traits (risk_pref, aggression_style, etc.)
        
        Returns:
            total_reward: Combined reward
            components: Individual reward components
        """
        values = [old_value, new_value, old_drawdown, new_drawdown, forecast_skill, preference_score, market_damage, kaiser_damage_bonus]
        if not np.isfinite(values).all() or old_value <= 0 or new_value <= 0: raise ValueError("Rewards require finite inputs and positive portfolio values")
        pnl = np.log(new_value / old_value) / self.return_unit
        # Drawdown penalty (only for increases in drawdown)
        drawdown = -self.drawdown_penalty * np.maximum(new_drawdown - old_drawdown, 0.) / self.return_unit
        if agent_characteristics is None: agent_characteristics = {}
        risk_pref = agent_characteristics.get('risk_preference', 0.5)
        aggression_style = agent_characteristics.get('aggression_style', 'balanced')
        if is_adversarial:
            pnl_weight_adj = 0.4 + 0.3 * risk_pref  # Higher PnL focus
            damage_weight_adj = 0.2 + 0.2 * (1.0 - risk_pref)  # Lower damage focus
            if aggression_style == 'front_run':
                damage_weight_adj *= 1.1
                pnl_weight_adj *= 1.15
            elif aggression_style == 'liquidity_trap':
                damage_weight_adj *= 1.2
                pnl_weight_adj *= 1.1
            elif aggression_style == 'momentum_reverse':
                damage_weight_adj *= 1.05
                pnl_weight_adj *= 1.1
            elif aggression_style == 'volatility_spike':
                damage_weight_adj *= 1.1
                pnl_weight_adj *= 1.05
            
            # Normalize
            total_weight = pnl_weight_adj + 0.5 + 0.2 + damage_weight_adj
            pnl_weight_adj /= total_weight
            damage_weight_adj /= total_weight
            
            # Cap damage reward to prevent excessive damage-seeking behavior.  The capped
            # amount is split between ordinary disturbance and the Kaiser term so that the
            # logged components always sum to the reward actually paid.
            raw_damage = market_damage + kaiser_damage_bonus
            damage_reward = min(damage_weight_adj * raw_damage, 0.5)
            kaiser_part = damage_reward * kaiser_damage_bonus / raw_damage if raw_damage > 0 else 0.0
            # No survival bonus: an adversary that does nothing must earn nothing (the
            # "no intervention, no reward" invariant is checked by the test-suite).
            pnl_term = pnl_weight_adj * self.pnl_weight * pnl
            forecast_term = 0.5 * self.forecast_weight * forecast_skill
            preference_term = 0.2 * self.preference_weight * preference_score
            total = pnl_term + forecast_term + preference_term + drawdown + damage_reward
            components = RewardComponents(pnl=pnl_term,forecast=forecast_term,preference=preference_term,drawdown=drawdown,damage=damage_reward - kaiser_part,kaiser_damage=kaiser_part)
        else:
            pnl_weight_adj = self.pnl_weight * (0.8 + 0.4 * risk_pref)
            drawdown_adj = self.drawdown_penalty * (1.0 + 0.5 * (1.0 - risk_pref))
            drawdown_term = -drawdown_adj * np.maximum(new_drawdown - old_drawdown, 0.) / self.return_unit
            total = (pnl_weight_adj * pnl + self.forecast_weight * forecast_skill + self.preference_weight * preference_score + drawdown_term)
            components = RewardComponents(pnl=pnl_weight_adj * pnl,forecast=self.forecast_weight * forecast_skill,preference=self.preference_weight * preference_score,drawdown=drawdown_term)
        
        return total, components
    
    def calculate_forecast_skill(self,individual_mse: float,baseline_mse: float) -> float:
        """Calculate forecast skill as tanh of MSE improvement over baseline.
        Positive scores require beating persistence on that realized window.
        """
        return float(np.tanh((baseline_mse - individual_mse) / max(baseline_mse, 1e-8)))
    
    def aggregate_components(self, components_list: List[RewardComponents]) -> RewardComponents:
        total = RewardComponents()
        for comp in components_list:
            total.pnl += comp.pnl
            total.forecast += comp.forecast
            total.preference += comp.preference
            total.drawdown += comp.drawdown
            total.damage += comp.damage
            total.kaiser_damage += comp.kaiser_damage
        return total
    
    def state(self) -> Dict[str, Any]: return {'pnl_weight': self.pnl_weight,'forecast_weight': self.forecast_weight,'preference_weight': self.preference_weight,'drawdown_penalty': self.drawdown_penalty}
    
    def restore(self, state: Dict[str, Any]):
        self.pnl_weight = state['pnl_weight']
        self.forecast_weight = state['forecast_weight']
        self.preference_weight = state['preference_weight']
        self.drawdown_penalty = state['drawdown_penalty']
