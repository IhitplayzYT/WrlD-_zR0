from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import re
from collections import Counter
from src.utils.logging import get_logger, LogCategory

logger = get_logger()


class SummaryType(Enum):
    MARKET_ACTIVITY = "market_activity"
    AGENT_BEHAVIOR = "agent_behavior"
    EVENT_IMPACT = "event_impact"
    TREND_ANALYSIS = "trend_analysis"
    OVERVIEW = "overview"


@dataclass
class Summary:

    summary_type: SummaryType
    content: str
    key_points: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    generated_at: datetime = field(default_factory=datetime.now)


class SummaryModel:
    """ Generate summaries and commentary seen ONLY BY THE USER """
    def __init__(self):
        self.templates = self._initialize_templates()
        self.patterns = self._initialize_patterns()
    
    def _initialize_templates(self) -> Dict[SummaryType, List[str]]:
        return {
            SummaryType.MARKET_ACTIVITY: [
                "Market activity has been {trend} with {volatility} volatility. "
                "Trading volume reached {volume} with {direction} price movement.",
                "The market showed {sentiment} sentiment, with prices moving {direction} "
                "by {change}%. Volume was {volume} indicating {liquidity} liquidity."
            ],
            SummaryType.AGENT_BEHAVIOR: [
                "Agent {agent_id} displayed {behavior} behavior, executing {n_trades} trades "
                "with an average profit of {avg_profit}. The agent's current strategy "
                "is {strategy}.",
                "{agent_id} showed {performance} performance, achieving a total return "
                "of {total_return}. The agent's risk tolerance is {risk_tolerance}."
            ],
            SummaryType.EVENT_IMPACT: [
                "The {event_type} event had {impact} impact on the market, "
                "causing price {direction} of {change}%. The effect lasted {duration} steps.",
                "Market reacted {reaction} to the {event_type} event, with volatility "
                "increasing by {volatility_change}%. The event affected {affected_sectors} sectors."
            ],
            SummaryType.TREND_ANALYSIS: [
                "Analysis shows a {trend} trend over the last {period} steps. "
                "Key patterns include {patterns}. The trend strength is {strength}.",
                "Market regime is currently {regime} with {confidence} confidence. "
                "Expected continuation probability is {probability}."
            ],
            SummaryType.OVERVIEW: [
                "Market overview: {market_state}. Top performer was {top_agent} with {top_return} return. "
                "Market volatility is {volatility} and liquidity is {liquidity}.",
                "Simulation summary: {n_agents} agents participated over {n_steps} steps. "
                "Average agent return was {avg_return}. Market experienced {n_events} significant events."
            ]
        }
    
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize pattern recognition patterns"""
        return {
            'uptrend': ['increasing', 'rising', 'bullish', 'positive'],
            'downtrend': ['decreasing', 'falling', 'bearish', 'negative'],
            'high_volatility': ['volatile', 'unstable', 'fluctuating'],
            'low_volatility': ['stable', 'calm', 'steady'],
            'high_volume': ['heavy', 'active', 'busy'],
            'low_volume': ['light', 'thin', 'quiet']
        }
    
    def summarize_market_activity(self,market_state: Dict[str, Any],price_history: List[float],volume: float,n_steps: int) -> Summary:
        if len(price_history) > 1:
            price_change = (price_history[-1] - price_history[0]) / price_history[0] * 100
            direction = "upward" if price_change > 0 else "downward"
            change = abs(price_change)
        else:
            direction = "stable"
            change = 0.0
        volatility = market_state.get('volatility', 0.02)
        volatility_desc = "high" if volatility > 0.03 else "low"
        volume_desc = "high" if volume > 1000 else "low"
        sentiment = "positive" if direction == "upward" else "negative"
        template = self.templates[SummaryType.MARKET_ACTIVITY][0]
        content = template.format(trend=direction,volatility=volatility_desc,volume=volume,direction=direction,change=f"{change:.2f}",sentiment=sentiment,liquidity=volume_desc)
        key_points = [f"Price moved {direction} by {change:.2f}%",f"Volatility: {volatility_desc}",f"Volume: {volume_desc}"]
        return Summary(summary_type=SummaryType.MARKET_ACTIVITY,content=content,key_points=key_points,metrics={'price_change': price_change,'volatility': volatility,'volume': volume})
    
    def summarize_agent_behavior(self,agent_id: str,agent_state: Dict[str, Any],trade_history: List[Dict],performance: Dict[str, float]) -> Summary:
        n_trades = len(trade_history)
        total_return = performance.get('total_return', 0.0)
        avg_profit = performance.get('avg_profit', 0.0)
        
        if total_return > 0.1:
            behavior = "aggressive profitable"
        elif total_return > 0:
            behavior = "conservative profitable"
        elif total_return > -0.1:
            behavior = "cautious"
        else:
            behavior = "loss-making"
        
        performance_desc = "strong" if total_return > 0.05 else "moderate" if total_return > 0 else "weak"
        risk_tolerance = agent_state.get('risk_tolerance', 0.5)
        risk_desc = "high" if risk_tolerance > 0.7 else "moderate" if risk_tolerance > 0.3 else "low"
        template = self.templates[SummaryType.AGENT_BEHAVIOR][0]
        content = template.format(agent_id=agent_id,behavior=behavior,n_trades=n_trades,avg_profit=f"{avg_profit:.2f}",strategy="momentum-based",performance=performance_desc,total_return=f"{total_return:.2%}",risk_tolerance=risk_desc)
        key_points = [f"Executed {n_trades} trades",f"Total return: {total_return:.2%}",f"Risk tolerance: {risk_desc}"]
        return Summary(summary_type=SummaryType.AGENT_BEHAVIOR,content=content,key_points=key_points,metrics=performance)
    
    def summarize_event_impact(self,event_type: str,market_before: Dict[str, Any],market_after: Dict[str, Any],duration: int) -> Summary:
        price_before = market_before.get('current_price', 100.0)
        price_after = market_after.get('current_price', 100.0)
        price_change = (price_after - price_before) / price_before * 100
        volatility_before = market_before.get('volatility', 0.02)
        volatility_after = market_after.get('volatility', 0.02)
        volatility_change = (volatility_after - volatility_before) / volatility_before * 100 if volatility_before > 0 else 0
        direction = "upward" if price_change > 0 else "downward"
        impact = "significant" if abs(price_change) > 2 else "moderate" if abs(price_change) > 0.5 else "minimal"
        reaction = "strongly" if abs(price_change) > 2 else "moderately" if abs(price_change) > 0.5 else "mildly"
        template = self.templates[SummaryType.EVENT_IMPACT][0]
        content = template.format(event_type=event_type,impact=impact,direction=direction,change=f"{abs(price_change):.2f}",duration=duration,reaction=reaction,volatility_change=f"{volatility_change:.1f}",affected_sectors="technology")
        key_points = [f"Price moved {direction} by {abs(price_change):.2f}%",f"Volatility changed by {volatility_change:.1f}%",f"Duration: {duration} steps"]
        return Summary(summary_type=SummaryType.EVENT_IMPACT,content=content,key_points=key_points,metrics={'price_change': price_change,'volatility_change': volatility_change,'duration': duration})
    
    def summarize_trend_analysis(self,price_history: List[float],regime: str,regime_confidence: float) -> Summary:
        if len(price_history) < 10:
            trend = "insufficient_data"
            strength = 0.0
        else:
            recent_prices = price_history[-10:]
            if recent_prices[-1] > recent_prices[0]:
                trend = "uptrend"
            elif recent_prices[-1] < recent_prices[0]:
                trend = "downtrend"
            else:
                trend = "sideways"
            price_range = max(recent_prices) - min(recent_prices)
            avg_price = sum(recent_prices) / len(recent_prices)
            strength = price_range / avg_price if avg_price > 0 else 0.0
        
        strength_desc = "strong" if strength > 0.05 else "moderate" if strength > 0.02 else "weak"
        confidence_desc = "high" if regime_confidence > 0.7 else "moderate" if regime_confidence > 0.5 else "low"
        template = self.templates[SummaryType.TREND_ANALYSIS][0]
        content = template.format(trend=trend,period=10,patterns="price momentum",strength=strength_desc,regime=regime,confidence=confidence_desc,probability=f"{regime_confidence:.1%}")
        key_points = [f"Current trend: {trend}",f"Trend strength: {strength_desc}",f"Regime: {regime}"]
        return Summary(summary_type=SummaryType.TREND_ANALYSIS,content=content,key_points=key_points,metrics={'trend_strength': strength,'regime_confidence': regime_confidence},confidence=regime_confidence)
    
    def generate_overview(self,agent_performances: Dict[str, float],market_state: Dict[str, Any],n_events: int,n_steps: int) -> Summary:
        if agent_performances:
            top_agent = max(agent_performances, key=agent_performances.get)
            top_return = agent_performances[top_agent]
            avg_return = sum(agent_performances.values()) / len(agent_performances)
        else:
            top_agent = "N/A"
            top_return = 0.0
            avg_return = 0.0
        volatility = market_state.get('volatility', 0.02)
        liquidity = market_state.get('liquidity', 0.5)
        volatility_desc = "high" if volatility > 0.03 else "moderate" if volatility > 0.01 else "low"
        liquidity_desc = "high" if liquidity > 0.7 else "moderate" if liquidity > 0.3 else "low"
        market_state_desc = f"volatility is {volatility_desc} and liquidity is {liquidity_desc}"
        template = self.templates[SummaryType.OVERVIEW][0]
        content = template.format(market_state=market_state_desc,top_agent=top_agent,top_return=f"{top_return:.2%}",volatility=volatility_desc,liquidity=liquidity_desc,n_agents=len(agent_performances),n_steps=n_steps,avg_return=f"{avg_return:.2%}",n_events=n_events)
        key_points = [f"Top performer: {top_agent} ({top_return:.2%})",f"Average return: {avg_return:.2%}",f"Events: {n_events}"]
        return Summary(summary_type=SummaryType.OVERVIEW,content=content,key_points=key_points,metrics={'top_return': top_return,'avg_return': avg_return,'n_events': n_events})
    
    def generate_commentary(self,summaries: List[Summary],max_length: int = 500) -> str:
        if not summaries:
            return "No data available for commentary."
        commentary_parts = []
        current_length = 0
        for summary in summaries:
            if current_length + len(summary.content) > max_length:
                break
            commentary_parts.append(summary.content)
            current_length += len(summary.content)
        all_key_points = []
        for summary in summaries:
            all_key_points.extend(summary.key_points)
        if all_key_points:
            commentary_parts.append("\nKey Points:")
            for point in all_key_points[:5]:  # Limit to top 5
                commentary_parts.append(f"- {point}")
        return "\n\n".join(commentary_parts)


if __name__ == "__main__":
    model = SummaryModel()
    
    market_state = {'volatility': 0.025, 'liquidity': 0.6}
    price_history = [100.0, 101.0, 102.0, 103.0, 104.0]
    
    market_summary = model.summarize_market_activity(market_state, price_history, volume=1500, n_steps=100)
    logger.info(LogCategory.EVALUATION, "Market Activity Summary", data={'content': market_summary.content, 'key_points': market_summary.key_points})
    
    agent_state = {'risk_tolerance': 0.7}
    trade_history = [{'profit': 10}, {'profit': -5}, {'profit': 15}]
    performance = {'total_return': 0.15, 'avg_profit': 6.67}
    
    agent_summary = model.summarize_agent_behavior("agent_0", agent_state, trade_history, performance)
    logger.info(LogCategory.AGENT, "Agent Behavior Summary", data={'content': agent_summary.content, 'key_points': agent_summary.key_points})
    
    trend_summary = model.summarize_trend_analysis(price_history, regime="bull", regime_confidence=0.8)
    logger.info(LogCategory.MARKET, "Trend Analysis Summary", data={'content': trend_summary.content, 'key_points': trend_summary.key_points})
    
    agent_performances = {'agent_0': 0.15, 'agent_1': 0.08, 'agent_2': -0.05}
    overview = model.generate_overview(agent_performances, market_state, n_events=5, n_steps=100)
    logger.info(LogCategory.EVALUATION, "Overview Summary", data={'content': overview.content, 'key_points': overview.key_points})
    
    logger.info(LogCategory.SYSTEM, "Summary model test completed")
