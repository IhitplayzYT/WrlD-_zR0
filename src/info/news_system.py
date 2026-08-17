from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import random
from datetime import datetime, timedelta
import numpy as np


class NewsCategory(Enum):
    """Categories of news events"""
    EARNINGS = "earnings"
    ECONOMIC = "economic"
    POLITICAL = "political"
    REGULATORY = "regulatory"
    MARKET = "market"
    COMPANY = "company"
    TECHNOLOGY = "technology"
    GEOPOLITICAL = "geopolitical"


class Sentiment(Enum):
    """Sentiment classifications"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass
class NewsEvent:
    """
    Individual news event with metadata and sentiment.
    """
    event_id: str
    category: NewsCategory
    headline: str
    content: str
    sentiment: Sentiment
    sentiment_score: float  # -1.0 to 1.0
    timestamp: datetime
    relevance_score: float  # 0.0 to 1.0
    source: str = "market_wire"
    impact_duration: int = 5  # Number of steps the news affects market
    price_impact_estimate: float = 0.0  # Estimated price impact
    volume_impact_estimate: float = 0.0  # Estimated volume impact
    
    def get_sentiment_value(self) -> float:
        """Get numeric sentiment value"""
        sentiment_values = {
            Sentiment.VERY_NEGATIVE: -1.0,
            Sentiment.NEGATIVE: -0.5,
            Sentiment.NEUTRAL: 0.0,
            Sentiment.POSITIVE: 0.5,
            Sentiment.VERY_POSITIVE: 1.0
        }
        return sentiment_values.get(self.sentiment, 0.0)


class SentimentAnalyzer:
    """
    Analyze sentiment of news content.
    Implements rule-based sentiment analysis for simulation.
    """
    
    def __init__(self):
        # Positive and negative word lists for sentiment analysis
        self.positive_words = [
            "growth", "profit", "increase", "rise", "gain", "strong", "positive",
            "beat", "exceed", "surpass", "outperform", "bullish", "optimistic",
            "expansion", "recovery", "success", "innovation", "breakthrough",
            "upgrade", "rally", "surge", "boom", "prosperous", "favorable"
        ]
        
        self.negative_words = [
            "decline", "loss", "decrease", "fall", "drop", "weak", "negative",
            "miss", "below", "underperform", "bearish", "pessimistic", "recession",
            "crisis", "failure", "downgrade", "crash", "plunge", "slump",
            "concern", "risk", "uncertainty", "volatile", "adverse", "trouble"
        ]
        
        self.very_positive_words = ["record", "exceptional", "outstanding", "phenomenal"]
        self.very_negative_words = ["catastrophic", "devastating", "severe", "critical"]
    
    def analyze(self, text: str) -> Tuple[Sentiment, float]:
        """
        Analyze sentiment of text.
        Returns (sentiment_enum, sentiment_score)
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        very_positive_count = sum(1 for word in words if word in self.very_positive_words)
        very_negative_count = sum(1 for word in words if word in self.very_negative_words)
        
        # Calculate sentiment score
        total_words = len(words)
        if total_words == 0:
            return Sentiment.NEUTRAL, 0.0
        
        # Weighted sentiment calculation
        sentiment_score = (
            (positive_count * 0.5) +
            (very_positive_count * 1.0) -
            (negative_count * 0.5) -
            (very_negative_count * 1.0)
        ) / total_words
        
        # Clamp score to [-1, 1]
        sentiment_score = max(-1.0, min(1.0, sentiment_score * 10))  # Amplify for more distinct scores
        
        # Determine sentiment category
        if sentiment_score >= 0.7:
            sentiment = Sentiment.VERY_POSITIVE
        elif sentiment_score >= 0.2:
            sentiment = Sentiment.POSITIVE
        elif sentiment_score > -0.2:
            sentiment = Sentiment.NEUTRAL
        elif sentiment_score > -0.7:
            sentiment = Sentiment.NEGATIVE
        else:
            sentiment = Sentiment.VERY_NEGATIVE
        
        return sentiment, sentiment_score


class NewsGenerator:
    """
    Generate news events based on market conditions and events.
    Creates realistic news narratives tied to simulation state.
    """
    
    def __init__(self, sentiment_analyzer: Optional[SentimentAnalyzer] = None):
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.news_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[NewsCategory, List[Dict]]:
        """Initialize news templates for each category"""
        return {
            NewsCategory.EARNINGS: [
                {"headline": "{company} reports {performance} quarterly earnings", "positive": True},
                {"headline": "{company} earnings {performance} analyst expectations", "positive": False},
                {"headline": "{company} revenue {performance} year-over-year", "positive": True},
            ],
            NewsCategory.ECONOMIC: [
                {"headline": "GDP growth {performance} expectations in Q{quarter}", "positive": True},
                {"headline": "Unemployment rate {performance} to {rate}%", "positive": False},
                {"headline": "Inflation data shows {trend} pressure", "positive": False},
            ],
            NewsCategory.POLITICAL: [
                {"headline": "Government announces new {policy} regulations", "positive": False},
                {"headline": "Trade talks with {country} show {progress}", "positive": True},
                {"headline": "Central bank signals {direction} policy shift", "positive": True},
            ],
            NewsCategory.MARKET: [
                {"headline": "Market {trend} as investors react to {factor}", "positive": True},
                {"headline": "Volatility {trend} amid {condition}", "positive": False},
                {"headline": "Trading volume {trend} on {factor}", "positive": True},
            ],
            NewsCategory.COMPANY: [
                {"headline": "{company} announces strategic partnership with {partner}", "positive": True},
                {"headline": "{company} CEO {action} amid {condition}", "positive": False},
                {"headline": "{company} launches new {product}", "positive": True},
            ],
            NewsCategory.TECHNOLOGY: [
                {"headline": "New {technology} breakthrough announced by {company}", "positive": True},
                {"headline": "Cybersecurity concerns {trend} in {sector}", "positive": False},
                {"headline": "AI adoption {trend} across industries", "positive": True},
            ],
            NewsCategory.GEOPOLITICAL: [
                {"headline": "Tensions {trend} between {country1} and {country2}", "positive": False},
                {"headline": "International trade agreement {progress}", "positive": True},
                {"headline": "Supply chain disruptions {trend} in {region}", "positive": False},
            ],
        }
    
    def generate_news(
        self,
        category: NewsCategory,
        market_state: Dict[str, Any],
        event_context: Optional[Dict] = None
    ) -> NewsEvent:
        """
        Generate a news event based on category and market state.
        """
        templates = self.news_templates.get(category, self.news_templates[NewsCategory.MARKET])
        template = random.choice(templates)
        
        # Fill template with context
        headline = self._fill_template(template["headline"], market_state, event_context)
        content = self._generate_content(headline, category, market_state)
        
        # Analyze sentiment
        sentiment, sentiment_score = self.sentiment_analyzer.analyze(headline + " " + content)
        
        # Adjust sentiment based on template's intended sentiment
        if template["positive"] and sentiment_score < 0:
            sentiment_score = abs(sentiment_score) * 0.5
        elif not template["positive"] and sentiment_score > 0:
            sentiment_score = -abs(sentiment_score) * 0.5
        
        # Recalculate sentiment based on adjusted score
        if sentiment_score >= 0.7:
            sentiment = Sentiment.VERY_POSITIVE
        elif sentiment_score >= 0.2:
            sentiment = Sentiment.POSITIVE
        elif sentiment_score > -0.2:
            sentiment = Sentiment.NEUTRAL
        elif sentiment_score > -0.7:
            sentiment = Sentiment.NEGATIVE
        else:
            sentiment = Sentiment.VERY_NEGATIVE
        
        # Estimate impact
        price_impact = sentiment_score * 0.05  # Max 5% price impact
        volume_impact = abs(sentiment_score) * 0.3  # Max 30% volume impact
        
        return NewsEvent(
            event_id=f"news_{datetime.now().timestamp()}",
            category=category,
            headline=headline,
            content=content,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            timestamp=datetime.now(),
            relevance_score=random.uniform(0.5, 1.0),
            impact_duration=random.randint(3, 10),
            price_impact_estimate=price_impact,
            volume_impact_estimate=volume_impact
        )
    
    def _fill_template(self, template: str, market_state: Dict, event_context: Optional[Dict]) -> str:
        """Fill template with context from market state"""
        replacements = {
            "{company}": random.choice(["TechCorp", "FinanceInc", "GlobalMega", "InnovateTech", "MarketLeader"]),
            "{performance}": random.choice(["beat", "missed", "matched", "exceeded", "fell short of"]),
            "{quarter}": str(random.randint(1, 4)),
            "{rate}": str(random.randint(3, 8)),
            "{trend}": random.choice(["increases", "decreases", "surges", "plunges", "stabilizes"]),
            "{direction}": random.choice(["hawkish", "dovish", "neutral", "aggressive"]),
            "{policy}": random.choice(["financial", "environmental", "trade", "technology"]),
            "{country}": random.choice(["China", "USA", "EU", "Japan", "UK"]),
            "{progress}": random.choice(["progress", "stall", "breakdown", "breakthrough"]),
            "{condition}": random.choice(["uncertainty", "optimism", "concerns", "stability"]),
            "{factor}": random.choice(["earnings", "economic data", "geopolitical events", "regulatory changes"]),
            "{partner}": random.choice(["competitor", "industry leader", "startup", "foreign firm"]),
            "{action}": random.choice(["resigns", "is promoted", "faces scrutiny", "announces strategy"]),
            "{product}": random.choice(["product line", "service", "platform", "technology"]),
            "{technology}": random.choice(["AI", "blockchain", "quantum computing", "5G"]),
            "{sector}": random.choice(["finance", "healthcare", "technology", "energy"]),
            "{country1}": random.choice(["USA", "China", "Russia", "EU"]),
            "{country2}": random.choice(["China", "Russia", "EU", "USA"]),
            "{region}": random.choice(["Asia", "Europe", "Americas", "Middle East"])
        }
        
        filled = template
        for placeholder, replacement in replacements.items():
            filled = filled.replace(placeholder, replacement)
        
        return filled
    
    def _generate_content(self, headline: str, category: NewsCategory, market_state: Dict) -> str:
        """Generate content body for news event"""
        content_templates = {
            NewsCategory.EARNINGS: [
                "The company reported strong performance driven by robust demand in key markets.",
                "Management cited challenging market conditions and increased competition.",
                "Analysts note that the results exceeded expectations in several key metrics."
            ],
            NewsCategory.ECONOMIC: [
                "Economists suggest the data indicates a strengthening economy.",
                "Market participants are concerned about the implications for monetary policy.",
                "The figures suggest continued uncertainty in the economic outlook."
            ],
            NewsCategory.POLITICAL: [
                "Industry analysts expect significant impact on market dynamics.",
                "The announcement comes amid ongoing negotiations between stakeholders.",
                "Experts believe the policy will have gradual effects on the sector."
            ],
            NewsCategory.MARKET: [
                "Traders are closely monitoring developments for further direction.",
                "Market sentiment remains divided on the near-term outlook.",
                "Institutional investors are adjusting portfolios in response."
            ],
            NewsCategory.COMPANY: [
                "The move is seen as part of broader strategic realignment.",
                "Industry observers note the significance of this development.",
                "The announcement has generated significant interest from investors."
            ],
            NewsCategory.TECHNOLOGY: [
                "The innovation is expected to disrupt traditional business models.",
                "Industry experts highlight the potential for widespread adoption.",
                "The development represents a significant advancement in the field."
            ],
            NewsCategory.GEOPOLITICAL: [
                "The situation continues to evolve with uncertain outcomes.",
                "Diplomatic efforts are ongoing to address the underlying issues.",
                "Market participants are assessing the potential economic implications."
            ],
        }
        
        templates = content_templates.get(category, content_templates[NewsCategory.MARKET])
        return random.choice(templates)


class NewsSystem:
    """
    Main news system that manages news generation, storage, and impact.
    """
    
    def __init__(self, max_news_history: int = 100):
        self.max_news_history = max_news_history
        self.news_generator = NewsGenerator()
        self.news_history: List[NewsEvent] = []
        self.active_news: List[NewsEvent] = []
        self.current_step = 0
        
    def generate_news_for_event(
        self,
        event_type: str,
        market_state: Dict[str, Any],
        event_context: Optional[Dict] = None
    ) -> Optional[NewsEvent]:
        """
        Generate news based on market event.
        """
        # Map event types to news categories
        event_to_category = {
            "earnings": NewsCategory.EARNINGS,
            "economic": NewsCategory.ECONOMIC,
            "political": NewsCategory.POLITICAL,
            "regulatory": NewsCategory.REGULATORY,
            "geopolitical": NewsCategory.GEOPOLITICAL
        }
        
        category = event_to_category.get(event_type, NewsCategory.MARKET)
        
        news_event = self.news_generator.generate_news(category, market_state, event_context)
        self.add_news(news_event)
        
        return news_event
    
    def add_news(self, news_event: NewsEvent):
        """Add news event to system"""
        self.news_history.append(news_event)
        self.active_news.append(news_event)
        
        # Maintain history size
        if len(self.news_history) > self.max_news_history:
            self.news_history.pop(0)
    
    def update(self):
        """Update active news based on impact duration"""
        self.current_step += 1
        
        # Remove expired news
        self.active_news = [
            news for news in self.active_news
            if (self.current_step - news.timestamp.timestamp() / 60) < news.impact_duration
        ]
    
    def get_active_news(self, max_count: int = 10) -> List[NewsEvent]:
        """Get currently active news events"""
        return self.active_news[:max_count]
    
    def get_news_sentiment(self, window: int = 10) -> Tuple[float, int]:
        """
        Get aggregate sentiment of recent news.
        Returns (average_sentiment_score, news_count)
        """
        recent_news = self.news_history[-window:] if len(self.news_history) >= window else self.news_history
        
        if not recent_news:
            return 0.0, 0
        
        avg_sentiment = np.mean([news.sentiment_score for news in recent_news])
        return avg_sentiment, len(recent_news)
    
    def get_news_by_category(self, category: NewsCategory, window: int = 20) -> List[NewsEvent]:
        """Get news events by category"""
        recent_news = self.news_history[-window:] if len(self.news_history) >= window else self.news_history
        return [news for news in recent_news if news.category == category]
    
    def calculate_news_impact(self) -> Dict[str, float]:
        """
        Calculate aggregate impact of active news on market.
        Returns dictionary of impact metrics.
        """
        if not self.active_news:
            return {
                'total_price_impact': 0.0,
                'total_volume_impact': 0.0,
                'sentiment_score': 0.0,
                'news_count': 0
            }
        
        total_price_impact = sum(news.price_impact_estimate for news in self.active_news)
        total_volume_impact = sum(news.volume_impact_estimate for news in self.active_news)
        avg_sentiment = np.mean([news.sentiment_score for news in self.active_news])
        
        return {
            'total_price_impact': total_price_impact,
            'total_volume_impact': total_volume_impact,
            'sentiment_score': avg_sentiment,
            'news_count': len(self.active_news)
        }
    
    def reset(self):
        """Reset news system for new episode"""
        self.news_history.clear()
        self.active_news.clear()
        self.current_step = 0


if __name__ == "__main__":
    # Test news system
    news_system = NewsSystem()
    
    # Generate some news
    market_state = {
        'current_price': 100.0,
        'volatility': 0.02,
        'trend': 'up'
    }
    
    for category in NewsCategory:
        news = news_system.news_generator.generate_news(category, market_state)
        news_system.add_news(news)
        print(f"\n{category.value.upper()}:")
        print(f"  Headline: {news.headline}")
        print(f"  Sentiment: {news.sentiment.value} ({news.sentiment_score:.2f})")
        print(f"  Price Impact: {news.price_impact_estimate:.2%}")
    
    # Get aggregate impact
    impact = news_system.calculate_news_impact()
    print(f"\nAggregate Impact:")
    print(f"  Total Price Impact: {impact['total_price_impact']:.2%}")
    print(f"  Total Volume Impact: {impact['total_volume_impact']:.2%}")
    print(f"  Sentiment Score: {impact['sentiment_score']:.2f}")
    print(f"  News Count: {impact['news_count']}")
