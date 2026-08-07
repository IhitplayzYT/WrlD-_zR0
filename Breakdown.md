# WrlD-_zR0 Project Breakdown & Roadmap

## Project Overview
A RLAIF (Reinforcement Learning from AI Feedback) based stock market simulation where multiple AI agents compete to generate profits in a simulated market environment. The system features adversarial dynamics, random market events, black box prediction periods, and realistic market mechanics.

## Current Implementation Status
- ✅ Data ingestion (CSV, Excel, PDF, Text)
- ✅ Synthetic stock data generation with market regimes
- ✅ Data augmentation for episode generation
- ✅ Basic CLI configuration system
- ✅ CV model for image description (Ollama integration)
- ✅ Project structure and dependencies

## High-Level Architecture

### Core Components
1. **Data Layer** (`src/data/`)
   - `generate.py` - Synthetic stock data with realistic market regimes
   - `augment.py` - Data augmentation for episode generation

2. **Ingestion Layer** (`src/ingestors/`)
   - `csv.py` - CSV/Excel data ingestion
   - `pdf.py` - PDF extraction with table parsing
   - `text.py` - Plain text data ingestion

3. **Model Layer** (`src/models/`)
   - `CV.py` - Computer vision for image analysis

4. **Configuration** (`src/`)
   - `main.py` - Entry point
   - `helper.py` - CLI and configuration
   - `consts.py` - Constants

---

## Detailed Roadmap

### Phase 1: Core Market Simulation Infrastructure (Priority: HIGH)

#### 1.1 Order Book Implementation
**Status**: ✅ Completed
**Location**: `src/market/order_book.py`
**Description**: Implement BTreeMap-based order book for realistic price discovery
**Tasks**:
- Create Order class with price, quantity, timestamp, agent_id
- Implement OrderBook class with buy/sell sides
- Use sortedcontainers or implement BTree for efficient matching
- Implement order matching engine
- Add price impact based on order depth
**Dependencies**: None
**Research Reference**: Review order book mechanics in research papers

#### 1.2 Market Environment (Gymnasium/PettingZoo)
**Status**: ✅ Completed
**Location**: `src/environment/market_env.py`
**Description**: Create multi-agent RL environment
**Tasks**:
- Extend gymnasium.Env or pettingzoo API
- Implement observation space (price history, order book, events)
- Implement action space (buy/sell/hold, quantity)
- Implement reward function (profit/loss, penalties)
- Handle episode termination conditions
- Implement black box periods
**Dependencies**: 1.1 (Order Book)
**Research Reference**: `DRL_for_optimal_trading_partial_information.pdf`

#### 1.3 Event System
**Status**: ✅ Completed
**Location**: `src/events/event_manager.py`
**Description**: Implement random and planned market events
**Tasks**:
- Create Event base class
- Implement random good/bad events
- Implement planned events (buybacks, wars, infrastructure)
- Implement event impact on stock prices
- Add event broadcasting/newspaper system
- Implement critical events with higher impact
```
class MarketEvent:
    id: int
    name: str
    category: EventType
    probability: float

    duration: int          # timesteps
    cooldown: int
    priority: int

    conditions: EventCondition
    effects: EventEffects

@dataclass
class EventEffects:
    price_drift: float              # expected directional movement
    volatility_multiplier: float
    liquidity_multiplier: float
    spread_multiplier: float

    buy_pressure: float
    sell_pressure: float

    volume_multiplier: float

    market_confidence: float

    sector_effects: dict[str, float]
    asset_effects: dict[int, float]

    correlation_change: float

    jump_probability: float
    jump_mean: float
    jump_std: float

    mean_reversion_strength: float

    order_cancel_rate: float
    order_arrival_rate: float

class EventType(Enum):
    ECONOMIC
    EARNINGS
    CENTRAL_BANK
    POLITICAL
    NATURAL_DISASTER
    WAR
    TECHNOLOGY
    COMPANY
    MARKET_STRUCTURE
    RUMOR
    PANIC
```
**Dependencies**: 1.2 (Market Environment)
**Research Reference**: Review event modeling in RL literature

#### 1.4 Time Management
**Status**: ✅ Completed
**Location**: `src/simulation/time_manager.py`
**Description**: Implement day/tick timing system
**Tasks**:
- Implement day cycle (1-2 mins per day)
- Implement intra-day movements (10 movements per day)
- Add timestamp-based agent access restrictions
- Implement black box frequency and duration
- Add rare agent blocking/killing events
**Dependencies**: 1.2 (Market Environment)

---

### Phase 2: Agent Implementation (Priority: HIGH)

#### 2.1 Base Agent Architecture
**Status**: ❌ Not Started
**Location**: `src/agents/base_agent.py`
**Description**: Create base agent class with traits
**Tasks**:
- Define Agent base class
- Implement agent traits (risk tolerance, patience, etc.)
- Add portfolio management
- Implement decision-making interface
- Add memory/state tracking
- Implement agent lifecycle (creation, death, transformation)
**Dependencies**: 1.2 (Market Environment)

#### 2.2 RLAIF Agent Implementation
**Status**: ❌ Not Started
**Location**: `src/agents/rlaif_agent.py`
**Description**: Implement profit-seeking RLAIF agents
**Tasks**:
- Extend base agent with RL policy (PPO/SAC from stable-baselines3)
- Implement observation preprocessing
- Add action interpretation (convert to orders)
- Implement reward shaping
- Add model checkpointing/loading
- Implement trait-based behavior modification
**Dependencies**: 2.1 (Base Agent), 1.2 (Market Environment)
**Research Reference**: `Evolution_of_RL_survey.pdf`

#### 2.3 Adversarial Agent Implementation
**Status**: ❌ Not Started
**Location**: `src/agents/adversarial_agent.py`
**Description**: Implement malicious agents that try to crash market
**Tasks**:
- Extend base agent with adversarial objectives
- Implement market manipulation strategies
- Add "Villain Points" system
- Implement bad event triggering mechanism
- Add transformation logic (good -> bad when threshold reached)
- Enforce max bad agent limit
**Dependencies**: 2.1 (Base Agent), 1.3 (Event System)

#### 2.4 NLP/CV Integration for Agents
**Status**: ❌ Not Started
**Location**: `src/agents/multimodal_agent.py`
**Description**: Add document/image analysis capabilities
**Tasks**:
- Integrate financial statement analysis (NLP)
- Add graph/trend interpretation (CV)
- Implement information extraction from documents
- Add news sentiment analysis
- Create feature fusion for multimodal observations
**Dependencies**: 2.2 (RLAIF Agent), existing ingestors/models

---

### Phase 3: Advanced Features (Priority: MEDIUM)

#### 3.1 News & Information System
**Status**: ❌ Not Started
**Location**: `src/info/news_system.py`
**Description**: Implement news generation and distribution
**Tasks**:
- Create news generator based on market events
- Implement newspaper broadcasting for top performers
- Add financial statement distribution
- Implement news sentiment impact on agents
- Create summary model for user commentary
**Dependencies**: 1.3 (Event System), 2.4 (NLP/CV Integration)

#### 3.2 Black Box Prediction System
**Status**: ❌ Not Started
**Location**: `src/simulation/black_box.py`
**Description**: Implement stock price prediction contests
**Tasks**:
- Implement black box period triggering
- Create prediction submission interface
- Implement prediction scoring (closest to actual)
- Award "King" status with multiplier
- Award "Villain Points" for accurate predictors
- Handle non-participation penalties
**Dependencies**: 1.2 (Market Environment), 1.4 (Time Management)

#### 3.3 Options Pricing
**Status**: ❌ Not Started
**Location**: `src/market/options.py`
**Description**: Implement options pricing and trading
**Tasks**:
- Implement options pricing models (Black-Scholes, etc.)
- Create options chain from stock movements
- Add options trading to action space
- Implement options Greeks calculation
- Add options to portfolio valuation
**Dependencies**: 1.1 (Order Book), 1.2 (Market Environment)

#### 3.4 Multi-Stock Support
**Status**: ❌ Not Started
**Location**: `src/market/multi_stock.py`
**Description**: Extend to multiple stocks with correlations
**Tasks**:
- Implement multiple stock instances
- Add stock correlation modeling
- Implement portfolio diversification
- Add interdependent stock relationships
- Extend observation space for multiple stocks
**Dependencies**: 1.2 (Market Environment), existing data pipeline

---

### Phase 4: Training & Evaluation (Priority: MEDIUM)

#### 4.1 Training Pipeline
**Status**: ❌ Not Started
**Location**: `src/training/trainer.py`
**Description**: Implement multi-agent training loop
**Tasks**:
- Create training orchestration
- Implement episode management
- Add cross-episode reward tracking
- Implement streak bonuses
- Add model parameter saving/loading
- Implement curriculum learning
**Dependencies**: Phase 2 (Agent Implementation)

#### 4.2 Evaluation Metrics
**Status**: ❌ Not Started
**Location**: `src/evaluation/metrics.py`
**Description**: Implement comprehensive evaluation
**Tasks**:
- Define success metrics (profit, Sharpe ratio, etc.)
- Implement agent performance tracking
- Add market health metrics
- Create visualization tools
- Implement comparison with baseline strategies
- Add statistical significance testing
**Dependencies**: 4.1 (Training Pipeline)

#### 4.3 Experiment Tracking
**Status**: ❌ Not Started
**Location**: `src/experiments/tracking.py`
**Description**: Integrate TensorBoard/WandB
**Tasks**:
- Set up TensorBoard logging
- Configure WandB integration
- Log agent performance
- Log market metrics
- Log hyperparameters
- Create experiment comparison dashboard
**Dependencies**: 4.1 (Training Pipeline)

---

### Phase 5: Real Market Integration (Priority: LOW)

#### 5.1 Real Data Pipeline
**Status**: ❌ Not Started
**Location**: `src/data/real_market.py`
**Description**: Connect to real market data APIs
**Tasks**:
- Integrate with market data APIs (Yahoo, Alpha Vantage, etc.)
- Implement real-time data streaming
- Add data caching and storage
- Implement data validation
- Add historical data backfilling
**Dependencies**: Phase 1 (Core Infrastructure)

#### 5.2 Paper Trading
**Status**: ❌ Not Started
**Location**: `src/live/paper_trading.py`
**Description**: Implement simulated real-money trading
**Tasks**:
- Create paper trading account
- Implement 1-week learning period
- Add simulated profit tracking
- Implement risk management
- Add performance comparison to buy-and-hold
**Dependencies**: 5.1 (Real Data Pipeline), Phase 4 (Training)

---

### Phase 6: Documentation & Testing (Priority: MEDIUM)

#### 6.1 Unit Tests
**Status**: ❌ Not Started
**Location**: `tests/`
**Description**: Comprehensive unit test suite
**Tasks**:
- Test data ingestion
- Test order book logic
- Test agent behaviors
- Test event system
- Test environment mechanics
- Achieve >80% code coverage
**Dependencies**: All implementation phases

#### 6.2 Integration Tests
**Status**: ❌ Not Started
**Location**: `tests/integration/`
**Description**: End-to-end testing
**Tasks**:
- Test full episode execution
- Test multi-agent interactions
- Test event impact
- Test training pipeline
- Test model loading/saving
**Dependencies**: Phase 4 (Training)

#### 6.3 Documentation
**Status**: ❌ Not Started
**Location**: `docs/`
**Description**: Comprehensive documentation
**Tasks**:
- API documentation
- Architecture diagrams
- Usage examples
- Configuration guide
- Troubleshooting guide
**Dependencies**: All implementation phases

---

## File Structure Plan

```
WrlD-_zR0/
├── README.md
├── Breakdown.md
├── requirements.txt
├── config.toml
├── src/
│   ├── main.py
│   ├── consts.py
│   ├── helper.py
│   ├── data/
│   │   ├── generate.py
│   │   ├── augment.py
│   │   └── real_market.py
│   ├── ingestors/
│   │   ├── csv.py
│   │   ├── pdf.py
│   │   └── text.py
│   ├── models/
│   │   ├── CV.py
│   │   └── nlp.py
│   ├── market/
│   │   ├── order_book.py
│   │   ├── options.py
│   │   └── multi_stock.py
│   ├── environment/
│   │   └── market_env.py
│   ├── events/
│   │   └── event_manager.py
│   ├── simulation/
│   │   ├── time_manager.py
│   │   └── black_box.py
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── rlaif_agent.py
│   │   ├── adversarial_agent.py
│   │   └── multimodal_agent.py
│   ├── info/
│   │   └── news_system.py
│   ├── training/
│   │   └── trainer.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── experiments/
│       └── tracking.py
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
├── research_papers/
│   ├── DRL_for_optimal_trading_partial_information.pdf
│   ├── Evolution_of_RL_survey.pdf
│   └── RL_from_delayed_observe_world_models.pdf
└── logs/
```

---

## Research Papers Integration

### DRL_for_optimal_trading_partial_information.pdf
**Focus**: Deep RL for optimal trading with partial information
**Relevance**: 
- Agent observation design under partial information
- Black box prediction mechanics
- Information asymmetry handling

### Evolution_of_RL_survey.pdf
**Focus**: Survey of RL evolution and techniques
**Relevance**:
- RLAIF methodology
- Multi-agent RL algorithms
- Reward shaping strategies

### RL_from_delayed_observe_world_models.pdf
**Focus**: RL with delayed observations and world models
**Relevance**:
- Handling delayed information (news, events)
- World model for market prediction
- Temporal credit assignment

---

## Implementation Priority Order

1. **Phase 1.1** - Order Book (foundational for all trading)
2. **Phase 1.2** - Market Environment (core RL interface)
3. **Phase 2.1** - Base Agent (agent architecture)
4. **Phase 1.4** - Time Management (simulation timing)
5. **Phase 2.2** - RLAIF Agent (primary agent type)
6. **Phase 1.3** - Event System (market dynamics)
7. **Phase 2.3** - Adversarial Agent (complete agent ecosystem)
8. **Phase 3.2** - Black Box Prediction (key feature)
9. **Phase 4.1** - Training Pipeline (enable learning)
10. **Phase 3.1** - News System (information flow)
11. **Phase 4.2** - Evaluation Metrics (measure progress)
12. **Phase 4.3** - Experiment Tracking (monitor training)
13. **Phase 2.4** - NLP/CV Integration (advanced features)
14. **Phase 3.3** - Options Pricing (advanced trading)
15. **Phase 3.4** - Multi-Stock Support (complexity increase)
16. **Phase 6.1** - Unit Tests (quality assurance)
17. **Phase 6.3** - Documentation (usability)
18. **Phase 5.1** - Real Data Pipeline (real-world application)
19. **Phase 5.2** - Paper Trading (final validation)
20. **Phase 6.2** - Integration Tests (final verification)

---

## Configuration Parameters

Current CLI parameters (from `src/helper.py`):
- `n_agnts`: Number of agents (default: 5)
- `mx_bad`: Max bad agents (default: 20% of agents)
- `mx_episodes`: Max episodes (default: 100)
- `episode_day_len`: Days per episode (default: 200)
- `each_day_movements`: Price movements per day (default: 10)
- `initial_amount`: Starting capital (default: 1000)
- `stock_data`: Optional real stock data files
- `event_probs`: Event probabilities (good, bad, critical)
- `news`: Enable news system (default: False)
- `black_bx_dur`: Black box duration (default: 5)
- `black_bx_freq`: Black box frequency (default: 20)

---

## Key Design Decisions

### Agent Architecture
- Use PettingZoo for multi-agent RL environment
- Stable-Baselines3 for individual agent policies
- Separate observation/action spaces per agent type
- Trait-based behavior modification for diversity

### Market Simulation
- Order book for realistic price discovery
- Event-driven market dynamics
- Time-based agent access restrictions
- Black box periods for prediction contests

### Training Strategy
- Start with single-stock simulation
- Progress to multi-stock after stability
- Use curriculum learning for complex features
- Implement cross-episode rewards for long-term learning

---

## Next Immediate Steps

1. **Implement Order Book** (`src/market/order_book.py`)
   - This is the foundation for all trading mechanics
   - Required before environment can be built

2. **Create Market Environment** (`src/env/market_env.py`)
   - Core RL interface for agents
   - Integrates order book and time management

3. **Build Base Agent架构** (`src/agents/base_agent.py`)
   - Define agent interface and traits
   - Required before specific agent implementations

4. **Implement Time Manager** (`src/simulation/time_manager.py`)
   - Controls simulation timing and black box periods
   - Critical for realistic simulation

---

## Progress Tracking

Use this breakdown to track progress:
- Mark completed items with ✅
- Mark in-progress items with 🔄
- Mark blocked items with ⏸️
- Update status as work progresses

---

## Notes for Future Agents

- This project uses Polars for data processing (not Pandas)
- CV model uses Ollama (llava:latest) - ensure Ollama is running
- Research papers in `research_papers/` contain relevant methodologies
- Current data generation uses realistic market regimes (Bull, Bear, etc.)
- The system is designed for 20 max agents with 1-2 minute days
- Order book implementation should use BTree for efficiency
- All logging should be JSON-based for easy parsing
- Model parameters should be saved for fast recreation
