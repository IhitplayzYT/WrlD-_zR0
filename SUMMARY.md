# RLAIF Trading Simulation - Software Specification

## Overview
This project implements a sophisticated multi-agent Reinforcement Learning from AI Feedback (RLAIF) trading simulation system. The simulation features autonomous trading agents that learn from AI feedback, participate in prediction markets during black box periods, and can transform between "good" and "adversarial" behaviors based on performance.

# Pipeline Architecture

### 1. Data Pipeline
**Location**: `src/data/`

#### Data Generation (`generate.py`)
- **Synthetic Data Generation**: Uses regime-switching Markov models with 5 market regimes (Bull, Bear, Sideways, Crash, Recovery)
- **Real Data Fetching**: `fetch_aapl_data()` function retrieves AAPL stock data from Yahoo Finance (2016-2022)
- **Features**: Volatility clustering, mean reversion, seasonal components, regime-dependent jumps

#### Data Augmentation (`augment.py`)
- **Data Compression**: Sampling-based compression for excess data
- **Data Expansion**: Random self-sampling with volatility transforms for insufficient data
- **Episode Generation**: Creates multiple training episodes from base data
- **Transformations**: Random day swapping, volatility transforms, multiplier effects

### 2. Environment Pipeline
**Location**: `src/environment/`

#### Market Environment (`market_env.py`)
- **Framework**: PettingZoo AEC (Agent-Environment Cycle) for multi-agent RL
- **State Space**: Price history, order book state, agent states, market events
- **Action Space**: [action_type, quantity, price] - 5 action types (hold, market buy/sell, limit buy/sell)
- **Reward System**: Profit-based rewards with penalties for non-participation during black box periods
- **Termination Conditions**: Economy collapse, significant growth, episode length reached

#### Observation Space (`observation_space.py`)
- **Enhanced Observations**: Temporal features, belief states, market state encoding
- **Multi-dimensional**: Combines price history, order book depth, agent states, event vectors

### 3. Agent Pipeline
**Location**: `src/agents/`

#### Base Agent (`base_agent.py`)
- **Trait System**: 15-dimensional trait space (risk tolerance, patience, trend following, etc.)
- **Behavior Modifiers**: Trait-based action modification
- **Memory System**: Limited memory of past events and performance
- **Evolutionary Features**: Trait crossover, mutation, reproduction

#### RLAIF Agent (`rlaif_agent.py`)
- **RL Algorithms**: PPO (Proximal Policy Optimization) and SAC (Soft Actor-Critic)
- **Preference Model**: Multi-objective AI feedback (profitability, risk, market impact, consistency)
- **Curriculum Learning**: 4-level progression system
- **Action Decision**: RL policy with AI feedback integration

#### Adversarial Agent (`adversarial_agent.py`)
- **Malicious Behavior**: Strategies to harm market stability
- **Villain Points**: Earned through successful market manipulation
- **Event Triggering**: Uses villain points to trigger negative market events

### 4. Training Pipeline
**Location**: `src/training/`

#### Training Manager (`trainer.py`)
- **Curriculum Learning**: Progressive complexity increase across 4 levels
- **Experience Replay**: Prioritized replay buffer for sample-efficient learning
- **Multi-Agent Coordination**: Simultaneous training of multiple agents
- **Evaluation**: Periodic evaluation with deterministic policies
- **Checkpointing**: Model saving and loading with training state

#### Training Configuration
- **Episodes**: Configurable number of training episodes
- **Curriculum**: Automatic progression based on performance
- **Replay Buffer**: Configurable size with prioritization
- **Adversarial Training**: Optional adversarial agent inclusion

### 5. Simulation Pipeline
**Location**: `src/simulation/`

#### Black Box System (`black_box.py`)
- **Prediction Contests**: Periodic black box periods for price prediction
- **Ensemble Methods**: Weighted average, voting, stacking ensembles
- **Prediction Market**: Betting mechanism on prediction outcomes
- **Leaderboard**: Reputation-based ranking system
- **King Mechanism**: Best predictor receives reward multiplier

#### World Model (`world_model.py`)
- **Market Dynamics**: Simulates realistic market behavior
- **Agent Interactions**: Multi-agent coordination and competition
- **Event Integration**: Market event effects on price and behavior

### 6. Event Pipeline
**Location**: `src/events/`

#### Event Manager (`event_manager.py`)
- **Event Types**: 12 categories (Economic, Earnings, Central Bank, Political, Natural Disaster, War, Technology, Company, Market Structure, Rumor, Panic)
- **Event Effects**: Price drift, volatility changes, liquidity effects, buy/sell pressure
- **Event Conditions**: Price/volatility thresholds, market cap requirements
- **Event Scheduling**: Probability-based triggering with cooldown periods

### 7. Market Pipeline
**Location**: `src/market/`

#### Order Book (`order_book.py`)
- **Data Structure**: B-tree map for efficient order management
- **Order Types**: Market orders, limit orders
- **Price Discovery**: Mid-price calculation from best bid/ask
- **Market Impact**: Order execution effects on price

#### Correlations (`correlations.py`)
- **Asset Correlations**: Cross-asset correlation modeling
- **Sector Effects**: Event-based sector impact modeling

### 8. User Interface Pipeline
**Location**: `src/gui/`

#### GUI (`gui/main.py`)
- **Framework**: PyQt6 with matplotlib integration
- **Components**: Stock ticker, price graph, news log, agent inspector, prediction panel, villain shop
- **Black Box Visualization**: Active periods shown as greyed regions, historical periods as axis highlights
- **Real-time Updates**: Timer-based simulation updates


## Models

### 1. Reinforcement Learning Models

#### PPO (Proximal Policy Optimization)
- **Implementation**: Stable-Baselines3
- **Hyperparameters**:
  - Learning rate: 3e-4
  - Batch size: 64
  - N steps: 2048
  - N epochs: 10
  - Gamma: 0.99
  - GAE lambda: 0.95
  - Clip range: 0.2

#### SAC (Soft Actor-Critic)
- **Implementation**: Stable-Baselines3
- **Hyperparameters**:
  - Learning rate: 3e-4
  - Buffer size: 100000
  - Batch size: 256
  - Gamma: 0.99

### 2. Preference Model (MORLAIF)
- **Architecture**: Multi-objective evaluation system
- **Objectives**:
  - Profitability (weight: 0.4)
  - Risk assessment (weight: 0.3)
  - Market impact (weight: 0.2)
  - Consistency (weight: 0.1)
- **Output**: Combined score with component breakdown

### 3. Prediction Models
- **Ensemble Methods**:
  - Weighted Average (confidence-based)
  - Voting (median-based)
  - Stacking (reputation-based)
- **Features**: Price level prediction, regime transition, trend direction, volatility level

### 4. Data Processing Models
- **Regime-Switching Model**: 5-state Markov chain for market regimes
- **Volatility Model**: EWMA (Exponentially Weighted Moving Average) for volatility clustering
- **Jump Model**: Regime-dependent jump diffusion process

## Optimizers

### 1. RL Optimizers
- **PPO Optimizer**: Adam optimizer with clipped surrogate objective
- **SAC Optimizer**: Adam optimizer for both actor and critic networks

### 2. Training Optimizers
- **Experience Replay**: Prioritized replay for sample efficiency
- **Curriculum Learning**: Progressive complexity optimization
- **Multi-Agent Learning**: Concurrent agent training with shared environment

### 3. Market Optimizers
- **Order Matching**: Efficient order book matching algorithm
- **Price Discovery**: Bid-ask spread optimization
- **Liquidity Management**: Dynamic liquidity adjustment

### 4. Prediction Optimizers
- **Ensemble Optimization**: Dynamic weight adjustment based on reputation
- **Consensus Building**: Market mechanism for prediction aggregation
- **Accuracy Calibration**: Confidence interval calibration

## Key Features

### 1. Black Box Prediction System
- **Periodic Contests**: Agents predict future prices during hidden periods
- **Reward Mechanism**: Accuracy-based rewards with king multiplier
- **Villain Points**: Earned for accurate predictions, spent on market events
- **Non-Participation Penalty**: Large penalty for not participating

### 2. Agent Transformation
- **Good → Bad**: Agents performing poorly (<50% initial value) become adversarial
- **Irreversible**: No transition from bad to good
- **Trait Modification**: Adversarial traits increase risk tolerance and contrarian bias

### 3. Event System
- **50+ Market Events**: Comprehensive event library with realistic parameters
- **Dynamic Effects**: Price drift, volatility changes, liquidity effects
- **Conditional Triggering**: Events trigger based on market conditions
- **Custom Events**: JSON-based custom event loading

### 4. Data Pipeline
- **Default AAPL Data**: Automatically fetches AAPL data from 2016-2022
- **Fallback Mechanism**: Synthetic data generation if fetch fails
- **Data Augmentation**: Expands/compresses data to match episode requirements
- **Multi-Format Support**: CSV, Excel, PDF, text file ingestion

### 5. Budget Configuration
- **Dynamic Initial Budget**: Set to 2x stock price at time=0
- **Automatic Calculation**: Based on first data point of fetched stock data
- **Configurable**: Override via command-line arguments

## Configuration

### Command-Line Arguments
- `-gui/--gui`: Launch GUI interface
- `-n=/--n_agents=`: Number of agents
- `-amnt=/--initial_amnt=`: Initial budget amount
- `-bb_dur=/--bbox_duration=`: Black box duration
- `-bb_freq=/--bbox_frequency=`: Black box frequency
- `-rng_bbox=/--random_bbox=`: Random black box parameters
- `-stock=/--stock_data=`: Custom stock data files
- `-event=/--event=`: Custom events JSON file

### Default Configuration
- **Agents**: 5 agents (max 1 adversarial)
- **Initial Budget**: 2x stock price at time=0 (dynamic)
- **Episodes**: 100 episodes
- **Episode Length**: 200 days × 10 movements/day
- **Black Box**: Duration 5, Frequency 20
- **Stock Data**: AAPL 2016-2022 (default)

## File Structure

```
WrlD-_zR0/
├── src/
│   ├── agents/          # Agent implementations
│   ├── data/            # Data generation and augmentation
│   ├── environment/     # RL environment
│   ├── events/          # Market event system
│   ├── gui/             # Graphical interface
│   ├── market/          # Market mechanics
│   ├── models/          # ML models (CV, NLP)
│   ├── simulation/      # Simulation components
│   ├── training/        # Training pipeline
│   └── main.py          # Main entry point
├── requirements.txt     # Python dependencies
├── config.toml         # Configuration file
├── checkpoints/         # Model checkpoints
├── logs/               # Training logs
└── venv/              # Virtual environment
```

### Performance Optimization
- **Polars**: High-performance data processing
- **Prioritized Replay**: Sample-efficient learning
- **Curriculum Learning**: Accelerated training progression


### Extensibility
- **Multi-Algorithm Support**: PPO, SAC, and custom algorithms


### Core ML/RL
- Stable-Baselines3 (PPO, SAC)
- Gymnasium/PettingZoo (RL environments)
- Transformers (NLP models)
- PyTorch (Deep learning)

### Data Processing
- Polars (Data manipulation)
- NumPy/SciPy (Numerical computing)
- Pandas (Data analysis)

