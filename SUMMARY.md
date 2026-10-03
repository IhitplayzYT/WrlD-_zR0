#### Market Environment (`market_env.py`)
- **Framework**: PettingZoo AEC
- **State Space**: Price history, order book state, agent states, market events
- **Action Space**: [action_type, quantity, price] - 5 action types (hold, market buy/sell, limit buy/sell)
- **Termination Conditions**: Economy collapse, significant growth, episode length reached

#### Observation Space (`observation_space.py`)
- **Enhanced Observations**: Temporal features, belief states, market state encoding
- **Multi-dimensional**: Combines price history, order book depth, agent states, event vectors

#### Base Agent (`base_agent.py`)
- **Trait System**: 15-dimensional trait space (risk tolerance, patience, trend following, etc.)
- **Memory System**: Limited memory of past events and performance
- **Evolutionary Features**: Trait crossover, mutation, reproduction

- **RL Algorithms**: PPO (Proximal Policy Optimization) and SAC (Soft Actor-Critic)
- **Preference Model**: Multi-objective AI feedback (profitability, risk, market impact, consistency)

#### Adversarial Agent (`adversarial_agent.py`)
- **Malicious Behavior**: Strategies to harm market stability
- **Villain Points**: Earned through successful market manipulation

#### Training Manager (`trainer.py`)
- **Curriculum Learning**: Progressive complexity increase across 4 levels
- **Experience Replay**: Prioritized replay buffer for sample-efficient learning

#### Black Box System (`black_box.py`)
- **Prediction Contests**: Periodic black box periods for price prediction
- **Ensemble Methods**: Weighted average, voting, stacking ensembles
- **Leaderboard**: Reputation-based ranking system
- **King Mechanism**: Best predictor receives reward multiplier

#### World Model (`world_model.py`)
- **Market Dynamics**: Simulates realistic market behavior
- **Agent Interactions**: Multi-agent coordination and competition
- **Event Integration**: Market event effects on price and behavior


#### Order Book (`order_book.py`)
- **Order Types**: Market orders, limit orders
- **Market Impact**: Order execution effects on price


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

### 1. Black Box Prediction System
- **Reward Mechanism**: Accuracy-based rewards with king multiplier
- **Villain Points**: Earned for accurate predictions, spent on market events
- **Non-Participation Penalty**: Large penalty for not participating

### 2. Agent Transformation
- **Good → Bad**: Agents performing poorly (<50% initial value) become adversarial
- **Irreversible**: No transition from bad to good
