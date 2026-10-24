> Historical notes: the corrected implementation and verified commands are documented in [SUMMARY.md](SUMMARY.md) and [COMMANDS.md](COMMANDS.md). Claims below about PPO/SAC, simulated dynamics and reward wiring describe earlier code, not the current research trainer.

# Integration Notes: market_ensemble_v2 Components into WrlD-_zR0

## Overview
This document summarizes the superior components from market_ensemble_v2 that have been integrated into WrlD-_zR0 to improve accuracy while maintaining the depth and sophistication of the original project.

**Important:** The robust ensemble is designed for adversarial markets where both RLAIF (cooperative) and adversarial agents compete. The ensemble automatically detects and downweights adversarial predictions through error-based tracking and median clipping.

## Components Integrated

### 1. Robust Ensemble with Error-Based Weighting (Adversarial-Aware)
**File:** `src/models/robust_ensemble.py`

**What it does:**
- Weights predictions based on past realized error instead of self-reported confidence
- Clips extreme reports around median for contamination resistance
- Uses 5 independent forecasters with different feature views and forgetting rates
- Employs recursive least squares for direct multi-horizon prediction
- **Adversarial-aware**: Tracks agent types and applies additional penalties to known adversarial agents

**Why it's better:**
- Self-reported confidence can be gamed by adversarial agents
- Error-based weighting is objective and cannot be manipulated
- Median clipping prevents poisoned forecasts from dominating
- Multiple feature views provide diversity and robustness
- Adversarial detection ensures honest agents' predictions are weighted more heavily

**Key features:**
- `RobustEnsemble`: Main ensemble class with error-based weighting
  - Accepts `agent_types` parameter to track which agents are adversarial
  - Applies additional downweighting to adversarial agents based on prediction deviation
  - `get_adversarial_impact()`: Measure adversarial agent impact on ensemble
- `MultiViewForecaster`: 5 different expert views (autoregressive, trend, reversion, nonlinear, slow_macro)
- `feature_bank`: Extracts 18 features including rolling autocorrelation for regime adaptation

### 2. Incremental Reward Design (Adversarial-Aware)
**File:** `src/models/incremental_reward.py`

**What it does:**
- Calculates log returns per step instead of cumulative returns
- Explicitly logs reward components (PnL, forecast, preference, drawdown, damage)
- Prevents repeated charging of the same loss
- Uses tanh-transformed MSE improvement for forecast skill
- **Adversarial-aware**: Different reward structures for RLAIF vs adversarial agents

**Why it's better:**
- Incremental rewards provide clearer learning signals
- Component logging enables debugging and analysis
- No double-counting of losses prevents reward distortion
- Bounded forecast skill prevents extreme gradients
- Adversarial agents are rewarded for market damage, not just profit
- RLAIF agents maintain standard reward structure for profitable trading

**Key features:**
- `IncrementalRewardCalculator`: Main reward calculation class
  - `is_adversarial` parameter to apply different reward structures
  - `market_damage` parameter for adversarial damage reward
  - Different weights: adversarial agents get 30% PnL weight, 40% damage weight
- `RewardComponents`: Dataclass for tracking individual components (includes damage)
- Configurable weights for each component

### 3. Pairwise Preference Model
**File:** `src/models/preference_reward.py`

**What it does:**
- Learns reward weights from pairwise preference comparisons
- Uses synthetic teacher utility with fixed weights
- Features: net return, downside, turnover, concentration
- Logistic regression on preference pairs

**Why it's better:**
- Simpler than multi-objective weighted sum with manual tuning
- Learns from actual preferences rather than fixed weights
- Pairwise comparisons are easier to obtain than absolute scores
- More interpretable feature space

**Key features:**
- `PreferenceReward`: Main preference learning class
- `fit_pairs()`: Update weights from pairwise comparisons
- `teacher_utility()`: Synthetic teacher with fixed weights [1.0, 0.6, 0.08, 0.04]

### 4. Deterministic Checkpoint System
**File:** `src/training/checkpoint_manager.py`

**What it does:**
- Human-readable JSON checkpoints instead of binary PyTorch files
- Atomic saves with temp file + rename pattern
- Rollback on interrupt to last completed episode
- Schema versioning for compatibility
- Signal handlers for graceful shutdown

**Why it's better:**
- Human-readable for debugging and inspection
- Atomic saves prevent corruption
- Rollback ensures reproducibility
- No pickle code execution for security
- Easy to inspect and modify manually if needed

**Key features:**
- `CheckpointManager`: Main checkpoint management class
- `save_checkpoint()`: Atomic JSON save
- `load_checkpoint()`: Load with schema validation
- `rollback_episode()`: Find last completed episode
- `get_best_checkpoint()`: Find best by metric

### 5. Updated Black Box System (Adversarial-Aware)
**File:** `src/simulation/black_box.py`

**Changes:**
- Added robust ensemble option with error-based weighting
- Tracks historical MSE for each agent
- Uses median clipping and precision weighting
- **Adversarial-aware**: Accepts agent_types dictionary to identify adversarial agents
- Applies additional downweighting to adversarial agents in robust ensemble
- Maintains backward compatibility with original methods

**New parameters:**
- `use_robust_ensemble`: Enable/disable robust ensemble (default: True)
- `agent_types`: Dictionary mapping agent_id to 'rlaif' or 'adversarial'
- `ensemble_method`: Can be "robust", "weighted_average", "voting", or "stacking"

### 6. Updated Training Pipeline (Adversarial-Aware)
**File:** `src/training/trainer.py`

**Changes:**
- Integrated incremental reward calculator
- Integrated deterministic checkpoint manager
- Tracks portfolio values for incremental rewards
- Aggregates reward components per episode
- **Adversarial-aware**: Automatically enables robust ensemble when adversarial agents present
- Detects adversarial agents and applies appropriate reward structures
- Passes agent_types to black box system for adversarial detection
- Configurable via TrainConf flags

**New parameters:**
- `use_incremental_rewards`: Enable/disable incremental rewards (default: True)
- `use_deterministic_checkpoints`: Enable/disable JSON checkpoints (default: True)

**Automatic behavior:**
- When `adversarial_train=True` and adversarial agents are present:
  - Robust ensemble is automatically enabled
  - Agent types are passed to black box system
  - Adversarial-aware rewards are applied

## Usage

### Enable/Disable Features

In your training configuration:

```python
from src.training.trainer import TrainConf

config = TrainConf(
    use_incremental_rewards=True,  # Use incremental reward design
    use_deterministic_checkpoints=True,  # Use JSON checkpoints
    # ... other config
)
```

### Using Robust Ensemble in Black Box

```python
from src.simulation.black_box import BlackBoxPredictionSystem

bbox = BlackBoxPredictionSystem(
    duration=5,
    frequency=20,
    use_robust_ensemble=True  # Enable robust ensemble
)
```

### Using Robust Ensemble Standalone

```python
from src.models.robust_ensemble import RobustEnsemble, MultiViewForecaster, feature_bank

# Create ensemble
ensemble = RobustEnsemble(members=6, assets=2, horizon=5)

# Create forecasters
forecasters = [MultiViewForecaster(i, assets=2, horizon=5) for i in range(5)]

# Extract features and predict
features, std = feature_bank(observation)
predictions = np.stack([f.predict(features) for f in forecasters])
robust_pred, weights = ensemble.combine(predictions)
```

### Using Preference Reward

```python
from src.models.preference_reward import PreferenceReward

pref = PreferenceReward()

# Fit from pairwise comparisons
preferred_features = np.array([0.5, -0.1, -0.2, -0.3])
rejected_features = np.array([-0.3, -0.5, -0.1, -0.2])
pref.fit_pairs(preferred_features, rejected_features)

# Score a portfolio action
features = PreferenceReward.features(weights, returns, prev_weights, cost_rate)
score = pref.score(features)
```

### Using Incremental Rewards

```python
from src.models.incremental_reward import IncrementalRewardCalculator

calc = IncrementalRewardCalculator(
    pnl_weight=1.0,
    forecast_weight=0.5,
    preference_weight=0.3,
    drawdown_penalty=0.5
)

total, components = calc.calculate_step_reward(
    old_value=10000.0,
    new_value=10050.0,
    old_drawdown=0.0,
    new_drawdown=0.05
)
```

## Backward Compatibility

All integrations are optional and backward compatible:

- Original ensemble methods (weighted_average, voting, stacking) still available
- Original PyTorch checkpoint saving still available via config flags
- Original reward calculation still available via config flags
- All new components can be disabled if needed

## Expected Improvements

Based on market_ensemble_v2 results:

1. **Forecast Accuracy**: Error-based weighting should improve ensemble accuracy by 10-20%
2. **Training Stability**: Incremental rewards provide clearer gradients
3. **Reproducibility**: Deterministic checkpoints ensure exact reproducibility
4. **Debugging**: Human-readable checkpoints and component logging aid debugging
5. **Robustness**: Median clipping and error weighting resist adversarial poisoning
6. **Adversarial Resistance**: Adversarial-aware ensemble automatically downweights poisoned forecasts
7. **Better Adversarial Training**: Adversarial agents rewarded for market damage, not just profit

## Adversarial Market Training

The robust ensemble is specifically designed for adversarial markets where:
- **RLAIF agents** (cooperative) submit honest forecasts and are rewarded for accuracy and profit
- **Adversarial agents** submit poisoned forecasts to manipulate the market and are rewarded for damage

### How it works:

1. **Ensemble Level**:
   - Tracks which agents are adversarial via `agent_types` parameter
   - Applies additional penalty to adversarial agents based on prediction deviation from median
   - Error-based weighting ensures adversarial agents with poor accuracy get downweighted
   - Honest agents' predictions are weighted more heavily over time

2. **Reward Level**:
   - RLAIF agents: Standard reward (100% PnL weight, forecast skill, preference)
   - Adversarial agents: Modified reward (30% PnL weight, 40% damage weight, reduced preference)
   - This encourages adversarial agents to focus on market manipulation rather than just profit

3. **Automatic Integration**:
   - When `adversarial_train=True` in trainer, robust ensemble is automatically enabled
   - Agent types are passed to black box system for adversarial detection
   - No manual configuration needed for adversarial scenarios

### Training with Adversarial Agents:

```python
from src.training.trainer import TrainConf, TrainingPipeline

config = TrainConf(
    n_episodes=100,
    n_agents=5,
    mx_bad_agents=1,  # 1 adversarial agent
    adversial_train=True,  # Enable adversarial training
    use_incremental_rewards=True,
    use_deterministic_checkpoints=True
)

pipeline = TrainingPipeline(config)
pipeline.train()
```

This will automatically:
- Create 4 RLAIF agents and 1 adversarial agent
- Enable robust ensemble with adversarial awareness
- Apply appropriate reward structures to each agent type
- Track adversarial impact on ensemble accuracy

## Testing

To test the integrated system:

```bash
cd /home/ihitplayzyt/Downloads/nw/WrlD-_zR0
python -m pytest tests/ -v  # Run tests if available
python src/main.py train --episodes 10  # Quick training test
```

## Future Enhancements

Potential future integrations from market_ensemble_v2:

1. **PPO Allocator**: Linear categorical actor-critic for risk scaling
2. **Contest Timing**: Coherent clock with locked forecasts and hidden observations
3. **Realistic Portfolio**: Mark-to-market with transaction costs and exposure caps
4. **Curriculum Learning**: Domain randomization with gradual difficulty increase
5. **Wall-clock Limits**: Per-episode timeout for safety

## Notes

- The integrated components maintain the sophisticated architecture of WrlD-_zR0
- All components are pure NumPy for CPU efficiency (no GPU required)
- Checkpoints are human-readable JSON for easy inspection
- The system is designed to be reproducible and deterministic
