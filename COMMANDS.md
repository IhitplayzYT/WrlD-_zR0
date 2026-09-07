# RLAIF Trading Simulation - Command Reference


### Run Standard Simulation
```bash
python src/main.py
```

### Load Custom Stock Data
```bash
python src/main.py -stock=AAPL:/path/to/aapl.csv
python src/main.py -stock=GOOGL:/path/to/googl.xlsx
python src/main.py -stock=TSLA:/path/to/tsla.pdf
```


### Load Custom Events from JSON
```bash
python src/main.py -event=/path/to/events.json
# or
python src/main.py --event=/path/to/events.json
```

## Training Commands

### Run Training Pipeline
```bash
python -m src.training.trainer
```

### Run Training with Custom Configuration
```python
from src.training.trainer import TrainingPipeline, TrainingConfig

# Create custom configuration
config = TrainingConfig(
    n_episodes=1000,
    episode_length=200,
    n_agents=5,
    max_bad_agents=1,
    initial_cash=10000,
    checkpoint_dir="./checkpoints",
    log_dir="./logs",
    save_frequency=100,
    eval_frequency=50,
    log_frequency=10,
    use_curriculum=True,
    use_experience_replay=True,
    replay_buffer_size=10000,
    replay_batch_size=32
)

# Create and run training pipeline
pipeline = TrainingPipeline(config)
pipeline.train()
```

### Run Training with Specific Curriculum Level
```python
from src.training.trainer import TrainingPipeline, TrainingConfig, CurriculumLevel

# Define custom curriculum levels
curriculum_levels = [
    CurriculumLevel(level=0, n_agents=3, max_bad=0, volatility=0.01),
    CurriculumLevel(level=1, n_agents=5, max_bad=1, volatility=0.02),
    CurriculumLevel(level=2, n_agents=10, max_bad=2, volatility=0.03),
]

config = TrainingConfig(
    n_episodes=1000,
    curriculum_levels=curriculum_levels,
    use_curriculum=True
)

pipeline = TrainingPipeline(config)
pipeline.train()
```


### Load Saved Model
```python
import torch
from src.training.trainer import TrainingPipeline, TrainingConfig

# Load checkpoint
checkpoint_path = "./checkpoints/checkpoint_ep500.pt"
checkpoint = torch.load(checkpoint_path)

# Restore training state
config = TrainingConfig(**checkpoint['config'])
pipeline = TrainingPipeline(config)

# Restore episode rewards and lengths
pipeline.episode_rewards = checkpoint['episode_rewards']
pipeline.episode_lengths = checkpoint['episode_lengths']
pipeline.agent_performances = checkpoint['agent_performances']

# Restore curriculum level
if pipeline.curriculum_manager:
    pipeline.curriculum_manager.current_level = checkpoint['curriculum_level']

# Load agent models
for agent_id, model_path in checkpoint['agent_models'].items():
    if agent_id in pipeline.env.agent_objects:
        agent = pipeline.env.agent_objects[agent_id]
        if hasattr(agent, 'load_model'):
            agent.load_model(model_path)

# Continue training from checkpoint
pipeline.train()
```

### Load Specific Agent Model
```python
import torch
from src.agents.rlaif_agent import RLAIFAgent

# Load agent model
agent = RLAIFAgent(...)
agent.load_model("./checkpoints/agent_0_ep500.pt")
```

### Manual Model Save
```python
from src.agents.rlaif_agent import RLAIFAgent

agent = RLAIFAgent(...)
# ... train agent ...

# Save model manually
agent.save_model("./custom_model_path.pt")
```

## Integration Test Commands

### Run Integration Tests
```bash
python test_integration.py
```

### Run Specific Test
```python
from test_integration import *

# Test environment creation
test_environment_creation()

# Test environment step
test_environment_step()

# Test black box system
test_black_box_system()

# Test observation space
test_observation_space()

# Test agent traits
test_agent_traits()

# Test RLAIF agent
test_rlaif_agent()

# Test adversarial agent
test_adversarial_agent()

# Test evaluation metrics
test_evaluation_metrics()

# Test training pipeline
test_training_pipeline()
```

## TUI Keybinds

When running with `-tui` flag:
- `q`: Quit
- `space`: Pause/Resume simulation
- `1`: Switch to Market View tab
- `2`: Switch to Model Management tab
- `up/down`: Scroll in focused widget
- `tab`: Switch to next stock in price graph
- `left/right`: Navigate villain shop items
- `enter`: Purchase selected item from shop
- `+/-`: Increase/decrease duration for selected shop item
- `t`: Toggle timing (immediate/delayed) for selected shop item
- `mouse scroll`: Zoom in/out on price graph

## Complete Example Commands

### Full Training Run
```bash
python src/main.py -n=10 -m_bad=2 -l_ep=500 -e_mov=10 -amnt=10000 -news -bb_dur=5 -bb_freq=20
```

### TUI with Custom Configuration
```bash
python src/main.py -tui -n=15 -m_bad=3 -l_ep=1000 -e_mov=20 -amnt=50000
```

### Training with Custom Events
```bash
python src/main.py -event=custom_events.json -n=20 -l_ep=1000
```

## Model Save Features

The training pipeline includes comprehensive save/load functionality:

**Automatic Saving:**
- Checkpoints saved every N episodes (configurable via `save_frequency`)
- Final checkpoint always saved
- Training logs saved to JSON format
- Agent models saved individually if they implement `save_model` method

**Checkpoint Contents:**
- Episode number
- Episode rewards history
- Episode lengths history
- Agent performances
- Curriculum level
- Configuration parameters
- Agent model paths

**Save Locations:**
- Checkpoints: `./checkpoints/checkpoint_ep{N}.pt`
- Agent models: `./checkpoints/{agent_id}_ep{N}.pt`
- Training logs: `./logs/training_log.json`

**Model Persistence:**
- PyTorch models saved using `torch.save()`
- Supports resuming training from checkpoints
- Supports loading individual agent models
- Configuration preserved for reproducibility
