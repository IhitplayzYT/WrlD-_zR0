from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from collections import deque
import random

from environment.market_env import MarketEnvironment
from agents import BaseAgent, RLAIFAgent, AdversarialAgent, AgentType
from src.utils.logging import get_logger, LogCategory
from src.tracking.experiment import ExperimentTracker

logger = get_logger()


@dataclass
class CurriculumLevel:
    """
    Curriculum learning level configuration.
    Defines complexity parameters for training progression.
    """
    level: int
    max_complexity: float  # 0.0 to 1.0
    event_frequency: float  # 0.0 to 1.0
    market_volatility: float  # 0.0 to 1.0
    agent_diversity: int  # Number of different agent types
    black_box_frequency: int  # Steps between black box periods
    reward_scaling: float  # Multiplier for rewards
    description: str = ""


class CurriculumManager:
    """
    Manages curriculum learning progression.
    Gradually increases complexity as agents improve.
    """
    
    def __init__(self, max_levels: int = 4):
        self.max_levels = max_levels
        self.current_level = 0
        self.level_history: List[int] = []
        self.performance_history: List[float] = []
        
        # Define curriculum levels
        self.levels = [
            CurriculumLevel(
                level=0,
                max_complexity=0.3,
                event_frequency=0.3,
                market_volatility=0.2,
                agent_diversity=1,
                black_box_frequency=50,
                reward_scaling=1.0,
                description="Basic market with minimal events"
            ),
            CurriculumLevel(
                level=1,
                max_complexity=0.5,
                event_frequency=0.5,
                market_volatility=0.4,
                agent_diversity=2,
                black_box_frequency=30,
                reward_scaling=1.2,
                description="Moderate events and volatility"
            ),
            CurriculumLevel(
                level=2,
                max_complexity=0.7,
                event_frequency=0.7,
                market_volatility=0.6,
                agent_diversity=3,
                black_box_frequency=20,
                reward_scaling=1.5,
                description="Complex events with adversarial agents"
            ),
            CurriculumLevel(
                level=3,
                max_complexity=1.0,
                event_frequency=1.0,
                market_volatility=0.8,
                agent_diversity=4,
                black_box_frequency=15,
                reward_scaling=2.0,
                description="Full complexity with all features"
            )
        ]
    
    def get_current_level(self) -> CurriculumLevel:
        """Get current curriculum level configuration"""
        return self.levels[self.current_level]
    
    def should_progress(self, performance: float, window: int = 5) -> bool:
        """
        Check if agents should progress to next curriculum level.
        Based on performance over recent episodes.
        """
        if self.current_level >= self.max_levels - 1:
            return False
        
        self.performance_history.append(performance)
        if len(self.performance_history) > window:
            self.performance_history.pop(0)
        
        # Need consistent good performance to progress
        if len(self.performance_history) >= window:
            avg_performance = np.mean(self.performance_history)
            threshold = 0.5 + (self.current_level * 0.15)  # Increasing threshold
            return avg_performance > threshold
        
        return False
    
    def progress(self):
        """Progress to next curriculum level"""
        if self.current_level < self.max_levels - 1:
            self.current_level += 1
            self.level_history.append(self.current_level)
            self.performance_history.clear()
            logger.info(LogCategory.TRAINING, f"Progressed to curriculum level {self.current_level}: {self.levels[self.current_level].description}")
    
    def reset(self):
        """Reset curriculum to level 0"""
        self.current_level = 0
        self.level_history.clear()
        self.performance_history.clear()


@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    n_episodes: int = 1000
    episode_length: int = 200
    n_agents: int = 5
    max_bad_agents: int = 1
    initial_cash: float = 1000.0
    save_frequency: int = 100
    eval_frequency: int = 50
    log_frequency: int = 10
    use_curriculum: bool = True
    use_experience_replay: bool = True
    replay_buffer_size: int = 10000
    prioritized_replay: bool = True
    adversarial_training: bool = True
    self_play: bool = False
    transfer_learning: bool = False
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


class ExperienceReplay:
    """
    Experience replay buffer for sample-efficient learning.
    Supports prioritized replay for rare events.
    """
    
    def __init__(self, capacity: int, prioritized: bool = True):
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity) if prioritized else None
    
    def add(
        self,
        observation: Dict,
        action: np.ndarray,
        reward: float,
        next_observation: Dict,
        done: bool,
        priority: Optional[float] = None
    ):
        """Add experience to buffer"""
        self.buffer.append((observation, action, reward, next_observation, done))
        
        if self.prioritized:
            if priority is None:
                # Default priority based on reward magnitude
                priority = abs(reward)
            self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch of experiences"""
        if self.prioritized and len(self.buffer) > 0:
            # Prioritized sampling based on priorities
            priorities = np.array(self.priorities)
            probabilities = priorities / (priorities.sum() + 1e-8)
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            return [self.buffer[i] for i in indices]
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)))
            return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class TrainingPipeline:
    """
    Advanced training pipeline with curriculum learning and multi-agent coordination.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Curriculum manager
        self.curriculum_manager = CurriculumManager() if config.use_curriculum else None
        
        # Experience replay
        self.replay_buffer = ExperienceReplay(
            config.replay_buffer_size,
            config.prioritized_replay
        ) if config.use_experience_replay else None
        
        # Training state
        self.current_episode = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.agent_performances: Dict[str, List[float]] = {}
        
        # Environment (will be created per curriculum level)
        self.env = None
        
        # Logging
        self.training_log = []
        self.start_time = datetime.now()
        
        # Experiment tracking
        self.experiment_tracker = ExperimentTracker(
            log_dir=config.log_dir,
            experiment_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.experiment_tracker.log_hyperparameters(config.__dict__)
    
    def create_environment(self, curriculum_level: Optional[CurriculumLevel] = None) -> MarketEnvironment:
        """Create environment with curriculum level configuration"""
        if curriculum_level is None and self.curriculum_manager:
            curriculum_level = self.curriculum_manager.get_current_level()
        
        # Generate price data with appropriate volatility
        price_data = self._generate_price_data(curriculum_level)
        
        # Determine agent types based on diversity
        if curriculum_level:
            n_types = curriculum_level.agent_diversity
            agent_types = ["rlaif"] * (self.config.n_agents - 1)
            if n_types >= 2 and self.config.adversarial_training:
                agent_types[-1] = "adversarial"
            if n_types >= 3:
                agent_types[-2] = "rlaif"  # Different RLAIF config
        else:
            agent_types = ["rlaif"] * self.config.n_agents
        
        # Create environment
        env = MarketEnvironment(
            n_agents=self.config.n_agents,
            max_bad_agents=self.config.max_bad_agents,
            initial_cash=self.config.initial_cash,
            episode_length=self.config.episode_length,
            price_data=price_data,
            black_box_frequency=curriculum_level.black_box_frequency if curriculum_level else 20,
            black_box_duration=5,
            use_enhanced_observation=True,
            agent_types=agent_types
        )
        
        return env
    
    def _generate_price_data(self, curriculum_level: Optional[CurriculumLevel]) -> List[float]:
        """Generate price data with appropriate volatility"""
        volatility = 0.01 if curriculum_level is None else curriculum_level.market_volatility * 0.03
        n_steps = self.config.episode_length * 10
        
        prices = [100.0]
        for _ in range(n_steps - 1):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))
        
        return prices
    
    def train_episode(self, env: MarketEnvironment) -> Dict[str, float]:
        """Train for one episode"""
        observations, infos = env.reset()
        episode_reward = 0.0
        episode_length = 0
        agent_rewards = {agent_id: 0.0 for agent_id in env.agents}
        
        done = False
        while not done:
            # Get current agent
            agent = env.agent_selection
            
            # Get agent object
            agent_obj = env.agent_objects[agent]
            
            # Agent decides action
            action = agent_obj.decide_action(observations[agent], env.action_space)
            
            # Store observation for replay
            old_observation = observations[agent].copy()
            
            # Step environment
            next_observations, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update rewards
            episode_reward += reward
            agent_rewards[agent] += reward
            episode_length += 1
            
            # Add to experience replay
            if self.replay_buffer:
                self.replay_buffer.add(
                    old_observation,
                    action,
                    reward,
                    next_observations[agent],
                    done,
                    priority=abs(reward) if self.config.prioritized_replay else None
                )
            
            observations = next_observations
        
        # Update agent performances
        for agent_id, reward in agent_rewards.items():
            if agent_id not in self.agent_performances:
                self.agent_performances[agent_id] = []
            self.agent_performances[agent_id].append(reward)
        
        # Apply reward scaling from curriculum
        if self.curriculum_manager:
            curriculum_level = self.curriculum_manager.get_current_level()
            episode_reward *= curriculum_level.reward_scaling
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'agent_rewards': agent_rewards,
            'average_agent_reward': np.mean(list(agent_rewards.values()))
        }
    
    def evaluate(self, env: MarketEnvironment, n_episodes: int = 5) -> Dict[str, float]:
        """Evaluate current agent performance"""
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(n_episodes):
            observations, infos = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            done = False
            while not done:
                agent = env.agent_selection
                agent_obj = env.agent_objects[agent]
                
                # Use deterministic policy for evaluation
                action = agent_obj.decide_action(observations[agent], env.action_space)
                
                observations, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'max_reward': np.max(eval_rewards),
            'min_reward': np.min(eval_rewards)
        }
    
    def train(self):
        """Main training loop"""
        logger.info(LogCategory.TRAINING, f"Starting training for {self.config.n_episodes} episodes", data={'n_episodes': self.config.n_episodes, 'curriculum_learning': self.config.use_curriculum, 'experience_replay': self.config.use_experience_replay})
        
        # Create initial environment
        self.env = self.create_environment()
        
        for episode in range(self.config.n_episodes):
            self.current_episode = episode
            
            # Train episode
            episode_metrics = self.train_episode(self.env)
            
            # Record metrics
            self.episode_rewards.append(episode_metrics['episode_reward'])
            self.episode_lengths.append(episode_metrics['episode_length'])
            
            # Log progress
            if episode % self.config.log_frequency == 0:
                self._log_progress(episode, episode_metrics)
            
            # Evaluation
            if episode % self.config.eval_frequency == 0 and episode > 0:
                eval_metrics = self.evaluate(self.env, n_episodes=3)
                self._log_evaluation(episode, eval_metrics)
                
                # Check curriculum progression
                if self.curriculum_manager:
                    avg_performance = eval_metrics['mean_reward']
                    if self.curriculum_manager.should_progress(avg_performance):
                        self.curriculum_manager.progress()
                        # Recreate environment with new curriculum level
                        self.env = self.create_environment()
            
            # Save checkpoint
            if episode % self.config.save_frequency == 0 and episode > 0:
                self._save_checkpoint(episode)
        
        # Final save
        self._save_checkpoint(self.config.n_episodes)
        self._save_training_log()
        
        logger.info(LogCategory.TRAINING, "Training completed")
        self._print_summary()
    
    def _log_progress(self, episode: int, metrics: Dict[str, float]):
        """Log training progress"""
        avg_reward = np.mean(self.episode_rewards[-self.config.log_frequency:])
        avg_length = np.mean(self.episode_lengths[-self.config.log_frequency:])
        
        log_entry = {
            'episode': episode,
            'episode_reward': metrics['episode_reward'],
            'episode_length': metrics['episode_length'],
            'average_reward': avg_reward,
            'average_length': avg_length,
            'curriculum_level': self.curriculum_manager.current_level if self.curriculum_manager else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_log.append(log_entry)
        
        logger.info(LogCategory.TRAINING, f"Episode {episode} progress", data={'episode': episode, 'reward': metrics['episode_reward'], 'avg_reward': avg_reward, 'length': metrics['episode_length'], 'curriculum_level': log_entry['curriculum_level']})
        
        # Log metrics to experiment tracker
        self.experiment_tracker.log_metrics({
            'episode_reward': metrics['episode_reward'],
            'average_reward': avg_reward,
            'episode_length': metrics['episode_length'],
            'curriculum_level': log_entry['curriculum_level']
        }, step=episode)
    
    def _log_evaluation(self, episode: int, metrics: Dict[str, float]):
        """Log evaluation results"""
        logger.info(LogCategory.EVALUATION, f"Evaluation at episode {episode}", data={'episode': episode, 'mean_reward': metrics['mean_reward'], 'std_reward': metrics['std_reward'], 'max_reward': metrics['max_reward']})
        
        # Log evaluation metrics to experiment tracker
        self.experiment_tracker.log_metrics({
            f'eval_mean_reward': metrics['mean_reward'],
            f'eval_std_reward': metrics['std_reward'],
            f'eval_max_reward': metrics['max_reward'],
            f'eval_min_reward': metrics['min_reward']
        }, step=episode)
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_ep{episode}.pt"
        
        checkpoint = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'agent_performances': self.agent_performances,
            'curriculum_level': self.curriculum_manager.current_level if self.curriculum_manager else 0,
            'config': self.config.__dict__
        }
        
        # Save agent models
        if self.env:
            agent_models = {}
            for agent_id, agent in self.env.agent_objects.items():
                if hasattr(agent, 'save_model'):
                    model_path = Path(self.config.checkpoint_dir) / f"{agent_id}_ep{episode}.pt"
                    agent.save_model(str(model_path))
                    agent_models[agent_id] = str(model_path)
            checkpoint['agent_models'] = agent_models
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(LogCategory.TRAINING, f"Saved checkpoint", data={'episode': episode, 'checkpoint_path': str(checkpoint_path)})
    
    def _save_training_log(self):
        """Save training log to JSON"""
        log_path = Path(self.config.log_dir) / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        logger.info(LogCategory.TRAINING, f"Saved training log", data={'log_path': str(log_path)})
    
    def _print_summary(self):
        """Print training summary"""
        summary_data = {
            'total_episodes': self.current_episode + 1,
            'average_reward': float(np.mean(self.episode_rewards)),
            'max_reward': float(np.max(self.episode_rewards)),
            'min_reward': float(np.min(self.episode_rewards)),
            'average_episode_length': float(np.mean(self.episode_lengths)),
            'final_curriculum_level': self.curriculum_manager.current_level if self.curriculum_manager else 0,
            'level_progression': self.curriculum_manager.level_history if self.curriculum_manager else [],
            'training_duration': str(datetime.now() - self.start_time)
        }
        logger.info(LogCategory.TRAINING, "Training Summary", data=summary_data)
        
        # Log final summary to experiment tracker
        self.experiment_tracker.log_summary(summary_data)
        self.experiment_tracker.finish()


if __name__ == "__main__":
    # Test training pipeline
    config = TrainingConfig(
        n_episodes=100,
        episode_length=200,
        n_agents=3,
        use_curriculum=True,
        use_experience_replay=True,
        save_frequency=50,
        eval_frequency=25,
        log_frequency=10
    )
    
    pipeline = TrainingPipeline(config)
    pipeline.train()
