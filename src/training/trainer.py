from src.utils.logging import JSONLogger, get_logger, LogCategory
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any 
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from collections import deque
from tqdm import tqdm
import signal
import sys
from src.environment.market_env import MarketEnvironment
from src.tracking.experiment import ExperimentTracker
from src.models.incremental_reward import IncrementalRewardCalculator, RewardComponents
from src.training.checkpoint_manager import CheckpointManager

EPS = 1e-8

@dataclass
class TrainingLevel:
    lvl: int
    complexity: float  # 0.0 to 1.0
    event_freq: float 
    volatility: float 
    n_agent: int
    bbox_freq: int 
    rwrd_scale: float 
    desc: str = ""
    thresh: float = 0.5

class TrainingScheduler:
    def __init__(self, mx_lvl: int = 4, logger: Optional[JSONLogger] = None):
        self.mx_lvl = mx_lvl
        self.curr_lvl_idx = 0
        self.lvl_hist: List[int] = []
        self.performance: List[float] = []
        self.logger = logger or get_logger()
        self.lvls = [
            TrainingLevel(0, 0.3, 0.3, 0.2, 1, 50, 1.0, "Minimal events", 0.5),
            TrainingLevel(1, 0.5, 0.5, 0.4, 2, 30, 1.15, "Moderate events and volatility", 0.65),
            TrainingLevel(2, 0.7, 0.7, 0.6, 3, 20, 1.2, "Complex events with adversarial agents", 0.8),
            TrainingLevel(3, 1.0, 1.0, 0.8, 4, 15, 1.4, "Full complexity with all features", 0.95),
        ]
    
    def get_curr_lvl(self) -> TrainingLevel:
        return self.lvls[self.curr_lvl_idx]
    
    def should_progress(self, perf: float, window: int = 5) -> bool:
        if self.curr_lvl_idx >= self.mx_lvl - 1:
            return False
        self.performance.append(perf)
        l = len(self.performance)
        if l > window:
            self.performance.pop(0)
            l -= 1
        if l >= window:
            return float(np.mean(self.performance)) > self.lvls[self.curr_lvl_idx].thresh
        return False
    
    def progress(self):
        if self.curr_lvl_idx < self.mx_lvl - 1:
            self.curr_lvl_idx += 1
            self.lvl_hist.append(self.curr_lvl_idx)
            self.performance.clear()
            self.logger.info(LogCategory.TRAINING, f"Progressed to curriculum level {self.curr_lvl_idx}: {self.lvls[self.curr_lvl_idx].desc}")
    
    def reset(self):
        self.curr_lvl_idx = 0
        self.lvl_hist.clear()
        self.performance.clear()

@dataclass
class TrainConf:
    n_episodes: int = 100
    episode_len: int = 200
    n_agents: int = 5
    mx_bad_agents: int = 1
    init_cash: float = 10000.0
    save_freq: int = 10
    eval_freq: int = 10
    log_freq: int = 10
    terminal_log_freq: int = 1
    replay_buff_sz: int = 1000
    use_trainer: bool = True
    use_experience_replay: bool = True
    prioritize_replay: bool = True
    adversial_train: bool = True
    self_play: bool = False
    transfer_learning: bool = False
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    sequential_levels: bool = False
    episodes_per_level: int = 25
    bankruptcy_threshold: float = 0.5
    use_incremental_rewards: bool = True
    use_deterministic_checkpoints: bool = True


class ExperienceReplay:
    def __init__(self, cap: int, prioritized: bool = True):
        self.cap = cap 
        self.is_priority = prioritized
        self.buff: deque = deque(maxlen=cap)
        self.priorities: Optional[deque] = deque(maxlen=cap) if prioritized else None
    
    def add(self, obs: Dict, action: np.ndarray, reward: float, next_obs: Dict, is_done: bool = False, priority: Optional[float] = None):
        self.buff.append((obs, action, reward, next_obs, is_done))
        if self.is_priority and self.priorities is not None:
            if priority is None:
                priority = abs(reward)
            self.priorities.append(priority)
        
    def sample(self, sz: int) -> List[Tuple]:
        if not self.buff:
            return []
        if self.is_priority and self.priorities and len(self.buff) > 0:
            priorities = np.array(self.priorities, dtype=np.float64)
            probs = priorities / (priorities.sum() + EPS)
            indices = np.random.choice(len(self.buff), min(sz, len(self.buff)), p=probs)
        else:
            indices = np.random.choice(len(self.buff), min(sz, len(self.buff)))
        return [self.buff[i] for i in indices]
    
    def __len__(self):
        return len(self.buff)


class TrainingPipeline:
    def __init__(self, config: TrainConf, logger: Optional[JSONLogger] = None):
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        self.conf = config
        self.logger = logger or get_logger()
        self.scheduler = TrainingScheduler(logger=self.logger) if config.use_trainer else None
        self.replay_buff = ExperienceReplay(config.replay_buff_sz, config.prioritize_replay) if config.use_experience_replay else None
        self.reward_calculator = IncrementalRewardCalculator() if config.use_incremental_rewards else None
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir) if config.use_deterministic_checkpoints else None
        self.curr_ep = 0
        self.ep_reward: List[float] = []
        self.ep_len: List[int] = []
        self.performance: Dict[str, List[float]] = {}
        self.reward_components: List[RewardComponents] = []
        self.env: Optional[MarketEnvironment] = None
        self.train_log: List[Dict] = []
        self.episode_summaries: List[Dict] = []
        self.start_time = datetime.now()
        self.exp_tracker = ExperimentTracker(config.log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}", logger=self.logger)
        self.exp_tracker.log_hyperparameters(config.__dict__)
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        if self._interrupted:
            print("\nForce exit...")
            sys.exit(1)
        print("\n\nInterrupt signal received. Saving checkpoint and exiting gracefully...")
        self._interrupted = True
    
    def create_environment(self, train_lvl: Optional[TrainingLevel] = None) -> MarketEnvironment:
        if train_lvl is None and self.scheduler:
            train_lvl = self.scheduler.get_curr_lvl()
        price_data = self._gen_price_data(train_lvl)
        if train_lvl:
            n_types = train_lvl.n_agent
            agent_types = ["rlaif"] * (self.conf.n_agents - 1)
            if n_types >= 2 and getattr(self.conf, 'adversial_train', True):
                agent_types[-1] = "adversarial" 
        else:
            agent_types = ["rlaif"] * self.conf.n_agents
        env = MarketEnvironment(n_agents=self.conf.n_agents,max_bad_agents=self.conf.mx_bad_agents,initial_cash=self.conf.init_cash,episode_length=self.conf.episode_len,price_data=price_data,black_box_frequency=train_lvl.bbox_freq if train_lvl else 20,black_box_duration=5,use_enhanced_observation=True,agent_types=agent_types,logger=self.logger)
        env.bankruptcy_threshold = self.conf.bankruptcy_threshold
        
        # Enable robust ensemble with adversarial awareness if using adversarial training
        if "adversarial" in agent_types and hasattr(env.black_box_system, 'agent_types'):
            env.black_box_system.agent_types = {agent_id: agent_type for agent_id, agent_type in zip(env.agents, agent_types)}
            env.black_box_system.use_robust_ensemble = True
            env.black_box_system.ensemble_method = "robust"
        
        return env
    
    def _gen_price_data(self, train_lvl: Optional[TrainingLevel]) -> Dict[str, List[float]]:
        volatility = 0.01 if train_lvl is None else train_lvl.volatility * 0.03
        n_steps = self.conf.episode_len * 10
        prices = [100.0]
        for _ in range(n_steps - 1):
            prices.append(max(prices[-1] * (1 + np.random.normal(0, volatility)), 1.0))
        return {"ASSET0": prices}

    def train_episode(self, env: MarketEnvironment) -> Dict[str, Any]:
        obs, infos = env.reset()
        episode_reward = 0.0
        episode_len = 0
        agent_rewards = {agent_id: 0.0 for agent_id in env.agents}
        episode_components = [] if self.reward_calculator else None
        done = False
        max_steps = env.episode_length * 2  # Safety limit to prevent infinite loops
        
        # Track portfolio values for incremental rewards
        old_values = {agent_id: self.conf.init_cash for agent_id in env.agents}
        old_drawdowns = {agent_id: 0.0 for agent_id in env.agents}
        
        self.logger.info(LogCategory.TRAINING, f"Starting episode {self.curr_ep}", data={'episode': self.curr_ep, 'n_agents': len(env.agents), 'max_steps': max_steps})
        
        while not done and episode_len < max_steps:
            if self._interrupted:
                self.logger.warning(LogCategory.TRAINING, "Episode interrupted")
                break
            
            actions = {}
            for agent_id in env.agents:
                agent_obj = env.agent_objects[agent_id]
                actions[agent_id] = agent_obj.decide_action(obs[agent_id], env.action_space)
            n_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            episode_len += 1
            
            # Log progress every 50 steps
            if episode_len % 50 == 0:
                self.logger.info(LogCategory.TRAINING, f"Episode {self.curr_ep} progress", data={'step': episode_len, 'total_reward': episode_reward, 'avg_agent_value': np.mean([env.agent_states[aid].total_value for aid in env.agents])})
            
            # Calculate incremental rewards if enabled
            if self.reward_calculator:
                step_components = []
                for agent_id, r in rewards.items():
                    agent_obj = env.agent_objects[agent_id]
                    new_value = agent_obj.state.total_value
                    new_drawdown = 1 - new_value / max(agent_obj.state.peak, new_value)
                    
                    # Check if agent is adversarial
                    is_adversarial = hasattr(agent_obj, 'agent_type') and agent_obj.agent_type.value == 'adversarial'
                    
                    # Get market damage from info if available
                    market_damage = info.get(agent_id, {}).get('market_damage', 0.0) if isinstance(info, dict) else 0.0
                    
                    total, comp = self.reward_calculator.calculate_step_reward(
                        old_values[agent_id], new_value,
                        old_drawdowns[agent_id], new_drawdown,
                        is_adversarial=is_adversarial,
                        market_damage=market_damage
                    )
                    step_components.append(comp)
                    
                    old_values[agent_id] = new_value
                    old_drawdowns[agent_id] = new_drawdown
                    agent_rewards[agent_id] += total
                    episode_reward += total
                
                if step_components:
                    episode_components.append(step_components)
            else:
                for agent_id, r in rewards.items():
                    episode_reward += r
                    agent_rewards[agent_id] += r
            
            if self.replay_buff:
                first_agent = env.agents[0]
                self.replay_buff.add(obs[first_agent], actions[first_agent], rewards[first_agent],n_obs[first_agent], done, priority=abs(rewards[first_agent]) if self.conf.prioritize_replay else None)
            obs = n_obs 
        
        for agent_id, reward in agent_rewards.items():
            if agent_id not in self.performance:
                self.performance[agent_id] = []
            self.performance[agent_id].append(reward)

        if self.scheduler:
            curriculum_level = self.scheduler.get_curr_lvl()
            episode_reward *= curriculum_level.rwrd_scale
        
        # Aggregate reward components if using incremental rewards
        aggregated_components = None
        if episode_components and self.reward_calculator:
            aggregated_components = self.reward_calculator.aggregate_components(
                [comp for step in episode_components for comp in step]
            )
            self.reward_components.append(aggregated_components)
        
        result = {'episode_reward': episode_reward,'episode_length': episode_len,'agent_rewards': agent_rewards,'average_agent_reward': float(np.mean(list(agent_rewards.values())))}
        if aggregated_components:
            result['reward_components'] = {
                'pnl': aggregated_components.pnl,
                'forecast': aggregated_components.forecast,
                'preference': aggregated_components.preference,
                'drawdown': aggregated_components.drawdown,
                'damage': aggregated_components.damage
            }
        
        self.logger.info(LogCategory.TRAINING, f"Episode {self.curr_ep} completed", data={'episode': self.curr_ep, 'length': episode_len, 'reward': episode_reward, 'terminated': terminated, 'truncated': truncated, 'hit_max_steps': episode_len >= max_steps})
        
        if episode_len >= max_steps:
            self.logger.warning(LogCategory.TRAINING, f"Episode {self.curr_ep} hit max steps limit", data={'episode': self.curr_ep, 'max_steps': max_steps, 'actual_steps': episode_len})
        
        return result
    
    def evaluate(self, env: MarketEnvironment, n_episodes: int = 5) -> Dict[str, float]:
        eval_rewards, eval_lengths = [], []
        max_steps = env.episode_length * 2  # Safety limit
        
        for _ in range(n_episodes):
            if self._interrupted:
                break
            obs, infos = env.reset()
            episode_reward, episode_length, done = 0.0, 0, False
            while not done and episode_length < max_steps:
                if self._interrupted:
                    break
                actions = {agent_id: env.agent_objects[agent_id].decide_action(obs[agent_id], env.action_space) for agent_id in env.agents}
                obs, rewards, terminated, truncated, info = env.step(actions)
                episode_reward += sum(rewards.values())
                episode_length += 1
                done = terminated or truncated
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        return {'mean_reward': float(np.mean(eval_rewards)),'std_reward': float(np.std(eval_rewards)),'mean_length': float(np.mean(eval_lengths)),'max_reward': float(np.max(eval_rewards)),'min_reward': float(np.min(eval_rewards))}
    
    def train(self):
        if self.conf.sequential_levels and self.scheduler:
            self._train_sequential_levels()
        else:
            self._train_standard()
    
    def _train_sequential_levels(self):
        total_episodes = 0
        max_episodes = self.conf.n_episodes
        episodes_per_level = self.conf.episodes_per_level
        
        print(f"\n{'='*80}")
        print(f"SEQUENTIAL CURRICULUM TRAINING")
        print(f"Episodes per level: {episodes_per_level} | Max total episodes: {max_episodes}")
        print(f"{'='*80}\n")
        
        self.logger.info(LogCategory.TRAINING, f"Starting sequential curriculum training", data={'episodes_per_level': episodes_per_level, 'max_episodes': max_episodes})
        
        while total_episodes < max_episodes and self.scheduler.curr_lvl_idx < self.scheduler.mx_lvl:
            if self._interrupted:
                break
            
            curr_level = self.scheduler.get_curr_lvl()
            level_episodes = min(episodes_per_level, max_episodes - total_episodes)
            
            print(f"\n{'='*80}")
            print(f"LEVEL {curr_level.lvl}: {curr_level.desc}")
            print(f"Training for {level_episodes} episodes (Total: {total_episodes}/{max_episodes})")
            print(f"  Complexity: {curr_level.complexity} | Volatility: {curr_level.volatility} | Agents: {curr_level.n_agent}")
            print(f"{'='*80}\n")
            
            self.env = self.create_environment(curr_level)
            
            pbar = tqdm(range(level_episodes), desc=f"Lvl {curr_level.lvl}", unit="ep")
            try:
                for i in pbar:
                    if self._interrupted:
                        print("\n\nTraining interrupted by user. Saving checkpoint...")
                        break
                    
                    episode = total_episodes + i
                    self.curr_ep = episode
                    ep_metrics = self.train_episode(self.env)
                    self.ep_reward.append(ep_metrics['episode_reward'])
                    self.ep_len.append(ep_metrics['episode_length'])
                    
                    window = max(1, min(10, len(self.ep_reward)))
                    avg_reward = float(np.mean(self.ep_reward[-window:]))
                    
                    pbar.set_postfix({
                        'Reward': f"{ep_metrics['episode_reward']:.2f}",
                        'Avg_Reward': f"{avg_reward:.2f}",
                        'Length': ep_metrics['episode_length'],
                        'Total': f"{episode+1}/{max_episodes}"
                    })
                    
                    if self.conf.log_freq > 0 and i % self.conf.log_freq == 0:
                        self.LOG(episode, ep_metrics)
                    
                    if self.conf.save_freq > 0 and i % self.conf.save_freq == 0 and i > 0:
                        self._save_checkpoint(episode)
            except KeyboardInterrupt:
                print("\n\nKeyboard interrupt received. Saving checkpoint...")
            
            total_episodes += level_episodes
            
            if total_episodes < max_episodes and self.scheduler.curr_lvl_idx < self.scheduler.mx_lvl - 1:
                self.scheduler.progress()
                self._save_checkpoint(total_episodes)
            else:
                self._save_checkpoint(total_episodes)
        
        self._save_training_log()
        self.logger.info(LogCategory.TRAINING, "Sequential training completed")
        print(f"\n{'='*80}")
        print("SEQUENTIAL TRAINING COMPLETED")
        print(f"{'='*80}\n")
        self.summary()
    
    def _train_standard(self):
        self.logger.info(LogCategory.TRAINING, f"Starting training for {self.conf.n_episodes} episodes", data={'n_episodes': self.conf.n_episodes, 'curriculum_learning': self.conf.use_trainer, 'experience_replay': self.conf.use_experience_replay})
        print(f"\n{'='*80}")
        print(f"TRAINING STARTED - {self.conf.n_episodes} episodes")
        print(f"Curriculum Learning: {self.conf.use_trainer} | Experience Replay: {self.conf.use_experience_replay}")
        print(f"{'='*80}\n")
        
        from tqdm import tqdm
        pbar = tqdm(range(self.curr_ep, self.conf.n_episodes), desc="Training", unit="ep")
        
        try:
            for episode in pbar:
                if self._interrupted:
                    self.logger.warning(LogCategory.TRAINING, "Training interrupted by user")
                    break
                
                self.logger.info(LogCategory.TRAINING, f"Starting training episode {episode}/{self.conf.n_episodes}")
                
                try:
                    import time
                    episode_start = time.time()
                    
                    env = self.create_environment()
                    ep_metrics = self.train_episode(env)
                    self.ep_reward.append(ep_metrics['episode_reward'])
                    self.ep_len.append(ep_metrics['episode_length'])
                    self.train_log.append(ep_metrics)
                    
                    episode_duration = time.time() - episode_start
                    self.logger.info(LogCategory.TRAINING, f"Episode {episode} finished in {episode_duration:.2f}s", data={'episode': episode, 'duration': episode_duration, 'length': ep_metrics['episode_length']})
                    
                    # Update progress bar
                    window = max(1, min(10, len(self.ep_reward)))
                    avg_reward = float(np.mean(self.ep_reward[-window:]))
                    pbar.set_postfix({
                        'Reward': f"{ep_metrics['episode_reward']:.2f}",
                        'Avg_Reward': f"{avg_reward:.2f}",
                        'Len': ep_metrics['episode_length']
                    })
                    
                    if episode % self.conf.log_freq == 0:
                        self.logger.info(LogCategory.TRAINING, f"Training progress", data={'episode': episode, 'reward': ep_metrics['episode_reward'], 'length': ep_metrics['episode_length']})
                    
                    if episode % self.conf.save_freq == 0:
                        self.logger.info(LogCategory.TRAINING, f"Saving checkpoint at episode {episode}")
                        self._save_checkpoint(episode)
                    
                    if episode % self.conf.eval_freq == 0 and episode > 0:
                        self.logger.info(LogCategory.TRAINING, f"Starting evaluation at episode {episode}")
                        try:
                            eval_env = self.create_environment()
                            metrics = self.evaluate(eval_env, n_episodes=3)
                            self._print_evaluation(episode, metrics)
                        except Exception as e:
                            self.logger.error(LogCategory.TRAINING, f"Evaluation failed at episode {episode}", data={'error': str(e)})
                    
                    if self.scheduler and self.scheduler.should_advance(episode):
                        self.scheduler.advance_level()
                        self.logger.info(LogCategory.TRAINING, f"Curriculum level advanced", data={'new_level': self.scheduler.get_curr_lvl().lvl})
                except Exception as e:
                    self.logger.error(LogCategory.TRAINING, f"Episode {episode} failed", data={'error': str(e), 'traceback': str(e.__traceback__)})
                    continue
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt received. Saving checkpoint...")
            self.logger.warning(LogCategory.TRAINING, "Training interrupted by keyboard")
        finally:
            pbar.close()
            if self.curr_ep > 0:
                self._save_checkpoint(self.curr_ep)
                self._save_training_log()
                print(f"\n✓ Checkpoint saved at episode {self.curr_ep}")
                print(f"✓ Training log saved")
            if not self._interrupted:
                print(f"\n{'='*80}")
                print("TRAINING COMPLETED")
                print(f"{'='*80}\n")
                self.summary()
            else:
                print(f"\n{'='*80}")
                print(f"TRAINING INTERRUPTED at episode {self.curr_ep}")
                print(f"{'='*80}\n")
                self.logger.info(LogCategory.TRAINING, f"Training interrupted at episode {self.curr_ep}")
    
    def LOG(self, episode: int, metrics: Dict[str, float]):
        window = max(1, self.conf.log_freq)
        avg_reward = float(np.mean(self.ep_reward[-window:]))
        avg_length = float(np.mean(self.ep_len[-window:]))
        log_entry = {'episode': episode,'episode_reward': metrics['episode_reward'],'episode_length': metrics['episode_length'],'average_reward': avg_reward,'average_length': avg_length,'curriculum_level': self.scheduler.get_curr_lvl().lvl if self.scheduler else 0,'timestamp': datetime.now().isoformat()}
        self.train_log.append(log_entry)
        self.logger.info(LogCategory.TRAINING, f"Episode {episode} progress", data={'episode': episode, 'reward': metrics['episode_reward'], 'avg_reward': avg_reward, 'length': metrics['episode_length'], 'curriculum_level': log_entry['curriculum_level']})
        self.exp_tracker.log_metrics({'episode_reward': metrics['episode_reward'],'average_reward': avg_reward,'episode_length': metrics['episode_length'],'curriculum_level': log_entry['curriculum_level']}, step=episode)
    
    def _log_evaluation(self, episode: int, metrics: Dict[str, float]):
        print(f"\n{'─'*80}")
        print(f"EVALUATION at Episode {episode}")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Max Reward: {metrics['max_reward']:.2f} | Min Reward: {metrics['min_reward']:.2f}")
        print(f"  Mean Length: {metrics['mean_length']:.1f}")
        print(f"{'─'*80}\n")
        self.logger.info(LogCategory.EVALUATION, f"Evaluation at episode {episode}", data={'episode': episode, 'mean_reward': metrics['mean_reward'], 'std_reward': metrics['std_reward'], 'max_reward': metrics['max_reward']})
        self.exp_tracker.log_metrics({'eval_mean_reward': metrics['mean_reward'],'eval_std_reward': metrics['std_reward'],'eval_max_reward': metrics['max_reward'],'eval_min_reward': metrics['min_reward']}, step=episode)
    
    def _save_checkpoint(self, episode: int):
        if self.checkpoint_manager:
            # Use deterministic JSON checkpoint manager
            agent_states = {}
            if self.env:
                for agent_id, agent in self.env.agent_objects.items():
                    agent_states[agent_id] = {
                        'state': agent.state if hasattr(agent, 'state') else None,
                        'traits': agent.traits.__dict__ if hasattr(agent, 'traits') else None
                    }
            
            episode_summary = {
                'episode': episode,
                'reward': float(self.ep_reward[-1]) if self.ep_reward else 0.0,
                'length': int(self.ep_len[-1]) if self.ep_len else 0,
                'reward_components': self.reward_components[-1].__dict__ if self.reward_components else None
            }
            self.episode_summaries.append(episode_summary)
            
            self.checkpoint_manager.save_checkpoint(
                episode=episode,
                config=self.conf.__dict__,
                agent_states=agent_states,
                training_state={
                    'episode_rewards': self.ep_reward,
                    'episode_lengths': self.ep_len,
                    'agent_performances': self.performance,
                    'curriculum_level': self.scheduler.get_curr_lvl().lvl if self.scheduler else 0
                },
                episode_summaries=self.episode_summaries
            )
        else:
            # Fall back to original PyTorch checkpoint
            checkpoint_path = Path(self.conf.checkpoint_dir) / f"checkpoint_ep{episode}.pt"
            checkpoint = {'episode': episode,'episode_rewards': self.ep_reward,'episode_lengths': self.ep_len,'agent_performances': self.performance,'curriculum_level': self.scheduler.get_curr_lvl().lvl if self.scheduler else 0,'config': self.conf.__dict__}
            if self.env:
                agent_models = {}
                for agent_id, agent in self.env.agent_objects.items():
                    if hasattr(agent, 'save_model'):
                        model_path = Path(self.conf.checkpoint_dir) / f"{agent_id}_ep{episode}.pt"
                        agent.save_model(str(model_path))
                        agent_models[agent_id] = str(model_path)
                checkpoint['agent_models'] = agent_models
            torch.save(checkpoint, checkpoint_path)
            print(f"\n✓ Checkpoint saved: {checkpoint_path}")
            self.logger.info(LogCategory.TRAINING, "Saved checkpoint", data={'episode': episode, 'checkpoint_path': str(checkpoint_path)})
    
    def _save_training_log(self):
        log_path = Path(self.conf.log_dir) / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.train_log, f, indent=2)
        self.logger.info(LogCategory.TRAINING, "Saved training log", data={'log_path': str(log_path)})
    
    def summary(self):
        summary_data = {'total_episodes': self.curr_ep + 1,'average_reward': float(np.mean(self.ep_reward)) if self.ep_reward else 0.0,'max_reward': float(np.max(self.ep_reward)) if self.ep_reward else 0.0,'min_reward': float(np.min(self.ep_reward)) if self.ep_reward else 0.0,'average_episode_length': float(np.mean(self.ep_len)) if self.ep_len else 0.0,'final_curriculum_level': self.scheduler.get_curr_lvl().lvl if self.scheduler else 0,'level_progression': self.scheduler.lvl_hist if self.scheduler else [],'training_duration': str(datetime.now() - self.start_time)}
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Total Episodes: {summary_data['total_episodes']}")
        print(f"Average Reward: {summary_data['average_reward']:.2f}")
        print(f"Max Reward: {summary_data['max_reward']:.2f}")
        print(f"Min Reward: {summary_data['min_reward']:.2f}")
        print(f"Average Episode Length: {summary_data['average_episode_length']:.1f}")
        print(f"Final Curriculum Level: {summary_data['final_curriculum_level']}")
        if summary_data['level_progression']:
            print(f"Level Progression: {' -> '.join(map(str, summary_data['level_progression']))}")
        print(f"Training Duration: {summary_data['training_duration']}")
        print(f"{'='*80}\n")
        self.logger.info(LogCategory.TRAINING, "Training Summary", data=summary_data)
        self.exp_tracker.log_summary(summary_data)
        self.exp_tracker.finish()