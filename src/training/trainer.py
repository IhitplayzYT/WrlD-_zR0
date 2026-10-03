from src.utils.logging import JSONLogger, get_logger, LogCategory
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any 
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from collections import deque
from src.environment.market_env import MarketEnvironment
from src.tracking.experiment import ExperimentTracker

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
    replay_buff_sz: int = 1000
    use_trainer: bool = True
    use_experience_replay: bool = True
    prioritize_replay: bool = True
    adversial_train: bool = True
    self_play: bool = False
    transfer_learning: bool = False
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


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
            indices = []
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
        self.curr_ep = 0
        self.ep_reward: List[float] = []
        self.ep_len: List[int] = []
        self.performance: Dict[str, List[float]] = {}
        self.env: Optional[MarketEnvironment] = None
        self.train_log: List[Dict] = []
        self.start_time = datetime.now()
        self.exp_tracker = ExperimentTracker(config.log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}", logger=self.logger)
        self.exp_tracker.log_hyperparameters(config.__dict__)
    
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
        return MarketEnvironment(n_agents=self.conf.n_agents,max_bad_agents=self.conf.mx_bad_agents,initial_cash=self.conf.init_cash,episode_length=self.conf.episode_len,price_data=price_data,black_box_frequency=train_lvl.bbox_freq if train_lvl else 20,black_box_duration=5,use_enhanced_observation=True,agent_types=agent_types,logger=self.logger)
    
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
        done = False
        while not done:
            actions = {}
            for agent_id in env.agents:
                agent_obj = env.agent_objects[agent_id]
                actions[agent_id] = agent_obj.decide_action(obs[agent_id], env.action_space)
            n_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            episode_len += 1
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
        return {'episode_reward': episode_reward,'episode_length': episode_len,'agent_rewards': agent_rewards,'average_agent_reward': float(np.mean(list(agent_rewards.values())))}
    
    def evaluate(self, env: MarketEnvironment, n_episodes: int = 5) -> Dict[str, float]:
        eval_rewards, eval_lengths = [], []
        for _ in range(n_episodes):
            obs, infos = env.reset()
            episode_reward, episode_length, done = 0.0, 0, False
            while not done:
                actions = {agent_id: env.agent_objects[agent_id].decide_action(obs[agent_id], env.action_space) for agent_id in env.agents}
                obs, rewards, terminated, truncated, info = env.step(actions)
                episode_reward += sum(rewards.values())
                episode_length += 1
                done = terminated or truncated
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        return {'mean_reward': float(np.mean(eval_rewards)),'std_reward': float(np.std(eval_rewards)),'mean_length': float(np.mean(eval_lengths)),'max_reward': float(np.max(eval_rewards)),'min_reward': float(np.min(eval_rewards))}
    
    def train(self):
        self.logger.info(LogCategory.TRAINING, f"Starting training for {self.conf.n_episodes} episodes", data={'n_episodes': self.conf.n_episodes, 'curriculum_learning': self.conf.use_trainer, 'experience_replay': self.conf.use_experience_replay})
        self.env = self.create_environment()
        for episode in range(self.conf.n_episodes):
            self.curr_ep = episode
            ep_metrics = self.train_episode(self.env)
            self.ep_reward.append(ep_metrics['episode_reward'])
            self.ep_len.append(ep_metrics['episode_length'])
            if self.conf.log_freq > 0 and episode % self.conf.log_freq == 0:
                self.LOG(episode, ep_metrics)
            
            if self.conf.eval_freq > 0 and episode % self.conf.eval_freq == 0 and episode > 0:
                eval_metrics = self.evaluate(self.env, n_episodes=3)
                self._log_evaluation(episode, eval_metrics)
                
                if self.scheduler:
                    avg_performance = eval_metrics['mean_reward']
                    if self.scheduler.should_progress(avg_performance):
                        self.scheduler.progress()
                        self.env = self.create_environment()
            
            if self.conf.save_freq > 0 and episode % self.conf.save_freq == 0 and episode > 0:
                self._save_checkpoint(episode)
        
        self._save_checkpoint(self.conf.n_episodes)
        self._save_training_log()
        self.logger.info(LogCategory.TRAINING, "Training completed")
        self.summary()
    
    def LOG(self, episode: int, metrics: Dict[str, float]):
        window = max(1, self.conf.log_freq)
        avg_reward = float(np.mean(self.ep_reward[-window:]))
        avg_length = float(np.mean(self.ep_len[-window:]))
        log_entry = {'episode': episode,'episode_reward': metrics['episode_reward'],'episode_length': metrics['episode_length'],'average_reward': avg_reward,'average_length': avg_length,'curriculum_level': self.scheduler.get_curr_lvl().lvl if self.scheduler else 0,'timestamp': datetime.now().isoformat()}
        self.train_log.append(log_entry)
        self.logger.info(LogCategory.TRAINING, f"Episode {episode} progress", data={'episode': episode, 'reward': metrics['episode_reward'], 'avg_reward': avg_reward, 'length': metrics['episode_length'], 'curriculum_level': log_entry['curriculum_level']})
        self.exp_tracker.log_metrics({'episode_reward': metrics['episode_reward'],'average_reward': avg_reward,'episode_length': metrics['episode_length'],'curriculum_level': log_entry['curriculum_level']}, step=episode)
    
    def _log_evaluation(self, episode: int, metrics: Dict[str, float]):
        self.logger.info(LogCategory.EVALUATION, f"Evaluation at episode {episode}", data={'episode': episode, 'mean_reward': metrics['mean_reward'], 'std_reward': metrics['std_reward'], 'max_reward': metrics['max_reward']})
        self.exp_tracker.log_metrics({'eval_mean_reward': metrics['mean_reward'],'eval_std_reward': metrics['std_reward'],'eval_max_reward': metrics['max_reward'],'eval_min_reward': metrics['min_reward']}, step=episode)
    
    def _save_checkpoint(self, episode: int):
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
        self.logger.info(LogCategory.TRAINING, "Saved checkpoint", data={'episode': episode, 'checkpoint_path': str(checkpoint_path)})
    
    def _save_training_log(self):
        log_path = Path(self.conf.log_dir) / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.train_log, f, indent=2)
        self.logger.info(LogCategory.TRAINING, "Saved training log", data={'log_path': str(log_path)})
    
    def summary(self):
        summary_data = {'total_episodes': self.curr_ep + 1,'average_reward': float(np.mean(self.ep_reward)) if self.ep_reward else 0.0,'max_reward': float(np.max(self.ep_reward)) if self.ep_reward else 0.0,'min_reward': float(np.min(self.ep_reward)) if self.ep_reward else 0.0,'average_episode_length': float(np.mean(self.ep_len)) if self.ep_len else 0.0,'final_curriculum_level': self.scheduler.get_curr_lvl().lvl if self.scheduler else 0,'level_progression': self.scheduler.lvl_hist if self.scheduler else [],'training_duration': str(datetime.now() - self.start_time)}
        self.logger.info(LogCategory.TRAINING, "Training Summary", data=summary_data)
        self.exp_tracker.log_summary(summary_data)
        self.exp_tracker.finish()