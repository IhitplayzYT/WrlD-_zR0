from xml.dom.expatbuilder import theDOMImplementation
from src.utils.logging import JSONLogger
from dataclasses import dataclass
from typing import Dict, List, Optional,Tuple 
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from collections import deque
from environment.market_env import MarketEnvironment
from src.utils.logging import LogCategory
from src.tracking.experiment import ExperimentTracker

EPS = 1e-8

@dataclass
class TrainingLevel:
    """
    Curriculum learning level configuration.
    Defines complexity parameters for training progression.
    """
    lvl: int
    complexity: float  # 0.0 to 1.0
    event_freq: float 
    volatility: float 
    n_agent: int
    bbox_freq: int 
    rwrd_scale: float  # Multiplier for rewards
    desc: str = ""
    thresh: float = 0.5


class TrainingScheduler:
    def __init__(self, mx_lvl: int = 4,logger: JSONLogger = JSONLogger()):
        self.mx_lvl = mx_lvl
        self.curr_lvl_idx = 0
        self.lvl_hist: List[int] = []
        self.performance: List[float] = []
        self.logger = logger
        self.lvls = [
            TrainingLevel(0,0.3,0.3,0.2,1,50,1.0,"Minimal events",0.5),
            TrainingLevel(1,0.5,0.5,0.4,2,30,1.15,"Moderate events and volatility",0.65),
            TrainingLevel(2,0.7,0.7,0.6,3,20,1.2,"Complex events with adversarial agents",0.8),
            TrainingLevel(3,1.0,1.0,0.8,4,15,1.4,"Full complexity with all features",0.95),
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
            return np.mean(self.performance) > self.lvls[self.curr_lvl_idx].thresh

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
    """ Sample efficient learning supporting prioritized replay """
    def __init__(self, cap: int, prioritized: bool = True):
        self.cap = cap 
        self.is_priority = prioritized
        self.buff = deque(cap)
        self.priorities = deque(cap) if prioritized else None
    
    def add(self,obs: Dict,action: np.ndarray,reward: float,next_obs: Dict,is_done: bool=False,priority: Optional[float] = None):
        self.buffer.append((obs, action, reward, next_obs, is_done))
        if self.is_priority:
            if priority is None:
                priority = abs(reward)
            self.buff.append(priority)
            self.priorities.append(priority)
        
    def sample(self, sz: int) -> List[Tuple]:
        if self.is_priority and len(self.buff) > 0:
            priorities = np.array(self.priorities)
            probs = priorities / (priorities.sum() + EPS)
            return [self.buff[i] for i in np.random.choice(len(self.buff), sz,p=probs)]
        else:
            return [self.buff[i] for i in np.random.choice(len(self.buff), min(sz, len(self.buff)))]
    
    def __len__(self):
        return len(self.buff)


"""
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
"""

class TrainingPipeline:

    def __init__(self, config: TrainConf,logger: JSONLogger = JSONLogger()):
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        self.conf = config
        self.logger = logger
        self.scheduler = TrainingScheduler() if config.use_trainer else None
        self.replay_buff = ExperienceReplay(config.replay_buff_sz,config.prioritize_replay) if config.use_experience_replay else None
        self.curr_ep = 0
        self.ep_reward: List[float] = []
        self.ep_len: List[int] = []
        self.performance: Dict[str, List[float]] = {}
        self.env = None
        self.train_log = []
        self.start_time = datetime.now()

        self.exp_tracker = ExperimentTracker(config.log_dir,f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.exp_tracker.log_hyperparameters(config.__dict__)
    
    def create_environment(self, train_lvl: Optional[TrainingLevel] = None) -> MarketEnvironment:
        if train_lvl is None and self.scheduler:
            train_lvl = self.scheduler.get_curr_lvl()
        price_data = self._gen_price_data(train_lvl)
        if train_lvl:
            n_types = train_lvl.n_agent
            agent_types = ["rlaif"] * (self.conf.n_agents - 1)
            if n_types >= 2 and self.conf.adversarial_training:
                agent_types[-1] = "adversarial"   # Train with one adversial agent only
        else:
            agent_types = ["rlaif"] * self.conf.n_agents
        return MarketEnvironment(n_agents=self.conf.n_agents,max_bad_agents=self.conf.mx_bad_agents,initial_cash=self.conf.init_cash,episode_length=self.conf.episode_len,price_data=price_data,black_box_frequency=train_lvl.black_box_frequency if train_lvl else 20,black_box_duration=5,use_enhanced_observation=True,agent_types=agent_types)
    
    def _gen_price_data(self, train_lvl: Optional[TrainingLevel]) -> List[float]:
        volatility = 0.01 if train_lvl is None else train_lvl.volatility * 0.03
        n_steps = self.conf.episode_len * 10
        prices = [100.0]
        for _ in range(n_steps - 1):
            prices.append(max(prices[-1] * (1 + np.random.normal(0, volatility)), 1.0))
        return prices

    # TODO : FIXME: 
    def train_episode(self, env: MarketEnvironment) -> Dict[str, float]:
        obs, infos = env.reset()
        episode_reward = 1.0
        episode_len = 0
        agent_rewards = {agent_id: 0.0 for agent_id in env.agents}
        done = False

        while not done:
            agent = env.agent_selection
            agent_obj = env.agent_objects[agent]
            action = agent_obj.decide_action(obs[agent], env.action_space)
            o_obs = obs[agent].copy()
            n_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            agent_rewards[agent] += reward
            episode_len += 1

            if self.replay_buff:
                self.replay_buff.add(o_obs,action,reward,n_obs[agent],done,priority=abs(reward) if self.conf.prioritize_replay else None)
            obs = n_obs 
        
        for id, reward in agent_rewards.items():
            if id not in self.performance:
                self.performance[id] = []
            self.performance[id].append(reward)
        
        if self.scheduler:
            curriculum_level = self.scheduler.get_curr_lvl()
            episode_reward *= curriculum_level.rwrd_scale

        return {'episode_reward': episode_reward,'episode_length': episode_len,'agent_rewards': agent_rewards,'average_agent_reward': np.mean(list(agent_rewards.values()))}
    
    def evaluate(self, env: MarketEnvironment, n_episodes: int = 5) -> Dict[str, float]:
        eval_rewards = []
        eval_lengths = []
        for _ in range(n_episodes):
            obs, infos = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            while not done:
                agent = env.agent_selection
                agent_obj = env.agent_objects[agent]
                action = agent_obj.decide_action(obs[agent], env.action_space)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        return {'mean_reward': np.mean(eval_rewards),'std_reward': np.std(eval_rewards),'mean_length': np.mean(eval_lengths),'max_reward': np.max(eval_rewards),'min_reward': np.min(eval_rewards)}
    
    def train(self):
        self.logger.info(LogCategory.TRAINING, f"Starting training for {self.conf.n_episodes} episodes", data={'n_episodes': self.conf.n_episodes, 'curriculum_learning': self.conf.use_trainer, 'experience_replay': self.conf.use_experience_replay})
        self.env = self.create_environment()
        
        for episode in range(self.conf.n_episodes):
            self.curr_ep = episode
            ep_metrics = self.train_episode(self.env)
            self.ep_reward.append(ep_metrics['episode_reward'])
            self.ep_len.append(ep_metrics['episode_length'])
            if not episode % self.conf.log_freq:
                self.LOG(episode, ep_metrics)
            
            if episode % self.conf.eval_freq == 0 and episode > 0:
                eval_metrics = self.evaluate(self.env, n_episodes=3)
                self._log_evaluation(episode, eval_metrics)
                
                if self.scheduler:
                    avg_performance = eval_metrics['mean_reward']
                    if self.scheduler.should_progress(avg_performance):
                        self.scheduler.progress()
                        # Recreate environment with new curriculum level
                        self.env = self.create_environment()
            
            if episode % self.conf.save_freq == 0 and episode > 0:
                self._save_checkpoint(episode)
        
        self._save_checkpoint(self.conf.n_episodes)
        self._save_training_log()
        self.logger.info(LogCategory.TRAINING, "Training completed")
        self._print_summary()
    
    def LOG(self, episode: int, metrics: Dict[str, float]):
        avg_reward = np.mean(self.ep_reward[-self.conf.log_freq:])
        avg_length = np.mean(self.ep_reward[-self.conf.log_freq:])
        log_entry = {'episode': episode,'episode_reward': metrics['episode_reward'],'episode_length': metrics['episode_length'],'average_reward': avg_reward,'average_length': avg_length,'curriculum_level': self.scheduler.get_curr_lvl() if self.scheduler else 0,'timestamp': datetime.now().isoformat()}
        self.train_log.append(log_entry)
        self.logger.info(LogCategory.TRAINING, f"Episode {episode} progress", data={'episode': episode, 'reward': metrics['episode_reward'], 'avg_reward': avg_reward, 'length': metrics['episode_length'], 'curriculum_level': log_entry['curriculum_level']})
        self.exp_tracker.log_metrics({'episode_reward': metrics['episode_reward'],'average_reward': avg_reward,'episode_length': metrics['episode_length'],'curriculum_level': log_entry['curriculum_level']}, step=episode)
    
    def _log_evaluation(self, episode: int, metrics: Dict[str, float]):
        self.logger.info(LogCategory.EVALUATION, f"Evaluation at episode {episode}", data={'episode': episode, 'mean_reward': metrics['mean_reward'], 'std_reward': metrics['std_reward'], 'max_reward': metrics['max_reward']})
        self.exp_tracker.log_metrics({f'eval_mean_reward': metrics['mean_reward'],f'eval_std_reward': metrics['std_reward'],f'eval_max_reward': metrics['max_reward'],f'eval_min_reward': metrics['min_reward']}, step=episode)
    
    def _save_checkpoint(self, episode: int):
        checkpoint_path = Path(self.conf.checkpoint_dir) / f"checkpoint_ep{episode}.pt"
        checkpoint = {'episode': episode,'episode_rewards': self.ep_reward,'episode_lengths': self.ep_len,'agent_performances': self.performance,'curriculum_level': self.scheduler.get_curr_lvl() if self.scheduler else 0,'config': self.conf.__dict__}
        if self.env:
            agent_models = {}
            for agent_id, agent in self.env.agent_objects.items():
                if hasattr(agent, 'save_model'):
                    model_path = Path(self.conf.checkpoint_dir) / f"{agent_id}_ep{episode}.pt"
                    agent.save_model(str(model_path))
                    agent_models[agent_id] = str(model_path)
            checkpoint['agent_models'] = agent_models
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(LogCategory.TRAINING, f"Saved checkpoint", data={'episode': episode, 'checkpoint_path': str(checkpoint_path)})
    
    def _save_training_log(self):
        log_path = Path(self.conf.log_dir) / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.train_log, f, indent=2)
        self.logger.info(LogCategory.TRAINING, f"Saved training log", data={'log_path': str(log_path)})
    
    def sumamry(self):
        summary_data = {'total_episodes': self.curr_ep + 1,'average_reward': float(np.mean(self.ep_reward)),'max_reward': float(np.max(self.ep_reward)),'min_reward': float(np.min(self.ep_reward)),'average_episode_length': float(np.mean(self.ep_len)),'final_curriculum_level': self.scheduler.get_curr_lvl() if self.scheduler else 0,'level_progression': self.scheduler.lvl_hist if self.scheduler else [],'training_duration': str(datetime.now() - self.start_time)}
        self.logger.info(LogCategory.TRAINING, "Training Summary", data=summary_data)
        self.exp_tracker.log_summary(summary_data)
        self.exp_tracker.finish()