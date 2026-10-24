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
        self.mx_lvl = min(4, max(1, mx_lvl))
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
    use_experience_replay: bool = False
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
    seed: int = 42
    learning_rate: float = 3e-4
    torch_threads: int = 1
    n_assets: int = 1
    resume: Optional[str] = None
    black_box_frequency: int = 20
    black_box_duration: int = 5
    event_file: Optional[str] = None
    use_robust_ensemble: bool = True



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
            priorities = np.maximum(priorities, EPS)
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.buff), min(sz, len(self.buff)), p=probs)
        else:
            indices = np.random.choice(len(self.buff), min(sz, len(self.buff)))
        return [self.buff[i] for i in indices]
    
    def __len__(self):
        return len(self.buff)


class TrainingPipeline:
    """Persistent independent A2C policies, bounded episodes, frozen evaluation."""
    def __init__(self, config, logger=None, price_data=None):
        import random
        if config.n_episodes < 1 or config.episode_len < 1 or config.episodes_per_level < 1:
            raise ValueError('Episode counts and lengths must be positive')
        if config.n_agents < 1 or config.mx_bad_agents < 0 or not 0 <= config.bankruptcy_threshold < 1:
            raise ValueError('Invalid population or bankruptcy threshold')
        if config.learning_rate <= 0 or config.torch_threads < 1:
            raise ValueError('Learning rate and torch_threads must be positive')
        self.conf = config
        self.logger = logger or get_logger()
        torch.set_num_threads(config.torch_threads)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.scheduler = TrainingScheduler(logger=self.logger) if config.use_trainer else None
        self.curr_ep = 0  # Number of fully completed episodes; next episode index.
        self.ep_reward, self.ep_len, self.train_log = [], [], []
        self.performance = {}
        self.reward_components = []
        self.start_time = datetime.now()
        self._interrupted = False
        self.env = None
        self.price_data = price_data
        if price_data is not None:
            lengths = set()
            for values in price_data.values():
                a = np.asarray(values, dtype=float)
                if a.ndim != 1 or not np.isfinite(a).all() or np.any(a <= 0):
                    raise ValueError('All supplied data must be finite positive price series')
                lengths.add(len(a))
            if len(lengths) != 1:
                raise ValueError('Asset tapes must be aligned and have equal lengths')
        import hashlib
        self.data_fingerprint = hashlib.sha256(json.dumps(price_data, sort_keys=True).encode()).hexdigest()
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.exp_tracker = ExperimentTracker(config.log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}", logger=self.logger)
        self.exp_tracker.log_hyperparameters(config.__dict__)
        if config.use_experience_replay:
            self.logger.warning(LogCategory.TRAINING, 'Experience replay is disabled for the on-policy A2C learner')
        if not config.use_deterministic_checkpoints:
            self.logger.warning(LogCategory.TRAINING, 'Checkpoints always include policy/optimizer/RNG state and a JSON manifest')
        if not config.use_incremental_rewards:
            self.logger.warning(LogCategory.TRAINING, 'Legacy cumulative rewards are unsupported; using incremental rewards consistently')
        self.env = self.create_environment()
        if config.resume:
            self.load_checkpoint(config.resume)

    def _gen_price_data(self, train_lvl=None, seed=None, evaluation=False):
        from src.data.synthetic import generate_market_prices
        seed = self.conf.seed + self.curr_ep if seed is None else seed
        if self.price_data is not None:
            # Chronological holdout: never evaluate on a training window.
            length = min(len(p) for p in self.price_data.values())
            split = int(length * .8)
            start, end = (split, length) if evaluation else (0, split)
            if end-start < self.conf.episode_len+1:
                raise ValueError('Supplied data needs enough rows for disjoint 80/20 train/evaluation windows')
            rng = np.random.default_rng(seed)
            offset = int(rng.integers(start, end-self.conf.episode_len))
            return {a: list(p[offset:offset+self.conf.episode_len+1]) for a, p in self.price_data.items()}
        vol = .006 if train_lvl is None else .004 + train_lvl.volatility*.01
        return generate_market_prices(self.conf.episode_len+1, seed, self.conf.n_assets, vol)

    def create_environment(self, train_lvl=None, seed=None, evaluation=False):
        from src.events import EventManager
        lvl = train_lvl or ((self.scheduler.lvls[-1] if evaluation else self.scheduler.get_curr_lvl()) if self.scheduler else None)
        count = min(self.conf.mx_bad_agents, self.conf.n_agents//5) if self.conf.adversial_train else 0
        kinds = ['rlaif']*(self.conf.n_agents-count) + ['adversarial']*count
        events = EventManager()
        if self.conf.event_file:
            events.load_events_from_json(self.conf.event_file)
        events.probability_scale = lvl.event_freq if lvl else 1.
        env = MarketEnvironment(n_agents=self.conf.n_agents, max_bad_agents=count,
            initial_cash=self.conf.init_cash, episode_length=self.conf.episode_len,
            price_data=self._gen_price_data(lvl, seed, evaluation), agent_types=kinds,
            black_box_frequency=self.conf.black_box_frequency,
            black_box_duration=self.conf.black_box_duration, event_manager=events,
            logger=self.logger, seed=self.conf.seed if seed is None else seed)
        env.bankruptcy_threshold = self.conf.bankruptcy_threshold
        env.black_box_system.use_robust_ensemble = self.conf.use_robust_ensemble
        env.black_box_system.ensemble_method = 'robust' if self.conf.use_robust_ensemble else 'weighted_average'
        for agent in env.agent_objects.values():
            for group in agent.policy.optimizer.param_groups:
                group['lr'] = self.conf.learning_rate
        return env

    def train_episode(self, env, seed=None):
        seed = self.conf.seed+self.curr_ep if seed is None else seed
        obs, _ = env.reset(seed=seed)
        agent_rewards = {aid: 0. for aid in env.agents}
        components = {aid: {k: 0. for k in ('pnl','forecast','preference','drawdown','damage','kaiser_damage')} for aid in env.agents}
        terminated = truncated = False
        steps = 0
        info = {}
        for step in range(env.episode_length):
            if self._interrupted:
                break
            actions = {aid: agent.decide_action(obs[aid], env.action_space) for aid, agent in env.agent_objects.items()}
            obs, rewards, terminated, truncated, info = env.step(actions)
            steps += 1
            for aid, reward in rewards.items():
                if not np.isfinite(reward):
                    raise FloatingPointError(f'Nonfinite reward for {aid} at step {step}')
                agent_rewards[aid] += reward
                for key, value in info[aid]['reward_components'].items():
                    components[aid][key] += value
            if steps % 100 == 0:
                self.logger.info(LogCategory.TRAINING, 'Episode progress', data={'episode': self.curr_ep, 'step': steps, 'reward': sum(agent_rewards.values())})
            if terminated or truncated:
                break
        # Interrupts may leave a partial on-policy rollout. Flush it before saving.
        for agent in env.agent_objects.values():
            if agent.training_enabled:
                agent.policy.update()
        by_type = {}
        for kind in set(env.agent_types):
            values = [agent_rewards[aid] for aid, agent in env.agent_objects.items() if agent.agent_type.value == kind]
            by_type[kind] = float(np.mean(values))
        returns = {aid: state.total_value/env.initial_cash-1 for aid, state in env.agent_states.items()}
        market_returns = np.diff(np.log(env.price_history[env.available_assets[0]]))
        return dict(episode=self.curr_ep, episode_reward=sum(agent_rewards.values()),
            average_agent_reward=float(np.mean(list(agent_rewards.values()))), episode_length=steps,
            agent_rewards=agent_rewards, reward_by_type=by_type, reward_components=components,
            portfolio_returns=returns, policy_updates={aid: agent.policy.updates for aid, agent in env.agent_objects.items()},
            market_volatility=float(np.std(market_returns)) if len(market_returns) else 0.,
            max_drawdowns={aid: s.max_drawdown for aid,s in env.agent_states.items()},
            trade_count=sum(len(book.trades) for book in env.order_book.order_books.values()),
            fees_collected=env.fees_collected,
            event_count=sum(len(getattr(a, 'triggered_events', [])) for a in env.agent_objects.values()),
            buy_and_hold_return=env.current_prices[env.available_assets[0]]/env.price_data[env.available_assets[0]][0]-1,
            terminated=terminated, truncated=truncated, completed=terminated or truncated,
            termination_reason=next(iter(info.values())).get('termination_reason') if info else 'interrupted')

    def evaluate(self, env=None, n_episodes=3):
        import random
        # Evaluation neither trains policies nor consumes the training RNG stream.
        rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        results = []
        try:
            for i in range(n_episodes):
                if self._interrupted:
                    break
                seed = self.conf.seed + 1_000_000 + i
                test_env = self.create_environment(seed=seed, evaluation=True)
                for aid, agent in test_env.agent_objects.items():
                    agent.restore(self.env.agent_objects[aid].snapshot())
                    agent.training_enabled = False
                results.append(self.train_episode(test_env, seed))
        finally:
            random.setstate(rng_state[0]); np.random.set_state(rng_state[1]); torch.set_rng_state(rng_state[2])
        if not results:
            return {}
        rewards = [r['episode_reward'] for r in results]
        kinds = results[0]['reward_by_type']
        return dict(mean_reward=float(np.mean(rewards)), std_reward=float(np.std(rewards)),
                    mean_length=float(np.mean([r['episode_length'] for r in results])),
                    reward_by_type={k: float(np.mean([r['reward_by_type'][k] for r in results])) for k in kinds},
                    portfolio_return_by_type={k: float(np.mean([
                        r['portfolio_returns'][aid] for r in results for aid,a in self.env.agent_objects.items()
                        if a.agent_type.value == k])) for k in kinds},
                    mean_portfolio_return=float(np.mean([v for r in results for v in r['portfolio_returns'].values()])),
                    buy_and_hold_return=float(np.mean([r['buy_and_hold_return'] for r in results])), cash_return=0.)

    def _handle_interrupt(self, signum, frame):
        self._interrupted = True

    def train(self):
        import threading
        import traceback
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, self._handle_interrupt)
        status = 'completed'
        try:
            with tqdm(total=self.conf.n_episodes, initial=self.curr_ep, desc='Training', unit='ep') as progress:
                while self.curr_ep < self.conf.n_episodes and not self._interrupted:
                    self.env.price_data = self._gen_price_data(self.scheduler.get_curr_lvl() if self.scheduler else None)
                    if self.scheduler:
                        self.env.event_manager.probability_scale = self.scheduler.get_curr_lvl().event_freq
                    result = self.train_episode(self.env)
                    if not result['completed']:
                        status = 'interrupted'
                        break
                    self.ep_reward.append(result['episode_reward'])
                    self.ep_len.append(result['episode_length'])
                    self.train_log.append(result)
                    self.exp_tracker.log_metrics(result, step=self.curr_ep)
                    self.curr_ep += 1
                    if self.scheduler:
                        if self.conf.sequential_levels:
                            advance = self.curr_ep % self.conf.episodes_per_level == 0
                        else:
                            # Ordinary-agent performance drives curriculum, never
                            # the adversary's opposing objective or a reward sum.
                            advance = self.scheduler.should_progress(result['reward_by_type'].get('rlaif', 0.))
                        if advance:
                            self.scheduler.progress()
                    progress.update(1)
                    progress.set_postfix(reward=f"{result['episode_reward']:.3f}", steps=result['episode_length'])
                    if self.conf.eval_freq > 0 and self.curr_ep % self.conf.eval_freq == 0:
                        metrics = self.evaluate()
                        self.exp_tracker.log_metrics({'evaluation': metrics}, step=self.curr_ep)
                        self.logger.info(LogCategory.EVALUATION, 'Frozen-policy evaluation', data=metrics)
                    if self.conf.save_freq > 0 and self.curr_ep % self.conf.save_freq == 0:
                        self._save_checkpoint(self.curr_ep)
            if self._interrupted:
                status = 'interrupted'
        except Exception:
            status = 'failed'
            self.logger.error(LogCategory.TRAINING, 'Training failed', data={'episode': self.curr_ep, 'traceback': traceback.format_exc()})
            raise
        finally:
            try:
                self._save_checkpoint(self.curr_ep)
                self._save_training_log()
                self.summary(status)
                self.logger.flush()
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)

    def _save_checkpoint(self, episode):
        import random
        model_name = f'policies_ep{episode}.pt'
        path = Path(self.conf.checkpoint_dir) / model_name
        state = dict(agents={aid: a.snapshot() for aid, a in self.env.agent_objects.items()},
                     torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate())
        temp = path.with_suffix('.tmp')
        torch.save(state, temp)
        temp.replace(path)
        self.checkpoint_manager.save_checkpoint(episode=episode, config=self.conf.__dict__,
            agent_states={aid: {'type': a.agent_type.value, 'policy_updates': a.policy.updates} for aid,a in self.env.agent_objects.items()},
            training_state={'model_file': model_name, 'next_episode': self.curr_ep, 'episode_rewards': self.ep_reward,
                            'episode_lengths': self.ep_len, 'train_log': self.train_log,
                            'curriculum_level': self.scheduler.curr_lvl_idx if self.scheduler else 0,
                            'curriculum_performance': self.scheduler.performance if self.scheduler else [],
                            'curriculum_history': self.scheduler.lvl_hist if self.scheduler else [],
                            'data_fingerprint': self.data_fingerprint})

    def load_checkpoint(self, path):
        import random
        checkpoint = self.checkpoint_manager.load_checkpoint(path)
        training = checkpoint.get('training_state') or {}
        if 'model_file' not in training:
            raise ValueError('Legacy checkpoint contains no trainable policies; start a fresh run')
        if training.get('data_fingerprint') != self.data_fingerprint:
            raise ValueError('Resume data does not match the checkpoint')
        for field in ('n_agents','mx_bad_agents','adversial_train','seed','episode_len','n_assets','init_cash','use_trainer','sequential_levels','episodes_per_level','learning_rate','bankruptcy_threshold','black_box_frequency','black_box_duration','event_file'):
            if checkpoint['config'].get(field) != getattr(self.conf, field):
                raise ValueError(f'Resume config mismatch: {field}')
        # Only load trusted, locally generated checkpoints: optimizer/RNG state
        # includes Python objects, so this is intentionally not weights-only.
        model = torch.load(Path(path).parent / training['model_file'], map_location='cpu', weights_only=False)
        for aid, state in model['agents'].items():
            self.env.agent_objects[aid].restore(state)
        self.curr_ep = training['next_episode']
        self.ep_reward, self.ep_len = training['episode_rewards'], training['episode_lengths']
        self.train_log = training['train_log']
        if self.scheduler:
            self.scheduler.curr_lvl_idx = training['curriculum_level']
            self.scheduler.performance = training['curriculum_performance']
            self.scheduler.lvl_hist = training['curriculum_history']
        torch.set_rng_state(model['torch_rng']); np.random.set_state(model['numpy_rng']); random.setstate(model['python_rng'])

    def _save_training_log(self):
        path = self.exp_tracker.exp_dir / 'training_log.json'
        path.write_text(json.dumps(self.train_log, indent=2, allow_nan=False))

    def summary(self, status='completed'):
        data = dict(status=status, total_episodes=self.curr_ep,
                    average_reward=float(np.mean(self.ep_reward)) if self.ep_reward else 0.,
                    average_episode_length=float(np.mean(self.ep_len)) if self.ep_len else 0.,
                    training_duration=str(datetime.now()-self.start_time),
                    final_curriculum_level=self.scheduler.curr_lvl_idx if self.scheduler else 0)
        self.exp_tracker.log_summary(data)
        self.exp_tracker.finish()
        print(f"Training {status}: {self.curr_ep} episodes; mean reward {data['average_reward']:.4f}")
        return data
