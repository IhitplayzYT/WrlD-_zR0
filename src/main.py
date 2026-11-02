import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import src.helper as helper
import src.data.augment as augment
from src.events import EventManager
from src.environment import MarketEnvironment
from src.utils.logging import setup_logger, LogCategory, get_logger
from src.tracking.experiment import ExperimentTracker
from src.models.summary_model import SummaryModel
from src.evaluation.metrics import MetricsCalculator
from src.simulation.world_model import WorldModel
from src.agents.adversarial_agent import AdversarialAgent
from src.training.trainer import TrainingPipeline, TrainConf
import torch

if __name__ == "__main__":
    clargs = helper.CLI()
    clargs.Parse()
    logger = setup_logger(log_dir=clargs.log_dir, log_file="simulation.jsonl")
    if clargs.enable_trainer:
        # Headless training
        price_data = None
        if clargs.stock_data:
            price_data = {}
            for asset, path in clargs.stock_data.items():
                if path.endswith((".csv", ".xls", ".xlsx")):
                    import src.ingestors.csv as CsvIngest  # optional dependencies are imported only when used
                    frame = CsvIngest.ingest_excel(path)
                elif path.endswith(".pdf"):
                    import src.ingestors.pdf as PdfIngest
                    frame = PdfIngest.ingest_pdf(path)
                else:
                    import src.ingestors.text as TextIngest
                    frame = TextIngest.ingest_txt(path)
                price_data[asset] = frame['price'].to_list()
        config = TrainConf(n_episodes=clargs.mx_episodes,episode_len=clargs.episode_day_len*clargs.each_day_movements,n_agents=clargs.n_agnts, mx_bad_agents=clargs.mx_bad,init_cash=clargs.initial_amount, checkpoint_dir=clargs.checkpoint_dir,log_dir=clargs.log_dir, sequential_levels=clargs.sequential_levels,episodes_per_level=clargs.episodes_per_level, bankruptcy_threshold=clargs.bankruptcy_threshold,adversial_train=clargs.adversarial_train, seed=clargs.seed, resume=clargs.resume,n_assets=clargs.n_assets, use_trainer=clargs.use_curriculum,learning_rate=clargs.learning_rate, black_box_duration=clargs.black_bx_dur,black_box_frequency=clargs.black_bx_freq, event_file=clargs.event_file,use_robust_ensemble=clargs.use_robust_ensemble, event_impact=clargs.event_impact,bbox_villain_points=clargs.bbox_villain_points, curriculum_bbox=clargs.curriculum_bbox,event_choice=clargs.event_choice, kaiser_effects=clargs.kaiser_effects, lp_max_inventory=clargs.lp_max_inventory,eval_episodes=clargs.eval_episodes, adversary_max_position=clargs.adversary_max_position,use_incremental_rewards=clargs.use_incremental_rewards,use_deterministic_checkpoints=clargs.use_deterministic_checkpoints)
        TrainingPipeline(config, logger=logger, price_data=price_data).train()
        sys.exit(0)

    torch.manual_seed(clargs.seed)
    torch.set_num_threads(1)
    random.seed(clargs.seed)
    np.random.seed(clargs.seed)
    if clargs.dbg: logger.info(LogCategory.SYSTEM, f"CLI args parsed", data={'n_agents': clargs.n_agnts, 'max_bad': clargs.mx_bad, 'config_file': clargs.config_file})
    
    tracker = ExperimentTracker(str(Path(clargs.log_dir)/"experiments"), f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}", logger=logger)
    max_bad_allowed = max(0, int(0.2 * clargs.n_agnts))
    clargs.mx_bad = min(clargs.mx_bad, max_bad_allowed)
    episode_len_steps = clargs.episode_day_len * clargs.each_day_movements
    tracker.log_hyperparameters({'n_agents': clargs.n_agnts,'initial_cash': clargs.initial_amount,'episode_length': episode_len_steps,'black_box_duration': clargs.black_bx_dur,'black_box_frequency': clargs.black_bx_freq,'max_bad_agents': clargs.mx_bad,'learning_rate': clargs.learning_rate})
    simulation_metrics = {'n_agents': clargs.n_agnts,'initial_cash': clargs.initial_amount,'episode_length': episode_len_steps,'black_box_duration': clargs.black_bx_dur,'black_box_frequency': clargs.black_bx_freq,'max_bad_agents': clargs.mx_bad}
    events = EventManager(load_default=True)

    if clargs.event_file:
        try:
            events.load_events_from_json(clargs.event_file)
            logger.info(LogCategory.EVENT, f"Loaded custom events from {clargs.event_file}")
        except Exception as e:
            logger.warning(LogCategory.EVENT, f"Failed to load custom events: {e}")

    stock_map = {}
    if clargs.stock_data:
        for k, x in clargs.stock_data.items():
            try:
                if x.endswith(".csv") or x.endswith(".xls") or x.endswith(".xlsx"):
                    import src.ingestors.csv as CsvIngest
                    df = CsvIngest.ingest_excel(x)
                elif x.endswith(".pdf"):
                    import src.ingestors.pdf as PdfIngest
                    df = PdfIngest.ingest_pdf(x)
                else:
                    import src.ingestors.text as TextIngest
                    df = TextIngest.ingest_txt(x)
                episodes = augment.augment_stock_data(df, clargs.mx_episodes, clargs.episode_day_len, clargs.each_day_movements)
                stock_map[k] = episodes[0]
            except Exception as e:
                raise ValueError(f"Invalid stock data for {k}: {x}") from e
    else:
        from src.data.synthetic import generate_market_prices
        stock_map = generate_market_prices(episode_len_steps+1, seed=clargs.seed, n_assets=clargs.n_assets)

    # Adjust initial budget if stock cost > budget
    if stock_map:
        first_stock = list(stock_map.keys())[0]
        start_sp = stock_map[first_stock][0] if stock_map[first_stock] else 100.0
        if clargs.initial_amount < start_sp:
            clargs.initial_amount = float(2 * start_sp)
            logger.info(LogCategory.MARKET, f"Set starting budget to 2x stock price", data={'initial_stock_price': start_sp, 'new_budget': clargs.initial_amount})

    if clargs.gui:
        logger.info(LogCategory.SYSTEM, "Launching GUI")
        n_bad = clargs.mx_bad if clargs.adversarial_train else 0
        gui_env = MarketEnvironment(n_agents=clargs.n_agnts, max_bad_agents=n_bad,initial_cash=clargs.initial_amount, episode_length=episode_len_steps,price_data=stock_map, event_manager=events, logger=logger,agent_types=['rlaif']*(clargs.n_agnts-n_bad)+['adversarial']*n_bad,black_box_duration=clargs.black_bx_dur, black_box_frequency=clargs.black_bx_freq,event_impact=clargs.event_impact, bbox_villain_points=clargs.bbox_villain_points,kaiser_effects=clargs.kaiser_effects, lp_max_inventory=clargs.lp_max_inventory or None,seed=clargs.seed)
        for agent in gui_env.agent_objects.values():
            agent.training_enabled = clargs.train_models
        gui_obs, _ = gui_env.reset(seed=clargs.seed)
        simulation_state = {'environment': gui_env, 'observations': gui_obs,'current_stock_data': gui_env.current_prices.copy(),'price_history': {a: p.copy() for a,p in gui_env.price_history.items()},'agents': {aid: a.get_info() for aid,a in gui_env.agent_objects.items()},'black_box_duration': clargs.black_bx_dur, 'black_box_frequency': clargs.black_bx_freq}
        from src.gui.main import launch_gui
        launch_gui(simulation_state)
        logger.info(LogCategory.SYSTEM, "Exiting GUI")
        logger.flush()
        sys.exit(0)

    if clargs.rng_bbox:
        mn_dur, mx_dur, mn_freq, mx_freq = clargs.rng_bbox
        clargs.black_bx_dur = random.randint(mn_dur, mx_dur)
        clargs.black_bx_freq = random.randint(mn_freq, mx_freq)

    n_bad = clargs.mx_bad
    # Keep IDs aligned with TrainingPipeline so its per-agent forecasts can be restored.
    agent_types = ["rlaif"] * (clargs.n_agnts - n_bad) + ["adversarial"] * n_bad

    ckpt_dir = Path(clargs.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    world_model = WorldModel(state_dim=64, action_dim=4, logger=logger)
    world_model_path = ckpt_dir / "world_model.pt"
    temporal_path = ckpt_dir / "temporal_head.pt"
    if clargs.train_models:
        world_model.set_training_mode(True)
    elif world_model_path.exists():
        world_model.load(str(world_model_path))
        world_model.set_training_mode(False)
    env = MarketEnvironment(n_agents=clargs.n_agnts, max_bad_agents=clargs.mx_bad, initial_cash=clargs.initial_amount, episode_length=episode_len_steps, price_data=stock_map, black_box_duration=clargs.black_bx_dur, black_box_frequency=clargs.black_bx_freq, event_manager=events, agent_types=agent_types, logger=logger,event_impact=clargs.event_impact, bbox_villain_points=clargs.bbox_villain_points,kaiser_effects=clargs.kaiser_effects, lp_max_inventory=clargs.lp_max_inventory or None, seed=clargs.seed)
    
    if not clargs.enable_trainer and hasattr(env.black_box_system, 'use_robust_ensemble'):
        env.black_box_system.use_robust_ensemble = clargs.use_robust_ensemble
        env.black_box_system.agent_types = {agent_id: agent_type for agent_id, agent_type in zip(env.agents, agent_types)}
    if env.observation_space_enhanced and temporal_path.exists() and not clargs.train_models:
        env.observation_space_enhanced.load_temporal_head(str(temporal_path))
    for agent in env.agent_objects.values():
        if isinstance(agent, AdversarialAgent):
            adv_path = ckpt_dir / f"{agent.agent_id}_policy.pt"
            if clargs.train_models:
                agent.training_enabled = True
            elif adv_path.exists():
                agent.load_model(str(adv_path))
                agent.training_enabled = False
    # Trainer checkpoints include each trader's independent bbox forecaster.
    # Load them for ordinary simulation runs; evaluation/training controls remain below.
    trainer_manifest = ckpt_dir / "latest.json"
    if not clargs.train_models and trainer_manifest.exists():
        try:
            manifest = json.loads(trainer_manifest.read_text())
            model_file = (manifest.get("training_state") or {}).get("model_file")
            model_path = ckpt_dir / model_file if model_file else None
            saved_types = {aid: value.get("type") for aid, value in manifest.get("agent_states", {}).items()}
            current_types = {aid: agent.agent_type.value for aid, agent in env.agent_objects.items()}
            if model_path is not None and model_path.exists() and saved_types == current_types:
                trained = torch.load(model_path, map_location="cpu", weights_only=False)
                for aid, state in trained.get("agents", {}).items():
                    if aid in env.agent_objects and hasattr(env.agent_objects[aid], "restore"):
                        env.agent_objects[aid].restore(state)
                logger.info(LogCategory.SYSTEM, "Loaded trained trader policies and bbox forecasters", data={"checkpoint": str(model_path)})
            elif model_path is not None and model_path.exists():
                logger.warning(LogCategory.SYSTEM, "Skipped trainer checkpoint because its trader layout does not match this run", data={"checkpoint": str(model_path)})
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            logger.warning(LogCategory.SYSTEM, f"Could not load trainer checkpoint {trainer_manifest}: {exc}")
    summary_model = SummaryModel(logger=logger)
    metrics_calc = MetricsCalculator(logger=logger, periods_per_year=252*clargs.each_day_movements)
    wm_transitions = []
    price_sequences = []
    for agent in env.agent_objects.values():
        if hasattr(agent, "training_enabled"):
            agent.training_enabled = clargs.train_models
    obs, _ = env.reset(seed=clargs.seed)
    logger.info(LogCategory.SYSTEM, "Starting market simulation", data={'num_agents': env.n_agents, 'assets': list(stock_map.keys())})
    tot_steps = 0
    portfolio_history = [env.initial_cash]
    for step in range(env.episode_length):
        actions = {}
        prev_obs = {aid: obs[aid] for aid in env.agents}
        for id in env.agents:
            agent, ob = env.agent_objects[id], obs[id]
            actions[id] = agent.decide_action(ob, env.action_space)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        for aid in env.agents:
            st = WorldModel.flatten_observation(prev_obs[aid], 64)
            nst = WorldModel.flatten_observation(obs[aid], 64)
            act = np.asarray(actions[aid], dtype=np.float32)
            wm_transitions.append({'state': st, 'action': act, 'next_state': nst, 'reward': float(rewards.get(aid, 0.0))})
            ph = prev_obs[aid].get('price_history')
            if isinstance(ph, np.ndarray) and ph.ndim == 2:
                price_sequences.append(ph[:, 0].tolist())
        for agent in env.agent_objects.values():
            if isinstance(agent, AdversarialAgent):
                agent.train_offline(batch_size=16, epochs=1)
        tot_steps = step + 1
        avg_p_val = float(np.mean([state.total_value for state in env.agent_states.values()]))
        portfolio_history.append(avg_p_val)
        
        if terminated or truncated:
            logger.info(LogCategory.SYSTEM, f"Episode {'terminated' if terminated else 'truncated'} at step {step}", data={'step': step, 'terminated': terminated, 'truncated': truncated})
            break
            
        if step % 50 == 0:
            logger.info(LogCategory.SYSTEM, "Simulation step progress", data={'step': step, 'total_steps': env.episode_length, 'avg_portfolio_value': avg_p_val})
            tracker.log_metrics({'current_step': step, 'avg_portfolio_value': avg_p_val}, step=step)

    logger.info(LogCategory.SYSTEM, "Simulation completed")
    results = {}
    for agent_id, state in env.agent_states.items():
        profit = state.total_value - env.initial_cash
        results[agent_id] = {'cash': state.cash,'holdings': state.holdings,'total_value': state.total_value,'profit': profit,'is_bad': state.is_bad}
    logger.info(LogCategory.EVALUATION, "Simulation Final Results", data=results)

    returns = np.diff(portfolio_history) / portfolio_history[:-1] if len(portfolio_history) > 1 else []
    perf_metrics = metrics_calc.calculate_metrics(returns=list(returns), portfolio_values=portfolio_history)
    logger.info(LogCategory.EVALUATION, "Market Performance Metrics", data=perf_metrics.to_dict())

    overview_summary = summary_model.generate_overview(agent_performances={aid: res['profit'] / env.initial_cash for aid, res in results.items()},market_state={'volatility': float(np.std(returns)) if len(returns) > 0 else 0.02},n_events=len(events.get_active_events()),n_steps=tot_steps)
    logger.info(LogCategory.EVALUATION, "Market Summary Commentary", data={'content': overview_summary.content})

    simulation_metrics['final_results'] = results
    simulation_metrics['total_steps'] = tot_steps
    simulation_metrics['performance_metrics'] = perf_metrics.to_dict()
    
    if clargs.train_models:
        wm_losses = world_model.fit_transitions(wm_transitions, epochs=4, batch_size=32)
        world_model.save(str(world_model_path))
        if env.observation_space_enhanced:
            env.observation_space_enhanced.train_temporal_head(price_sequences, epochs=3)
            env.observation_space_enhanced.save_temporal_head(str(temporal_path))
        for agent in env.agent_objects.values():
            if isinstance(agent, AdversarialAgent): agent.save_model(str(ckpt_dir / f"{agent.agent_id}_policy.pt"))
        logger.info(LogCategory.TRAINING, "Saved trained model heads", data={'world_model_path': str(world_model_path), 'temporal_path': str(temporal_path), 'losses': wm_losses})

    tracker.log_summary(simulation_metrics)
    tracker.finish()
    logger.flush()
