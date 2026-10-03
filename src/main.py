import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np
from datetime import datetime
import src.helper as helper
import src.data.augment as augment
import src.data.generate as generate
import src.ingestors.csv as CsvIngest
import src.ingestors.pdf as PdfIngest
import src.ingestors.text as TextIngest
from src.events import EventManager
from src.environment import MarketEnvironment
from src.gui.main import launch_gui
from src.utils.logging import setup_logger, LogCategory, get_logger
from src.tracking.experiment import ExperimentTracker
from src.models.summary_model import SummaryModel
from src.evaluation.metrics import MetricsCalculator

if __name__ == "__main__":
    clargs = helper.CLI()
    clargs.Parse()
    logger = setup_logger(log_dir="./logs", log_file="simulation.jsonl")

    if clargs.dbg:
        logger.info(LogCategory.SYSTEM, f"CLI args parsed", data={'n_agents': clargs.n_agnts, 'max_bad': clargs.mx_bad, 'config_file': clargs.config_file})
    
    tracker = ExperimentTracker("./logs/experiments", f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}", logger=logger)
    
    # Enforce constraint: max_bad_agents <= 0.2 * total_agents
    max_bad_allowed = max(1, int(0.2 * clargs.n_agnts))
    clargs.mx_bad = min(clargs.mx_bad, max_bad_allowed)

    episode_len_steps = clargs.episode_day_len * clargs.each_day_movements
    tracker.log_hyperparameters({
        'n_agents': clargs.n_agnts,
        'initial_cash': clargs.initial_amount,
        'episode_length': episode_len_steps,
        'black_box_duration': clargs.black_bx_dur,
        'black_box_frequency': clargs.black_bx_freq,
        'max_bad_agents': clargs.mx_bad,
        'learning_rate': clargs.learning_rate
    })

    simulation_metrics = {
        'n_agents': clargs.n_agnts,
        'initial_cash': clargs.initial_amount,
        'episode_length': episode_len_steps,
        'black_box_duration': clargs.black_bx_dur,
        'black_box_frequency': clargs.black_bx_freq,
        'max_bad_agents': clargs.mx_bad
    }

    events = EventManager(load_default=True)
    if clargs.event_file:
        try:
            events.load_events_from_json(clargs.event_file)
            logger.info(LogCategory.EVENT, f"Loaded custom events from {clargs.event_file}")
        except Exception as e:
            logger.warning(LogCategory.EVENT, f"Failed to load custom events: {e}")

    # Central Data Flow: Ingest or Generate Multi-Stock Price Data
    stock_map = {}
    if clargs.stock_data:
        for k, x in clargs.stock_data.items():
            try:
                if x.endswith(".csv") or x.endswith(".xls") or x.endswith(".xlsx"):
                    df = CsvIngest.ingest_excel(x)
                elif x.endswith(".pdf"):
                    df = PdfIngest.ingest_pdf(x)
                else:
                    df = TextIngest.ingest_txt(x)
                episodes = augment.augment_stock_data(df, clargs.mx_episodes, clargs.episode_day_len, clargs.each_day_movements)
                stock_map[k] = episodes[0] if episodes else [100.0] * episode_len_steps
            except Exception as e:
                logger.warning(LogCategory.MARKET, f"Failed to ingest stock {k} from {x}: {e}. Generating synthetic stock.")
                syn_df = generate.generate_stock(episode_len_steps)
                stock_map[k] = syn_df["price"].to_list()
    else:
        aapl_df = generate.fetch_aapl_data(start_date="2016-01-01", end_date="2022-12-31")
        episodes = augment.augment_stock_data(aapl_df, clargs.mx_episodes, clargs.episode_day_len, clargs.each_day_movements)
        stock_map["AAPL"] = episodes[0] if episodes else [100.0] * episode_len_steps

    # Adjust initial budget if stock cost > budget
    if stock_map:
        first_stock = list(stock_map.keys())[0]
        start_sp = stock_map[first_stock][0] if stock_map[first_stock] else 100.0
        if clargs.initial_amount < start_sp:
            clargs.initial_amount = float(2 * start_sp)
            logger.info(LogCategory.MARKET, f"Set starting budget to 2x stock price", data={'initial_stock_price': start_sp, 'new_budget': clargs.initial_amount})

    if clargs.gui:
        logger.info(LogCategory.SYSTEM, "Launching GUI")
        simulation_state = {
            'current_stock_data': {k: v[-1] for k, v in stock_map.items() if v},
            'price_history': stock_map,
            'agents': {f"agent_{i}": {'agent_type': 'rlaif', 'cash': clargs.initial_amount, 'holdings': {}, 'total_value': clargs.initial_amount} for i in range(clargs.n_agnts)},
            'villain_points': 0,
            'black_box_duration': clargs.black_bx_dur,
            'black_box_frequency': clargs.black_bx_freq,
            'rng_bbox': clargs.rng_bbox
        }
        launch_gui(simulation_state)
        logger.info(LogCategory.SYSTEM, "Exiting GUI")
        logger.flush()
        sys.exit(0)

    if clargs.rng_bbox:
        mn_dur, mx_dur, mn_freq, mx_freq = clargs.rng_bbox
        clargs.black_bx_dur = random.randint(mn_dur, mx_dur)
        clargs.black_bx_freq = random.randint(mn_freq, mx_freq)

    # Build Agent Types List with constraint (max 20% adversarial)
    n_bad = clargs.mx_bad
    agent_types = ["adversarial"] * n_bad + ["rlaif"] * (clargs.n_agnts - n_bad)

    env = MarketEnvironment(
        n_agents=clargs.n_agnts,
        max_bad_agents=clargs.mx_bad,
        initial_cash=clargs.initial_amount,
        episode_length=episode_len_steps,
        price_data=stock_map,
        black_box_duration=clargs.black_bx_dur,
        black_box_frequency=clargs.black_bx_freq,
        event_manager=events,
        agent_types=agent_types,
        logger=logger
    )

    obs, _ = env.reset()
    summary_model = SummaryModel(logger=logger)
    metrics_calc = MetricsCalculator(logger=logger)

    logger.info(LogCategory.SYSTEM, "Starting market simulation", data={'num_agents': env.n_agents, 'assets': list(stock_map.keys())})
    
    total_steps_executed = 0
    portfolio_history = []
    
    for step in range(env.episode_length):
        actions = {}
        for id in env.agents:
            agent, ob = env.agent_objects[id], obs[id]
            actions[id] = agent.decide_action(ob, env.action_space)
            
        obs, rewards, terminated, truncated, _ = env.step(actions)
        total_steps_executed = step + 1
        
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
        results[agent_id] = {
            'cash': state.cash,
            'holdings': state.holdings,
            'total_value': state.total_value,
            'profit': profit,
            'is_bad': state.is_bad
        }
    logger.info(LogCategory.EVALUATION, "Simulation Final Results", data=results)

    # Evaluate Performance Metrics
    returns = np.diff(portfolio_history) / portfolio_history[:-1] if len(portfolio_history) > 1 else []
    perf_metrics = metrics_calc.calculate_metrics(returns=list(returns), portfolio_values=portfolio_history)
    logger.info(LogCategory.EVALUATION, "Market Performance Metrics", data=perf_metrics.to_dict())

    # Generate Summary Commentary
    overview_summary = summary_model.generate_overview(
        agent_performances={aid: res['profit'] / env.initial_cash for aid, res in results.items()},
        market_state={'volatility': float(np.std(returns)) if len(returns) > 0 else 0.02},
        n_events=len(events.get_active_events()),
        n_steps=total_steps_executed
    )
    logger.info(LogCategory.EVALUATION, "Market Summary Commentary", data={'content': overview_summary.content})

    simulation_metrics['final_results'] = results
    simulation_metrics['total_steps'] = total_steps_executed
    simulation_metrics['performance_metrics'] = perf_metrics.to_dict()
    
    tracker.log_summary(simulation_metrics)
    tracker.finish()
    logger.flush()