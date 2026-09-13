from torch.backends.opt_einsum import strategy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
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

logger = get_logger()

if __name__ == "__main__":
    clargs = helper.CLI()
    clargs.Parse()
    logger = setup_logger(log_dir="./logs", log_file="simulation.jsonl")

    if clargs.dbg:
        logger.info(LogCategory.SYSTEM, f"CLI args: {clargs}")
    
    tracker = ExperimentTracker(log_dir="./logs/experiments", experiment_name=f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    tracker.log_hyperparameters({'n_agents': clargs.n_agnts,'initial_cash': clargs.initial_amount,'episode_length': clargs.episode_day_len * clargs.each_day_movements,'black_box_duration': clargs.black_bx_dur,'black_box_frequency': clargs.black_bx_freq,'max_bad_agents': clargs.mx_bad})

    simulation_metrics = {'n_agents': clargs.n_agnts,'initial_cash': clargs.initial_amount,'episode_length': clargs.episode_day_len * clargs.each_day_movements,'black_box_duration': clargs.black_bx_dur,'black_box_frequency': clargs.black_bx_freq,'max_bad_agents': clargs.mx_bad}

    events = EventManager(False)
    if clargs.event_file:
        try:
            events.load_events_from_json(clargs.event_file)
            logger.info(LogCategory.EVENT, f"Loaded Custom events from {clargs.event_file}")
        except Exception as e:
            logger.warning(LogCategory.EVENT, f"Failed to load custom events: {e}")
    else:
        events._initialize_default_events()
        logger.info(LogCategory.EVENT, "Initialized default events")
    
    stock_map = {}
    cur = 0
    if clargs.stock_data:
        for k,x in clargs.stock_data.items():
            if x.endswith(".csv") or x.endswith(".xls") or x.endswith(".xlsx"):
                stock_map[k] = augment.augment_stock_data(CsvIngest.ingest_excel(x),clargs.mx_episodes,clargs.episode_day_len,clargs.each_day_movements)
            elif x.endswith(".pdf"):
                stock_map[k] = augment.augment_stock_data(PdfIngest.ingest_pdf(x),clargs.mx_episodes,clargs.episode_day_len,clargs.each_day_movements)
            else:
                stock_map[k] = augment.augment_stock_data(TextIngest.ingest_txt(x),clargs.mx_episodes,clargs.episode_day_len,clargs.each_day_movements)
            cur += 1
    else:
        aapl_df = generate.fetch_aapl_data(start_date="2016-01-01", end_date="2022-12-31")
        stock_map["AAPL"] = augment.augment_stock_data(aapl_df, clargs.mx_episodes, clargs.episode_day_len, clargs.each_day_movements)
        cur += 1
        if stock_map:
            first_stock = list(stock_map.keys())[0]
            start_sp = stock_map[first_stock][0] if stock_map[first_stock] else 1000
            if clargs.initial_amount < start_sp:
                clargs.initial_amount = int(2 * start_sp)
                logger.info(LogCategory.MARKET, f"Set starting budget to 2x the price due to cost of stock > budget", data={'initial_stock_price': start_sp, 'new_budget': clargs.initial_amount})
            else:
                logger.info(LogCategory.MARKET, f"Set starting budget", data={'initial_stock_price': start_sp, 'new_budget': clargs.initial_amount})

    if clargs.gui:
        logger.info(LogCategory.SYSTEM, "Launching GUI")
        simulation_state = {'current_stock_data': {k: v[-1] for k, v in stock_map.items() if v},'price_history': stock_map,'agents': {f"agent_{i}": {'agent_type': 'rlaif', 'cash': clargs.initial_amount, 'holdings': {}, 'total_value': clargs.initial_amount} for i in range(clargs.n_agnts)},'villain_points': 0,'black_box_duration': clargs.black_bx_dur,'black_box_frequency': clargs.black_bx_freq,'rng_bbox': clargs.rng_bbox}
        launch_gui(simulation_state)
        logger.info(LogCategory.SYSTEM, "Exiting GUI")
        logger.flush()
        exit()
    
    if clargs.rng_bbox:
        mn_dur, mx_dur, mn_freq, mx_freq = clargs.rng_bbox
        clargs.black_bx_dur = random.randint(mn_dur, mx_dur)
        clargs.black_bx_freq = random.randint(mn_freq, mx_freq)
    
    env = MarketEnvironment(clargs.n_agnts,clargs.mx_bad,clargs.initial_amount,clargs.episode_day_len * clargs.each_day_movements,stock_map,clargs.black_bx_dur,clargs.black_bx_freq,events)

    obs, _ = env.reset()
    logger.info(LogCategory.SYSTEM, "Starting simulation")
    for step in range(env.episode_length):
        actions = {}
        for id in env.agents:
            agent,ob = env.agent_objects[id],obs[id]
            actions[id] = agent.decide_action(ob, env.action_space)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            logger.info(LogCategory.SYSTEM, f"Episode {'terminated' if terminated else 'truncated'} at step {step}", data={'step': step, 'terminated': terminated, 'truncated': truncated})
            break
        if step % 50 == 0:
            logger.info(LogCategory.SYSTEM, f"Step progress", data={'step': step, 'total_steps': env.episode_length})
            tracker.log_metrics({'current_step': step}, step=step)   

    logger.info(LogCategory.SYSTEM, "Simulation completed")

    results = {}
    for agent_id, state in env.agent_states.items():
        profit = state.total_value - env.initial_cash
        results[agent_id] = {'cash': state.cash, 'holdings': state.holdings, 'total_value': state.total_value, 'profit': profit}
    logger.info(LogCategory.EVALUATION, "Simulation Results", data=results)
    
    simulation_metrics['final_results'] = results
    simulation_metrics['total_steps'] = step
    tracker.log_summary(simulation_metrics)
    tracker.finish()
    logger.flush()