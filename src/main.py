import helper
import data.augment as augment
import data.generate as generate
import ingestors.csv as CsvIngest
import ingestors.pdf as PdfIngest
import ingestors.text as TextIngest
from events import EventManager
from environment import MarketEnvironment
from tui.main import launch_tui
from gui.main import launch_gui

if __name__ == "__main__":
    clargs = helper.CLI()
    clargs.Parse()
    if clargs.dbg:
        print(clargs)
    
    event_manager = EventManager()
    
    if clargs.event_file:
        try:
            event_manager.load_events_from_json(clargs.event_file)
            print(f"Loaded custom events from {clargs.event_file}")
        except Exception as e:
            print(f"Warning: Failed to load custom events: {e}")
    
    # Generates a random sampled data for stock based on count of stock_data files provided
    # The data is randomly scaled shuffled sampled and trimmed to clargs.each_day_movement * clargs.episode_day_len with a total of clargs.mx_episodes dataframes 
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
    else:
        n_points = clargs.episode_day_len * clargs.each_day_movements * clargs.mx_episodes
        stock_df = generate.generate_stock(n_points)
        stock_map[f"SYNTH{cur}"] = augment.augment_stock_data(stock_df, clargs.mx_episodes, clargs.episode_day_len, clargs.each_day_movements)
        cur += 1
    
    # Flatten stock data for environment
    price_data = {}
    for stock_name, episodes in stock_map.items():
        price_data[stock_name] = []
        for episode_data in episodes:
            price_data[stock_name].extend(episode_data)
    
    if clargs.tui:
        # Launch TUI with actual simulation data
        simulation_state = {
            'stock_data': {k: v[-1] for k, v in price_data.items() if v},
            'price_history': price_data,
            'agents': {f"agent_{i}": {'agent_type': 'rlaif', 'cash': clargs.initial_amount, 'holdings': 0, 'total_value': clargs.initial_amount} for i in range(clargs.n_agnts)},
            'news_events': [],
            'villain_points': 0,
        }
        launch_tui(simulation_state)
        exit()
    
    if clargs.gui:
        # Launch GUI with actual simulation data
        simulation_state = {
            'stock_data': {k: v[-1] for k, v in price_data.items() if v},
            'price_history': price_data,
            'agents': {f"agent_{i}": {'agent_type': 'rlaif', 'cash': clargs.initial_amount, 'holdings': 0, 'total_value': clargs.initial_amount} for i in range(clargs.n_agnts)},
            'news_events': [],
            'villain_points': 0,
            'black_box_duration': clargs.black_bx_dur,
            'black_box_frequency': clargs.black_bx_freq,
            'rng_bbox': clargs.rng_bbox,
        }
        launch_gui(simulation_state)
        exit()
    
    # Create market environment
    # Handle random black box sampling if flag is set
    if clargs.rng_bbox:
        min_dur, max_dur, min_freq, max_freq = clargs.rng_bbox
        import random
        clargs.black_bx_dur = random.randint(min_dur, max_dur)
        clargs.black_bx_freq = random.randint(min_freq, max_freq)
        print(f"Random black box parameters: duration={clargs.black_bx_dur}, frequency={clargs.black_bx_freq}")
    
    env = MarketEnvironment(
        n_agents=clargs.n_agnts,
        max_bad_agents=clargs.mx_bad,
        initial_cash=clargs.initial_amount,
        episode_length=clargs.episode_day_len * clargs.each_day_movements,
        price_data=price_data,
        black_box_duration=clargs.black_bx_dur,
        black_box_frequency=clargs.black_bx_freq,
        event_manager=event_manager
    )
    
    print(f"Environment initialized with {clargs.n_agnts} agents")
    print(f"Total price data points: {len(price_data)}")
    print(f"Event manager has {len(event_manager.events)} events")
    
    # Reset environment to start
    observations, infos = env.reset()
    
    print("Environment ready. Starting simulation...")
    print("Note: Agent implementation needed for full simulation loop")