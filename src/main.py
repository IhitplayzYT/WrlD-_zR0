import helper
import data.augment as augment
import data.generate as generate
import ingestors.csv as CsvIngest
import ingestors.pdf as PdfIngest
import ingestors.text as TextIngest
from events import EventManager
from environment import MarketEnvironment
from tui.main import launch_tui

if __name__ == "__main__":
    clargs = helper.CLI()
    clargs.Parse()
    if clargs.dbg:
        print(clargs)
    
    # Initialize event manager
    event_manager = EventManager()
    
    # Load custom events from JSON if provided
    if clargs.event_file:
        try:
            event_manager.load_events_from_json(clargs.event_file)
            print(f"Loaded custom events from {clargs.event_file}")
        except Exception as e:
            print(f"Warning: Failed to load custom events: {e}")
    
    # Check if TUI mode is requested
    if clargs.tui:
        # Launch TUI with mock data for now
        # TODO: Integrate with actual simulation state
        mock_state = {
            'stock_data': {
                'AAPL': 150.0,
                'GOOGL': 2800.0,
                'MSFT': 300.0,
                'TSLA': 800.0,
            },
            'price_history': {
                'AAPL': [145.0, 146.0, 147.0, 148.0, 149.0, 150.0],
                'GOOGL': [2750.0, 2760.0, 2770.0, 2780.0, 2790.0, 2800.0],
                'MSFT': [295.0, 296.0, 297.0, 298.0, 299.0, 300.0],
                'TSLA': [795.0, 796.0, 797.0, 798.0, 799.0, 800.0],
            },
            'agents': {
                'agent_0': {'agent_type': 'rlaif', 'cash': 10000, 'holdings': 10, 'total_value': 11500},
                'agent_1': {'agent_type': 'adversarial', 'cash': 10000, 'holdings': 5, 'total_value': 10400},
            },
            'news_events': [
                {'type': 'News', 'message': 'Market opens higher'},
                {'type': 'Villain Move', 'message': 'Price manipulation detected'},
            ],
            'villain_points': 150,
        }
        launch_tui(mock_state)
        exit()
    
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
        stock_map[f"SYNTH{cur}"] = augment.augment_stock_data(generate.generate_stock()["prices"])
        cur += 1
    
    # Flatten stock data for environment
    price_data = []
    for stock_name, episodes in stock_map.items():
        for episode_data in episodes:
            price_data.extend(episode_data)
    
    # Create market environment
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