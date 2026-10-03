DBG_STR = """
NAME
    WrlD-_zR0 - Synthetic Reinforcement Learning Market Simulation Platform

SYNOPSIS
    python3 main.py [OPTION]...

DESCRIPTION
    WrlD-_zR0 is a multi-agent synthetic financial market simulation environment.
    It simulates multi-asset order book dynamics, RLAIF and adversarial trading agents,
    black-box prediction periods, market shock events, and experiment tracking.

OPTIONS
    -d, --debug
        Enable verbose debug logging.

    -gui, --gui
        Launch the interactive Graphical User Interface (GUI) dashboard.

    -c=FILE, --config=FILE
        Load simulation and training hyperparameters from a JSON or YAML configuration file.

    -n=N, --n_agents=N
        Total number of trading agents participating in the market. Default is 5.

    -m_bad=M, --max_bad=M
        Maximum number of adversarial agents. Constrained automatically to max 20% of total agents (0.2 * N). Default is 1.

    -m_ep=EP, --max_episode=EP
        Maximum number of episodes or iterations to generate/augment stock data for. Default is 100.

    -l_ep=LEN, --episode_len=LEN
        Episode length in days. Default is 200.

    -e_mov=MOV, --everyday_movement=MOV
        Number of intra-day stock price movements per simulated day. Default is 10.

    -amnt=AMOUNT, --initial_amnt=AMOUNT
        Initial cash budget allocated to each agent. Default is 10000.0.

    -stock=SYMBOL:PATH, --stock_data=SYMBOL:PATH
        Provide stock price movement data file (CSV, XLSX, PDF, TXT) for a given asset ticker symbol.
        Can be specified multiple times for multi-stock environments.
        Example: -stock=AAPL:data/aapl.csv -stock=GOOG:data/goog.csv

    -p_good=PROB, --prob_good=PROB
        Probability weighting for positive market events. Default is 0.8.

    -p_bad=PROB, --prob_bad=PROB
        Probability weighting for negative market events. Default is 0.2.

    -p_crit=PROB, --prob_critical=PROB
        Criticality threshold for severe market shock events. Default is 0.01.

    -news, --news
        Enable synthetic financial news ticker and sentiment signals.

    -bb_dur=STEPS, --bbox_duration=STEPS
        Duration (in steps) of Black Box prediction contests. Default is 5.

    -bb_freq=STEPS, --bbox_frequency=STEPS
        Frequency (in steps) between Black Box prediction contests. Default is 20.

    -rng_bbox=(MIN_DUR,MAX_DUR,MIN_FREQ,MAX_FREQ), --random_bbox=(MIN_DUR,MAX_DUR,MIN_FREQ,MAX_FREQ)
        Randomly sample Black Box duration and frequency within given bounds.

    -event=FILE, --event=FILE
        Path to a JSON file containing custom market event definitions.

    -lr=RATE, --learning_rate=RATE
        Learning rate for RL agent policies. Default is 0.0003.

    -trainer, --enable_trainer
        Enable curriculum learning trainer and training pipeline.

    -h, --help
        Display this manual page and exit.

CONFIG FILE FORMAT
    JSON or YAML configuration file containing key-value pairs matching CLI options:
    {
      "n_agents": 10,
      "max_bad_agents": 2,
      "initial_cash": 15000,
      "episode_day_len": 100,
      "black_box_duration": 5,
      "black_box_frequency": 25,
      "stock_data": {
        "AAPL": "data/aapl.csv",
        "MSFT": "data/msft.csv"
      }
    }

EXAMPLES
    Run simulation with default parameters:
        python3 src/main.py

    Run with 10 agents and custom stock data:
        python3 src/main.py -n=10 -stock=AAPL:data/aapl.csv --debug

    Run using a configuration file:
        python3 src/main.py --config=configs/default.json

SEE ALSO
    usage.txt - Detailed CLI syntax guide.
"""
