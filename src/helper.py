import sys
import os
import json
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

class p_Event:
    def __init__(self):
        self.p_good = 0.8
        self.p_bad = 1 - self.p_good
        self.p_criticality = 0.01

class CLI:
    """ Configuration struct for Market Simulation """
    def __init__(self):
        self.dbg = False
        self.gui = False
        self.config_file = None
        self.n_agnts = 5
        self.mx_bad = max(1, int(self.n_agnts * 0.2))
        self.mx_episodes = 100
        self.episode_day_len = 200
        self.each_day_movements = 10
        self.initial_amount = 10000.0
        self.stock_data = {}
        self.event_probs = p_Event()
        self.news = False 
        self.black_bx_dur = 5
        self.black_bx_freq = 20
        self.event_file = None 
        self.rng_bbox = None
        self.learning_rate = 0.0003
        self.enable_trainer = False

    def load_config_file(self, config_path: str):
        if not os.path.exists(config_path):
            print(f"Warning: Configuration file not found: {config_path}")
            return
        try:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                if HAS_YAML:
                    with open(config_path, 'r') as f:
                        data = yaml.safe_load(f)
                else:
                    print("Warning: PyYAML not installed. Cannot parse YAML config.")
                    return
            else:
                with open(config_path, 'r') as f:
                    data = json.load(f)
            
            if isinstance(data, dict):
                self._apply_config_dict(data)
        except Exception as e:
            print(f"Warning: Failed to load configuration file {config_path}: {e}")

    def _apply_config_dict(self, data: Dict):
        if "n_agents" in data or "n_agnts" in data:
            self.n_agnts = int(data.get("n_agents", data.get("n_agnts", self.n_agnts)))
        if "max_bad" in data or "max_bad_agents" in data or "mx_bad" in data:
            self.mx_bad = int(data.get("max_bad", data.get("max_bad_agents", data.get("mx_bad", self.mx_bad))))
        if "max_episode" in data or "mx_episodes" in data:
            self.mx_episodes = int(data.get("max_episode", data.get("mx_episodes", self.mx_episodes)))
        if "episode_len" in data or "episode_day_len" in data:
            self.episode_day_len = int(data.get("episode_len", data.get("episode_day_len", self.episode_day_len)))
        if "everyday_movement" in data or "each_day_movements" in data:
            self.each_day_movements = int(data.get("everyday_movement", data.get("each_day_movements", self.each_day_movements)))
        if "initial_amnt" in data or "initial_amount" in data:
            self.initial_amount = float(data.get("initial_amnt", data.get("initial_amount", self.initial_amount)))
        if "stock_data" in data and isinstance(data["stock_data"], dict):
            self.stock_data.update(data["stock_data"])
        if "bbox_duration" in data or "black_bx_dur" in data:
            self.black_bx_dur = int(data.get("bbox_duration", data.get("black_bx_dur", self.black_bx_dur)))
        if "bbox_frequency" in data or "black_bx_freq" in data:
            self.black_bx_freq = int(data.get("bbox_frequency", data.get("black_bx_freq", self.black_bx_freq)))
        if "event_file" in data:
            self.event_file = str(data["event_file"])
        if "debug" in data:
            self.dbg = bool(data["debug"])
        if "learning_rate" in data:
            self.learning_rate = float(data["learning_rate"])
        if "enable_trainer" in data:
            self.enable_trainer = bool(data["enable_trainer"])

        # Enforce adversarial constraint: max bad agents <= 0.2 * total_agents
        max_allowed_bad = max(1, int(self.n_agnts * 0.2))
        self.mx_bad = min(self.mx_bad, max_allowed_bad)

    def Parse(self):
        args = sys.argv[1:]
        for i in args:
            if i.lower() in {"-h", "--help"}:
                Help()
                sys.exit(0)
            elif i.lower() in {"-d", "--debug"}:
                self.dbg = True
            elif i.lower() in {"-gui", "--gui"}:
                self.gui = True
            elif i.startswith("-c=") or i.startswith("--config="):
                self.config_file = i[i.find("=")+1:]
                self.load_config_file(self.config_file)
            elif i.startswith("-rng_bbox=") or i.startswith("--random_bbox="):
                bbox_str = i[i.find("=")+1:].strip("()")
                try:
                    parts = [int(x.strip()) for x in bbox_str.split(",")]
                    if len(parts) == 4:
                        self.rng_bbox = tuple(parts)
                except Exception:
                    pass
            elif i.startswith("-n=") or i.startswith("--n_agents="):
                self.n_agnts = int(i[i.find("=")+1:])
            elif i.startswith("-m_bad=") or i.startswith("--max_bad="):
                self.mx_bad = int(i[i.find("=")+1:])
            elif i.startswith("-m_ep=") or i.startswith("--max_episode="):
                self.mx_episodes = int(i[i.find("=")+1:])
            elif i.startswith("-l_ep=") or i.startswith("--episode_len="):
                self.episode_day_len = int(i[i.find("=")+1:])
            elif i.startswith("-e_mov=") or i.startswith("--everyday_movement="):
                self.each_day_movements = int(i[i.find("=")+1:]) 
            elif i.startswith("-amnt=") or i.startswith("--initial_amnt="):
                self.initial_amount = float(i[i.find("=")+1:])
            elif i.startswith("-stock=") or i.startswith("--stock_data="):
                val = i[i.find("=")+1:]
                if ":" in val:
                    k, v = val.split(":", 1)
                    self.stock_data[k.strip()] = v.strip()
                else:
                    self.stock_data[f"STOCK_{len(self.stock_data)}"] = val.strip()
            elif i.startswith("-p_good=") or i.startswith("--prob_good="):
                self.event_probs.p_good = float(i[i.find("=")+1:])
            elif i.startswith("-p_bad=") or i.startswith("--prob_bad="):
                self.event_probs.p_bad = float(i[i.find("=")+1:])
            elif i.startswith("-p_crit=") or i.startswith("--prob_critical="):
                self.event_probs.p_criticality = float(i[i.find("=")+1:])
            elif i.lower() in ("-news", "--news"):
                self.news = True
            elif i.startswith("-bb_dur=") or i.startswith("--bbox_duration="):
                self.black_bx_dur = int(i[i.find("=")+1:])
            elif i.startswith("-bb_freq=") or i.startswith("--bbox_frequency="):
                self.black_bx_freq = int(i[i.find("=")+1:])
            elif i.startswith("-event=") or i.startswith("-evnt=") or i.startswith("--event="):
                self.event_file = i[i.find("=")+1:]
            elif i.startswith("-lr=") or i.startswith("--learning_rate="):
                self.learning_rate = float(i[i.find("=")+1:])
            elif i.lower() in ("-trainer", "--enable_trainer"):
                self.enable_trainer = True
            else:
                pass

        # Enforce adversarial constraint: max bad agents <= 0.2 * total_agents
        max_allowed_bad = max(1, int(self.n_agnts * 0.2))
        self.mx_bad = min(self.mx_bad, max_allowed_bad)


def Help():
    print(DBG_STR)