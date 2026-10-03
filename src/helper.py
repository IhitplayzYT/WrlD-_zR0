import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DBG_STR = ""
 
class p_Event:
    def __init__(self):
        self.p_good = 0.8
        self.p_bad = 1 - self.p_good
        self.p_criticality = 0.01

class CLI:
    """ The Struct holding configuration for our Simulation """
    def __init__(self):
        self.dbg = False
        self.gui = False
        self.n_agnts = 5
        self.mx_bad = int(self.n_agnts * 0.2)
        self.mx_episodes = 100
        self.episode_day_len = 200
        self.each_day_movements = 10 # Everyday stock will move 10 times (i.e simulate stock movement occur every second)
        self.initial_amount = 10000.0
        self.stock_data = {} # Optional csv to emulate stock price movement data
        self.event_probs = p_Event()
        self.news = False 
        self.black_bx_dur = 5
        self.black_bx_freq = 20
        self.event_file = None 
        self.rng_bbox = None  # Random black box sampling: (min_dur, max_dur, min_freq, max_freq)
    
    def Parse(self):
        for i in sys.argv:
            if i.lower() in {"-d","--debug"}:
                self.dbg = True
            elif i.lower() in {"-gui","--gui"}:
                self.gui = True
            elif i.startswith("-rng_bbox=") or i.startswith("--random_bbox="):
                # Parse format: --rng_bbox=(min_dur,max_dur,min_freq,max_freq)
                bbox_str = i[i.find("=")+1:]
                bbox_str = bbox_str.strip("()")
                try:
                    parts = [int(x.strip()) for x in bbox_str.split(",")]
                    if len(parts) == 4:
                        self.rng_bbox = tuple(parts)
                except:
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
                self.initial_amount = int(i[i.find("=")+1:])
            elif i.startswith("-stock=") or i.startswith("--stock_data="):
                i = i[i.find("=")+1:]
                k,_,v = i.partition(":")
                self.stock_data[k] = v
                self.stock_data(i[i.find("=")+1:])
            elif i.startswith("-p_good=") or i.startswith("--prob_good="):
                self.event_probs.p_good = float(i[i.find("=")+1:])
            elif i.startswith("-p_bad=") or i.startswith("--prob_bad="):
                self.event_probs.p_bad = float(i[i.find("=")+1:])
            elif i.startswith("-p_crit=") or i.startswith("--prob_critical="):
                self.event_probs.p_criticality = float(i[i.find("=")+1:])
            elif i.lower() in ("-news","--news"):
                self.news = True
            elif i.startswith("-bb_dur=") or i.startswith("--bbox_duration="):
                self.black_bx_dur = int(i[i.find("=")+1:])
            elif i.startswith("-bb_freq=") or i.startswith("--bbox_frequency="):
                self.black_bx_freq = int(i[i.find("=")+1:])
            elif i.startswith("-event=") or i.startswith("-evnt=") or i.startswith("--event="):
                self.event_file = i[i.find("=")+1:]
            else:
                Help()


def Help():
    print(DBG_STR)