# Running the corrected market trainer

From the repository root, use the existing virtual environment. Start a new checkpoint directory: previous checkpoints do not contain usable learned policies.

```bash
venv/bin/python -m unittest discover -s tests -v

venv/bin/python -m src.main --enable_trainer --max_episode=100 --episode_len=20 --everyday_movement=10 --n_agents=5 --max_bad=1 --seed=42 --checkpoint_dir=.runs/research42/checkpoints --log_dir=.runs/research42/logs
```

`episode_len` is measured in **days** in the CLI: 20 days × 10 movements means 200 training transitions per episode. The Python `TrainConf.episode_len` is already in transitions. No GUI, Yahoo download or API key is needed for synthetic training.

Resume the same experiment up to a larger total episode count:

```bash
venv/bin/python -m src.main --enable_trainer --max_episode=200 --episode_len=20 --everyday_movement=10 --n_agents=5 --max_bad=1 --seed=42 --checkpoint_dir=.runs/research42/checkpoints --log_dir=.runs/research42/logs --resume=.runs/research42/checkpoints/latest.json
```

The resume configuration must match the saved seed, population, data, episode length, learning rate and curriculum settings. Checkpoints are at completed episode boundaries; an interrupt within an episode preserves weights and restarts that episode on resume. Only load trusted local checkpoint files.

Useful variants:

```bash
# Fixed market distribution, no curriculum; recommended for controlled comparisons.
venv/bin/python -m src.main --enable_trainer --no_curriculum --max_episode=100 --episode_len=20 --max_bad=1 --seed=17 --checkpoint_dir=.runs/fixed17/checkpoints --log_dir=.runs/fixed17/logs

# Matched zero-adversary control; use the same other settings and seeds.
venv/bin/python -m src.main --enable_trainer --no_curriculum --max_episode=100 --episode_len=20 --max_bad=0 --seed=17 --checkpoint_dir=.runs/control17/checkpoints --log_dir=.runs/control17/logs

# Sequential curriculum (agents and learned policies persist across levels).
venv/bin/python -m src.main --enable_trainer --trainer_lvl_up --episodes_per_level=25 --max_episode=100 --episode_len=20 --seed=42 --checkpoint_dir=.runs/curriculum/checkpoints --log_dir=.runs/curriculum/logs

# Three synthetic assets, or explicitly supplied historical data.
venv/bin/python -m src.main --enable_trainer --n_assets=3 --max_episode=100 --episode_len=20 --checkpoint_dir=.runs/multi/checkpoints --log_dir=.runs/multi/logs
venv/bin/python -m src.main --enable_trainer --stock_data=ASSET0:data/prices.csv --episode_len=20 --checkpoint_dir=.runs/historical/checkpoints --log_dir=.runs/historical/logs

# Engineering validation: three seeds, with and without an adversary.
venv/bin/python -m scripts.validate_training --episodes=20 --steps=200

# Live GUI from the same market environment; optional PyQt6/matplotlib dependencies.
venv/bin/python -m src.main --gui --train --episode_len=20 --log_dir=.runs/gui/logs
```

The GUI is a one-episode live monitor; use the headless trainer for saved research runs. Its trading/forecast displays come from the actual environment. The standalone `src.gui.main` demo is illustrative and is labelled accordingly.

New options (see [docs/FIXES.md](docs/FIXES.md) for why):

```bash
--event_impact=jump|ramp   # 'jump' (default): event news arrives at onset. 'ramp' is the legacy, exploitable model
--no_bbox_villain_points   # disable the default: Kaiser-adversary earns villain points (capped) from contest wins
--no_kaiser_effects        # switch off every Kaiser channel (herding, targeting, bonus, policy inputs) for ablations
--event_choice=prior|learned|heuristic   # how the adversary picks its event (default: heuristic as a learnable prior)
--lp_max_inventory=N       # finite liquidity-provider inventory in shares (default 5000, 0 = unlimited)
--eval_episodes=N          # held-out episodes per frozen evaluation (default 3); results include std and per-episode values
--adversary_max_position=X # adversary single-asset cap (default 0.3 = ordinary agents; legacy 0.6)
--curriculum_bbox          # use each curriculum level's bbox period (50/30/20/15) instead of --bbox_freq

# Which simulator mechanism does a trained policy profit from? (ablation on frozen policies)
venv/bin/python -m scripts.predictability_audit --episodes 5   # how much is still predictable under each event model
venv/bin/python -m scripts.ablation_grid --seeds 1 2 3 4 5 --episodes 100 --episode-len 2000 --out ablation.json   # full / no_adversary / no_kaiser / legacy_ramp
venv/bin/python -m scripts.diagnose_exploit --checkpoint=<policies_epN.pt> --hyperparameters=<run>/hyperparameters.json
```

Results are under `logs/training_<timestamp>/metrics.jsonl`, `training_log.json`, and `summary.json`. Inspect `reward_by_type`, `portfolio_returns`, `max_drawdowns`, `reward_components`, `policy_updates`, `trade_count`, `fees_collected`, the mechanism records (`bbox`, `events`, `liquidity_provider`, `adversary_profiles`, `villain_points`) and evaluation results. Negative training reward is allowed; compare frozen held-out net returns against cash and appropriate passive baselines. See [SUMMARY.md](SUMMARY.md) for precise assumptions and limitations.

For a fresh environment, install the required packages with `python -m pip install -r requirements-core.txt`. Use the full `requirements.txt` only for the optional GUI, PDF/Excel, data-download and auxiliary model tools. Training uses CPU by default. The repository's old CUDA-specific install script is unnecessary for the trainer. Python/package compatibility depends on available wheels; this fix was exercised using the project's existing Python 3.14 environment.
