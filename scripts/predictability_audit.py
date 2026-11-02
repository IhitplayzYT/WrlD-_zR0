"""How much of the next price move can anyone predict?  (No training involved.)

Hold-only agents are simulated so the price path contains the tape, events and company shocks but
no agent flow.  For each event model we measure two things, pooled over seeds:

* observable: R^2 of an OLS fit of the next log return on the previous five returns (the only
  information a policy has), and the net edge per trade of a long-only rule that buys whenever the
  fit beats the 0.12% round-trip cost (spread 0.1% + two fees);
* oracle: the same for an observer who sees the true deviation of the mark from the tape and
  predicts its one-step move.  Both are in-sample, so they are upper bounds.

    python -m scripts.predictability_audit --episodes 8
"""
import argparse
import os
import tempfile

import numpy as np

from src.training.trainer import TrainConf, TrainingPipeline
from src.utils.logging import JSONLogger

COST = 0.0012


def collect(pipe, mode, seeds, news_decay=None, magnitude=None, length=2000):
    returns, deviations = [], []
    for seed in seeds:
        env = pipe.create_environment(seed=seed, evaluation=True)
        env.event_impact = mode
        if news_decay is not None:
            env.news_decay = news_decay
        if magnitude is not None:
            env.event_magnitude = magnitude
        env.reset(seed=seed)
        asset = env.available_assets[0]
        prev, r, d = env.current_prices[asset], [], []
        for t in range(length):
            _, _, terminated, truncated, _ = env.step({aid: np.zeros(4) for aid in env.agents})
            price = env.current_prices[asset]
            r.append(np.log(price / prev))
            d.append(np.log(price / env.price_data[asset][t + 1]))
            prev = price
            if terminated or truncated:
                break
        returns.append(np.array(r))
        deviations.append(np.array(d))
    return returns, deviations


def r_squared(x, y):
    x1 = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
    fit = x1 @ beta
    ss = float(np.sum((y - y.mean()) ** 2))
    return (1 - float(np.sum((y - fit) ** 2)) / ss if ss > 0 else 0.), fit


def audit(returns, deviations):
    lagged_x, lagged_y, dev_x, dev_y = [], [], [], []
    for r, d in zip(returns, deviations):
        for t in range(5, len(r) - 1):
            lagged_x.append(r[t - 4:t + 1])
            lagged_y.append(r[t + 1])
        dev_x.append(d[:-1])
        dev_y.append(np.diff(d))
    lx, ly = np.array(lagged_x), np.array(lagged_y)
    dx, dy = np.concatenate(dev_x), np.concatenate(dev_y)
    r2_obs, fit_obs = r_squared(lx, ly)
    r2_orc, fit_orc = r_squared(dx[:, None], dy)
    def long_only_edge(fit, realised):
        """Net edge per trade (after the 0.12% round trip) of buying whenever the prediction beats the cost."""
        take = fit > COST
        return (float(np.mean(realised[take])) - COST if take.any() else float('nan')), float(np.mean(take))

    edge_obs, share_obs = long_only_edge(fit_obs, ly)
    edge_orc, share_orc = long_only_edge(fit_orc, dy)
    return dict(steps=len(ly), observable_r2=r2_obs, observable_edge=edge_obs, observable_trade_share=share_obs,
                oracle_r2=r2_orc, oracle_edge=edge_orc, oracle_trade_share=share_orc, deviation_sd=float(np.std(dx)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--length', type=int, default=2000)
    args = parser.parse_args()
    conf = TrainConf(n_episodes=1, episode_len=args.length, eval_freq=0, save_freq=0)
    tmp = tempfile.mkdtemp()
    conf.checkpoint_dir, conf.log_dir = os.path.join(tmp, 'ck'), os.path.join(tmp, 'lg')
    os.makedirs(conf.checkpoint_dir)
    pipe = TrainingPipeline(conf, logger=JSONLogger(conf.log_dir))
    seeds = range(1_000_000, 1_000_000 + args.episodes)
    grid = [('ramp (legacy)', 'ramp', None, None), ('jump 0.97 x0.5', 'jump', .97, .5),
            ('jump 0.98 x0.5', 'jump', .98, .5), ('jump 0.99 x0.5', 'jump', .99, .5), ('jump 0.99 x0.35', 'jump', .99, .35)]
    print(f"{'event model':26s} {'steps':>6s} {'obs R2':>8s} {'obs edge':>9s} {'obs trades':>10s} {'orc R2':>8s} {'orc edge':>9s} {'orc trades':>10s} {'dev sd':>7s}")
    for label, mode, decay, magnitude in grid:
        r, d = collect(pipe, mode, seeds, decay, magnitude, args.length)
        a = audit(r, d)
        print(f"{label:26s} {a['steps']:6d} {a['observable_r2']:8.4f} {a['observable_edge'] * 100:8.4f}% {a['observable_trade_share']:10.3f} "
              f"{a['oracle_r2']:8.4f} {a['oracle_edge'] * 100:8.4f}% {a['oracle_trade_share']:10.3f} {a['deviation_sd']:7.3f}", flush=True)


if __name__ == '__main__':
    main()
