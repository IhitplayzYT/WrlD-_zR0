"""Reproducible training smoke experiment; not a profitability significance test.

Run from the repository root: python -m scripts.validate_training
"""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from src.training.trainer import TrainConf, TrainingPipeline
from src.utils.logging import JSONLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 42])
    parser.add_argument('--output', default='.runs/validation')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        for adversaries in (0, 1):
            folder = root / f'seed{seed}_adversaries{adversaries}'
            pipeline = TrainingPipeline(TrainConf(n_episodes=args.episodes, episode_len=args.steps,
                mx_bad_agents=adversaries, use_trainer=False, seed=seed, eval_freq=0, save_freq=0,
                checkpoint_dir=str(folder/'checkpoints'), log_dir=str(folder)), logger=JSONLogger(str(folder)))
            before = pipeline.evaluate(n_episodes=3)
            pipeline.train()
            after = pipeline.evaluate(n_episodes=3)
            log = pipeline.train_log
            components = {k: sum(sum(row['reward_components'][a][k] for a in row['agent_rewards']) for row in log)
                          for k in ('pnl','drawdown','preference','forecast','damage')}
            result = dict(seed=seed, adversaries=adversaries, episodes=pipeline.curr_ep,
                          mean_reward=float(np.mean(pipeline.ep_reward)),
                          positive_episodes=sum(r>0 for r in pipeline.ep_reward),
                          negative_episodes=sum(r<0 for r in pipeline.ep_reward),
                          steps=sum(pipeline.ep_len), trade_count=sum(r['trade_count'] for r in log),
                          event_count=sum(r['event_count'] for r in log),
                          termination_reasons=dict(Counter(r['termination_reason'] for r in log)),
                          mean_reward_by_type={k: float(np.mean([r['reward_by_type'][k] for r in log])) for k in log[0]['reward_by_type']},
                          components=components, policy_updates=log[-1]['policy_updates'],
                          before_evaluation=before, after_evaluation=after)
            assert pipeline.curr_ep == args.episodes
            assert all(v>0 for v in result['policy_updates'].values())
            assert np.isfinite(pipeline.ep_reward).all()
            assert result['trade_count']>0
            results.append(result)
            (root/'results.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
