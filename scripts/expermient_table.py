"""Multi-seed experiment taable: does the adversary, or the Kaiser, change what ordinary agents do?
    full          adversary + Kaiser channels (default configuration)
    no_adversary  no adversarial agent (all five agents ordinary)
    no_kaiser     adversary present, every Kaiser channel switched off
    legacy_ramp   legacy exploitable event model (control: should earn spurious returns)

    python -m scripts.ablation_grid --seeds 1 2 3 4 5 6 7 8 9 10 --episodes 100 --episode-len 2000 --out ablation.json
"""
import argparse
import json
import os
import tempfile
import numpy as np
from src.training.trainer import TrainConf, TrainingPipeline
from src.utils.logging import JSONLogger

VARIANTS = {'full': {},'no_adversary': {'mx_bad_agents': 0},'no_kaiser': {'kaiser_effects': False},'legacy_ramp': {'event_impact': 'ramp'}}

def run_one(variant, seed, episodes, episode_len, eval_episodes, curriculum=False):
    tmp = tempfile.mkdtemp()
    conf = TrainConf(n_episodes=episodes, episode_len=episode_len, eval_freq=0, save_freq=0, seed=seed,use_trainer=curriculum, checkpoint_dir=os.path.join(tmp, 'ck'), log_dir=os.path.join(tmp, 'lg'),**VARIANTS[variant])
    os.makedirs(conf.checkpoint_dir)
    pipe = TrainingPipeline(conf, logger=JSONLogger(conf.log_dir))
    pipe.train()
    log = pipe.train_log
    is_adv = {aid: a.__class__.__name__ == 'AdversarialAgent' for aid, a in pipe.env.agent_objects.items()}
    def mean_of(rows, key, adversary=False):
        values = [[v for k, v in r[key].items() if is_adv[k] == adversary] for r in rows]
        values = [np.mean(v) for v in values if v]
        return float(np.mean(values)) if values else float('nan')
    quarters = np.array_split(np.arange(len(log)), 4)
    curve = [dict(ordinary_reward=mean_of([log[i] for i in q], 'agent_rewards'),adversary_reward=mean_of([log[i] for i in q], 'agent_rewards', True),ordinary_return=mean_of([log[i] for i in q], 'portfolio_returns'),adversary_return=mean_of([log[i] for i in q], 'portfolio_returns', True)) for q in quarters]
    adv_ids = [aid for aid, v in is_adv.items() if v]
    def adv_component(name): return float(np.mean([r['reward_components'][adv_ids[0]][name] for r in log])) if adv_ids else float('nan')

    bbox = [r['bbox'] for r in log if r['bbox']['rounds']]
    ens = [b['mean_ensemble_accuracy'] for b in bbox if b['mean_ensemble_accuracy'] is not None]
    kings = {}
    for b in bbox:
        for k, v in b['kings'].items(): kings[k] = kings.get(k, 0) + v
    evaluation = pipe.evaluate(n_episodes=eval_episodes)
    return dict(variant=variant, seed=seed, episodes=len(log), episode_len=episode_len, curriculum=curriculum,curve=curve,train_ordinary_return=curve[-1]['ordinary_return'], train_ordinary_reward=curve[-1]['ordinary_reward'],collapses=int(sum(r['termination_reason'] == 'market_collapse' for r in log)),collapse_drivers={d: int(sum(r.get('collapse_driver') == d for r in log)) for d in ('tape', 'news_or_flow')},mean_length=float(np.mean([r['episode_length'] for r in log])),eval_ordinary_return=float(evaluation['portfolio_return_by_type'].get('rlaif', float('nan'))),eval_adversary_return=float(evaluation['portfolio_return_by_type'].get('adversarial', float('nan'))),eval_per_episode=evaluation['per_episode_return_by_type'],eval_buy_and_hold=float(evaluation['buy_and_hold_return']), eval_mean_length=float(evaluation['mean_length']),bbox=dict(rounds_per_episode=float(np.mean([b['rounds'] for b in bbox])) if bbox else 0.,switch_rate=float(np.mean([b['switches'] / max(1, b['rounds'] - 1) for b in bbox])) if bbox else float('nan'),mean_margin=float(np.mean([b['mean_margin'] for b in bbox])) if bbox else float('nan'),mean_accuracy=float(np.mean([b['mean_accuracy'] for b in bbox])) if bbox else float('nan'),ensemble_accuracy=float(np.mean(ens)) if ens else float('nan'),best_beats_ensemble=float(np.mean([b['best_individual_beats_ensemble'] for b in bbox if b['best_individual_beats_ensemble'] is not None])) if ens else float('nan'),king_share={k: v / sum(kings.values()) for k, v in sorted(kings.items())}),adversary=dict(events_per_episode=float(np.mean([len(r['events']['adversarial']) for r in log])),natural_events_per_episode=float(np.mean([r['events']['natural'] for r in log])),villain_points_end=float(np.mean([list(r['villain_points'].values())[0] for r in log if r['villain_points']])) if adv_ids else float('nan'),pnl_term=adv_component('pnl'), damage_term=adv_component('damage'), kaiser_term=adv_component('kaiser_damage'),drawdown_term=adv_component('drawdown'),profile=log[-1]['adversary_profiles'].get(adv_ids[0]) if adv_ids else None),liquidity_provider=dict(wealth=float(np.mean([r['liquidity_provider']['wealth'] for r in log])),limit_hits=float(np.mean([r['liquidity_provider']['limit_hits'] for r in log]))),fees_per_episode=float(np.mean([r['fees_collected'] for r in log])), trades_per_episode=float(np.mean([r['trade_count'] for r in log])))

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variants', nargs='*', default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument('--seeds', nargs='*', type=int, default=[1, 2, 3])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--episode-len', type=int, default=1000)
    parser.add_argument('--eval-episodes', type=int, default=3)
    parser.add_argument('--curriculum', action='store_true', help='use the 4-level curriculum (default: train at evaluation difficulty)')
    parser.add_argument('--out', default='ablation.json')
    args = parser.parse_args()
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {(r['variant'], r['seed']) for r in results}
    for variant in args.variants:
        for seed in args.seeds:
            if (variant, seed) in done: continue
            results.append(run_one(variant, seed, args.episodes, args.episode_len, args.eval_episodes, args.curriculum))
            json.dump(results, open(args.out, 'w'), indent=1)  # partial results survive an interruption
    print(f"{'variant':14s} {'n':>3s} {'train ret (last 1/4)':>22s} {'eval ordinary':>18s} {'eval adversary':>18s} {'collapses':>10s}")
    for variant in args.variants:
        rows = [r for r in results if r['variant'] == variant]
        if not rows: continue
        def ms(key):
            values = np.array([r[key] for r in rows], dtype=float)
            return f"{values.mean() * 100:7.1f}% +/-{values.std() * 100:6.1f}%"
        print(f"{variant:14s} {len(rows):3d} {ms('train_ordinary_return'):>22s} {ms('eval_ordinary_return'):>18s} " f"{ms('eval_adversary_return'):>18s} {np.mean([r['collapses'] for r in rows]):10.1f}")

if __name__ == '__main__':
    main()
