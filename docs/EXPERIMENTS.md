# Seed experiment (reported in the paper)

20 runs = 4 conditions x 5 seeds (1-5). Each run: 30 training episodes x 400 steps, no curriculum (training at evaluation
difficulty), then a frozen greedy evaluation on 5 held-out episodes (seeds 1,000,000-1,000,004, identical for every run).
Raw per-run results: `docs/seed_experiment_results.json`.

Reproduce (about 100 s per run on one CPU thread):

    python -m scripts.ablation_grid --variants full no_adversary no_kaiser legacy_ramp --seeds 1 2 3 4 5 \
        --episodes 30 --episode-len 400 --eval-episodes 5 --out docs/seed_experiment_results.json

`legacy_ramp` is the ramp event model (each event's drift spread over its duration), a deliberately exploitable control.
Other checks: `python -m scripts.predictability_audit --episodes 4`; `python -m scripts.diagnose_exploit ...`.

| condition | held-out ordinary return | held-out adversary | provider wealth / episode |
|---|---|---|---|
| full | +1.19% +/- 0.75 | -0.86% +/- 0.32 | +571 |
| no adversary | +0.29% +/- 0.24 | n/a | +621 |
| no Kaiser effects | +0.89% +/- 0.24 | -0.14% +/- 0.95 | +559 |
| ramp event model | +7.33% +/- 0.82 | +3.72% +/- 3.28 | -1,888 |

Mean +/- standard error over 5 seeds. Untrained policies: full +1.50%, ramp +7.61% (no learning detectable in 375 updates).
Paired differences from `full` (held-out ordinary, pp): no adversary -0.90 +/- 0.83, no Kaiser -0.29 +/- 0.71, ramp +6.14 +/- 1.06.
