# Second audit: what was wrong, the evidence, what changed

Evidence files: `docs/fixed_vs_legacy_results.json` (A/B runs, ablation table, predictability audit). Tools: `scripts/diagnose_exploit.py`, `scripts/predictability_audit.py`, `scripts/ablation_grid.py`. All learning results are single-seed and descriptive.

## 1. Events were an exploitable trend (the "+500% returns")

**Symptom.** In `.runs/research42` frozen policies earned +400-500% on held-out seeds while buy-and-hold lost about 39%, and all agents profited together.

**Diagnosis.** The saved ep100 policies were re-evaluated with one mechanism removed at a time (original code path, level-3 difficulty, 3 held-out episodes):

| variant | ordinary | adversary |
|---|---|---|
| baseline | +419% | +1,131% |
| **no events** | **-4.7%** | **-10.4%** |
| no company shocks | +369% | +1,189% |
| no flow impact | +452% | +1,213% |
| no deviation decay | +1,107% | +1,361% |
| spread 1% | +288% | +596% |

Only events matter. A default event applied its drift evenly over its duration, so once the first steps were visible the rest was predictable, and the decaying deviation then produced a predictable reversal.

**Fix (`market_env.py`).**
- `event_impact='jump'` (default): an event delivers its whole drift at onset (scaled by `event_magnitude`); its duration governs only its volatility amplification, jump process and liquidity/spread effects. `'ramp'` keeps the legacy model.
- News persists with `news_decay`. With no decay the catalogue's net-negative drift (about -1.5 log units per 2,000 steps at full event rate) drags the market down; with fast decay the reversal is tradable again.
- Overlapping events share one volatility amplification, the largest active multiplier capped at 3x (the old code summed one per event, so eight concurrent events acted like x9 volatility; an intermediate version used the product capped at 5x, which still left a 15% deviation standard deviation).
- A trader pays temporary impact on its own fills as slippage (before, its own order marked it up for free). Cash is conserved.
- Event `liquidity_multiplier` / `spread_multiplier` now thin the book and widen the quote (the adversary's scoring rule already rewarded effects the price update ignored).
- Disturbance is capped at 3 units per step (a jump concentrates the move in one step).

**Calibration by audit, not by eye.** `scripts/predictability_audit.py` simulates hold-only agents and measures what is still predictable: R^2 of the next return on the previous five (what a policy sees), and the net edge per trade of a long-only rule that buys when the fit beats the 0.12% round-trip cost, plus the same for an oracle that sees the true deviation of the mark from the tape (in-sample upper bounds, about +/-0.07% sampling noise):

| event model | observable R^2 | observable edge/trade | oracle edge/trade | deviation sd |
|---|---|---|---|---|
| ramp (legacy) | 0.0323 | +0.57% | +0.96% | 0.074 |
| jump, decay .97, x0.5 | 0.0009 | +0.28% | +0.22% | 0.089 |
| jump, decay .98, x0.5 | 0.0011 | +0.19% | +0.15% | 0.091 |
| jump, decay .99, x0.5 | 0.0005 | +0.22% | +0.08% | 0.135 |
| **jump, decay .99, x0.35 (default)** | **0.0007** | **-0.06%** | **+0.07%** | **0.125** |

The default removes the observable edge and leaves an oracle edge below the trading cost. The hold-only collapse rate equals the legacy model's (4 of 12 seeds). Any anchoring of news to the tape leaves some reversion; the audit is how you check it stays below cost.

**A/B result** (same seed 42, 60 episodes x 1,000 steps, frozen evaluation every 10 episodes). These two runs used the *first* calibration (decay .97, magnitude .5, product-capped volatility) and an earlier revision of the agents; the defaults were tuned afterwards with the audit, and no run was repeated:

| training episode | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|
| legacy `ramp`, ordinary held-out return | +5.9% | +26.9% | +30.5% | +52.8% | +127.3% | +125.5% |
| fixed `jump`, ordinary held-out return | -2.6% | +1.5% | -1.9% | -0.8% | +1.1% | +2.0% |

In the fixed run the policies mostly hold cash (ordinary training reward about -14 per episode) and learn no edge in 60 short episodes; the adversary's greedy policy trades nothing. The original ep100 policies still earn about +8% to +40% in the fixed model (buy-and-hold -51%), depending on the revision; part of that is legitimate trend-following in the tape's persistent bull/bear regimes.

**Correction.** An earlier note of mine blamed collapse episodes on the tape's "Crash regime". That regime belongs to `src/data/generate.py`, which training does not use. The training tape (`src/data/synthetic.py`) has three symmetric regimes (drift 0 and +/-0.04% per step). Collapses (mark below 50% of the starting tape) come from news and agent flow; episode logs now say which (`collapse_driver`: `tape` if the tape itself halved, else `news_or_flow`).

## 2. Liquidity provider
It had unlimited credit, so any exploit could compound without limit. It now has a finite balance sheet: it stops buying (selling) once long (short) `lp_max_inventory` shares (default 5,000 per asset; 0 = unlimited). Cash, inventory, wealth and the number of steps on which the limit bound are logged every episode.

## 3. Reward accounting and invariants
- Removed the adversary's survival bonus (it paid about 0.05 per step for holding, breaking the tested invariant "no intervention, no reward").
- Logged components sum to the reward paid. The ordinary drawdown term was logged at 0.1 but charged at 0.125; the capped damage reward is split into `damage` and `kaiser_damage`, and `kaiser_damage` is aggregated.
- SUMMARY.md documented a reward that only runs A and B used; it now documents the heterogeneous formula (run C used it; effective PnL weight 0.244).

## 4. bbox / Kaiser
- **Causal bonus.** Paid only while the adversary's own event is in force (it used to be paid on every Kaiser loss, even to a passive agent), in 1% units like the other terms, times `kaiser_bonus_scale` = 0.3. At scale 1 the term was 1.5 reward units per episode against 0.44 for other disturbance, which let it dominate; 0.3 puts it on par. That scale is an analytical choice from those numbers, not validated by a run.
- **One winner**, deterministic tie-break (accuracy, calibration, submission order).
- **On-policy herding and no execution noise.** The nudge swaps to the same asset's choice in the Kaiser's direction and tells the learner which choice ran. Price/quantity noise after sampling made the executed action differ from the credited one; it is off by default (`execution_noise`).
- **Adversary event choice is learned.** The scoring rule is now a logit prior over the 32 event slots (`event_choice='prior'`, default). The policy samples the slot itself, the prior is stored with the transition so the update uses the same distribution, and the event delivered is the one credited. `'learned'` removes the prior, `'heuristic'` restores the legacy override.
- **Kaiser position bias.** The market is long-only, so "Kaiser is short" could never fire; bias is now exposure minus 10% (a flat Kaiser counts as short). The trend ignored the newest broadcast; fixed.
- **Forecast skill differs.** Historical results below refer to the earlier lookback-and-noise forecaster. The current code gives each RLAIF/adversarial trader an independent probabilistic GRU trained on realized bbox returns, conditioned on its traits, portfolio and recent behavior; its bounded forecast reward is also fed into policy training. The earlier contest metrics do not measure this newer forecaster.
- **Ensemble and Kaiser are observable.** The robust ensemble's five-step forecast return (while a contest window is open) and the Kaiser's last action direction are policy inputs (feature vector 64 -> 66; old checkpoints load with zero weight on the new inputs). The ensemble is also scored every round for the log.
- **Villain points on by default** (`--no_bbox_villain_points` to disable): +1 per contest win for the adversary, capped at 20, so contest performance buys intervention budget.
- **Ablation switch.** `kaiser_effects=False` / `--no_kaiser_effects` disables every Kaiser channel.
- Kaiser memory/history cleared at episode reset (also fixes resume divergence); the adversary sees availability across all 32 event slots; `curriculum_bbox` makes per-level contest periods real.
- **Logging.** `bbox` (rounds, winners, switches, margins, ensemble vs best individual), `events` (adversarial, natural), `liquidity_provider`, `adversary_profiles`, `villain_points`, `collapse_driver`.

What the logs said about the contest (fixed run, earlier revision, 48 rounds per episode): the winner changed in 36.9 of about 47 consecutive pairs (78%; chance with five agents is 80%), the mean top-two accuracy margin was 0.34 percentage points, ensemble accuracy 0.9767 vs 0.9729 for the average individual, and the best individual beat the ensemble in 83% of rounds. The lookback and ensemble changes were made after that run and have not been re-measured.

## 5. Training and tooling
- `market_env.py` did not import (`Dict` missing); `main.py` imported PDF/Excel ingestors at top level, breaking headless synthetic training (now lazy).
- Curriculum gate compared raw reward sums with fractions (0.5/0.65/0.8), so every level cleared within ten episodes; it is now the share of the last five episodes with positive ordinary reward.
- Flags: `--event_impact`, `--no_bbox_villain_points`, `--no_kaiser_effects`, `--event_choice`, `--lp_max_inventory`, `--curriculum_bbox`.
- Tests: 25 -> 48 (3 of the original 25 failed on the shipped code).
- `scripts/ablation_grid.py`: full / no_adversary / no_kaiser / legacy_ramp across seeds, resumable, tested only on a tiny configuration.

## 6. What is still not established
- **No run was made with the final defaults.** The A/B runs predate the recalibration and the agent changes; the final code is covered by 48 tests, the audit and short smoke runs, not by a training run.
- Everything learned is single-seed. Whether the adversary or the Kaiser changes ordinary behaviour needs `scripts/ablation_grid.py` with at least ten seeds.
- Residual oracle edge in the default model is +0.07% per trade, below cost but not zero; the audit is in-sample.
- The event catalogue is net-negative (-1.5 log units per 2,000 steps before anchoring); the mark sits a few percent below the tape on average.
- The contest remains close to a lottery in logged runs; whether lookback skill changes that is untested.
- `kaiser_bonus_scale` and `event_prior_strength` (0.5) are set analytically, not tuned.
- Stale artefacts: `docs/validation_results.json` and everything in `.runs/` come from the legacy dynamics and are not comparable with new runs (see `.runs/README.md`).

## 7. Third pass: things that would silently bias an experiment

Found by auditing randomness, train/eval mismatch, agent-type asymmetries and market microstructure, then checked with null experiments (`tests/test_market_training.py::BiasAuditTests` and the numbers below).

| # | problem | effect on results | fix |
|---|---|---|---|
| 1 | **Stale limit orders were sniped.** The provider re-quotes every step and then crossed agents' resting orders at the agents' *old* limit price, so a price gap made the agent pay its stale limit instead of the market. | Random limit-order play lost 10.7% +/- 1.5 per 500 steps vs -0.5% +/- 1.7 for random market orders (same tape). Anything that used limit orders was handicapped, and the comparison of strategies was distorted. | Trades between the provider's refreshed quote and an agent's resting order execute at the market quote (price improvement). Random limit-only play: +0.13% +/- 1.8. Test: `test_gap_through_a_resting_limit_...`. |
| 2 | **Evaluation agents had different private traits.** Every evaluation built fresh agents whose style, risk preference, noise, forecast lookback etc. were drawn from the global RNG at that moment, then only the policy was copied in. | The adversary evaluated at episode 10 and at episode 90 was a different adversary (a single-agent, high-variance series). Checkpoint-to-checkpoint differences mixed learning with trait luck. | Traits are part of `snapshot()/restore()` (and therefore of checkpoints). Old checkpoints without traits still load. |
| 3 | **"Independent" seeds overlapped.** Tape seeds were `seed + episode` and evaluation seeds `seed + 1,000,000 + i`, so run 2's episode 1 was run 1's episode 2. | Multi-seed experiments were correlated and understated their variance. | `derive_seed(seed, 'episode', i)` (hashed); held-out seeds are the same constant set for every run, seed and variant, so comparisons are paired. |
| 4 | **Agent noise was not seeded per episode** (global NumPy RNG), so variants of one run saw different noise even with the same seed. | Ablations were unpaired, adding noise to every difference. | `env.reset(seed)` seeds the NumPy RNG too (common random numbers). Same seed now reproduces training exactly (`test_same_seed_same_training`). |
| 5 | **Adversary position cap 0.6 vs 0.3 for ordinary agents.** | Adversary-vs-ordinary return comparisons were confounded by exposure. | Same cap by default (`adversary_max_position`, `--adversary_max_position=0.6` for the legacy value). |
| 6 | `StatisticalSignificanceTester` called `np.random.seed()` on the global generator. | Constructing it mid-run silently re-seeded the whole experiment. | Private generator. |
| 7 | Evaluation returned only means over 3 episodes. | No way to tell a difference from noise. | `eval_episodes` is configurable (`--eval_episodes=`), and the result now carries per-type standard deviation, per-episode values and the seeds used. |

Checks that passed (no change needed): cash is conserved to 1e-6 and shares to zero across 600-step random-play fuzz runs in both event models with a tight provider limit; no look-ahead reaches agents (the next tape price is used only inside the price update; events, forecasts and the preference teacher use past or realised data); a market order costs exactly 6.0 bps (5 bps half-spread + 1 bp fee); with a flat tape and no events random agents lose 1.13% +/- 0.08 (costs only) and the adversary does not differ from ordinary agents (-0.03% +/- 0.08); with moving tape and no events the difference is +0.17% +/- 1.22; with events +3.4% +/- 2.1 (not significant, 10 seeds). The synthetic tape has symmetric drift and no look-ahead.

Known design choices that still affect interpretation (not changed): an episode ends when *any* agent goes bankrupt or the market collapses, so an adversary can shorten ordinary agents' episodes (`termination_reason` is logged); the adversary knows its own event's sign before it fires (the insider mechanism is the point of the study); the buy-and-hold benchmark is gross and uses the mark (which contains news and agent flow); initial policy weights are not paired across variants with different agent counts.
