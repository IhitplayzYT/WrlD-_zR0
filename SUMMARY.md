# Synthetic RL markets with an adversarial minority

This file describes the corrected implementation. Earlier architecture notes and saved research drafts describe several features that were not actually connected to training; do not use them as evidence of implemented PPO, SAC, spoofing, or a learned world simulator.

## Problems corrected

- The logger deadlocked on its 100th buffered message: `log()` held a non-reentrant lock and called `flush()`, which acquired the same lock. It now uses a reentrant lock, with a subprocess timeout regression test.
- Ordinary agents used a heuristic with default traits that mostly selected hold. No ordinary policy was trained. Agent objects were also recreated on reset and between episodes.
- The environment used cumulative profit and arbitrary inactivity penalties, the trainer recomputed a different reward, and the adversary reshaped it a third time. Returned step information was empty. All consumers now use the same environment reward.
- Market orders changed portfolios without settling the matching counterparty; limit fills were never posted to portfolios. Resting orders could overspend the same funds. Portfolio marks were stale after prices moved.
- Resting quotes could freeze the historical tape. Zero padding generated artificial enormous returns. Historical augmentation shuffled price levels and compounded noise across episodes.
- The trainer called nonexistent scheduler/evaluation methods, swallowed exceptions, did not consistently advance its episode counter, and often saved no policies. Evaluation trained fresh agents instead of freezing the learned policies.
- Event lifetimes depended on wall-clock seconds. Forecast contests were not activated correctly and evaluated the end of the input tape instead of the forecast's realized future window. The leaderboard accessed a nonexistent field, and robust aggregation had a broadcasting error and used oracle adversary identities.
- Headless training imported the GUI and downloaded data that it then ignored. Supplied stock data now reaches the trainer; no download is required.

## Actual learning algorithm and population

Each ordinary and adversarial agent has an independent masked categorical advantage actor-critic (A2C-style) policy. There are two 64-unit tanh hidden layers, a policy head and a scalar value head. Adam uses learning rate 0.0003, gamma 0.99, GAE lambda 0.95, 32-transition rollouts, entropy coefficient 0.01, value loss coefficient 0.5, and gradient norm clipping at 0.5. Time-limit truncation bootstraps; market collapse/bankruptcy termination does not. This implementation does **not** claim PPO or SAC.

Inputs include past returns/volatility, current portfolio weights/cash, asset availability and event availability. No future prices or synthetic regime labels are supplied. Actions are hold and market/limit buys/sells for each available asset; invalid actions are masked. Size is bounded by available capital, inventory, a 10% per-decision budget and the agent's per-asset position limit (30% ordinary, 60% adversarial at submission). Price movement can subsequently move weights past those targets. No borrowing or short selling is allowed.

Adversaries have the same trading actions plus a fixed catalogue of up to 32 simulated event interventions. They start each episode with 10 intervention points, spending 5 only after a successful intervention. Roles remain fixed. The cap is floor(20% of the population), so fewer than five agents implies no adversary. This is a stylized shock-intervention threat model; it does not implement exchange-level spoofing/layering or claim that traders can cause real natural disasters.

Models and optimizers survive episode resets. Checkpoints contain weights, optimizer state, preference weights, RNG states, counters and curriculum state. A JSON manifest references a PyTorch payload. Old JSON checkpoints without trainable policies cannot be resumed. Exact resume is tested at completed episode boundaries on the same software stack; an interrupted partial episode restarts that episode with the saved weights. Load only your own trusted checkpoints.

## Reward definition

For every agent, let V be total marked-to-market cash plus inventory and D = 1 - V / peak(V). Define:

```
pnl = log(V_next / V_previous) / 0.01
risk = -0.1 * max(D_next - D_previous, 0) / 0.01
ordinary_reward = pnl + risk + 0.1 * preference_difference
adversarial_reward = 0.3 * pnl + risk + 0.4 * disturbance
```

Trading fees and spread losses are already in V; they are not deducted twice. `preference_difference` compares a learned pairwise utility score for the realized decision against keeping the previous holdings without trading. The teacher uses the explicitly defined net-return/downside/turnover/concentration utility in `PreferenceReward`; this is **synthetic preference feedback, not an LLM judge or human-labelled RLAIF dataset**. The preference model updates after scoring a transition and is frozen during evaluation. Adversaries receive no ordinary preference term.

`disturbance` is a bounded, per-asset average of the absolute local price intervention attributable to that agent: its pre-repricing net executed flow plus the current effects of events it successfully triggered. It is normalized by a 1% return unit. Unrelated market volatility, action entropy, unfilled submissions and expired events earn no disturbance bonus. This is a simulator-defined local disturbance proxy, not a measured causal harm to other traders or a full counterfactual-market evaluation.

Forecast contests are diagnostics, scored only after the actual forecast horizon elapses. Forecast reward weight is zero because the current trend forecaster is a baseline, not a trained forecast policy. Arbitrary hold, sideways-market and collapse bonuses are removed. A cash-only agent with no intervention receives zero; losing trades remain negative. Reward components, portfolio returns, fees, fills, drawdowns and objectives by agent type are logged separately. The sum of opponents' rewards is not a welfare or profitability measure.

## Market and data assumptions

The limit order book uses price/time priority and the resting order's price. Market orders are immediate-or-cancel; limit orders reserve capital/inventory and expire after five ticks. Every matched trade updates both counterparties. A clearly modelled external liquidity provider refreshes quotes around the current mark (default spread 0.1%, depth 1,000 shares per side per asset); its cash and inventory ledger and the fee account allow conservation checks. The provider has unlimited credit/inventory capacity. Default agent fee is 0.01% per fill.

A seeded exogenous price tape always advances. The mark has a decaying log-price deviation: `deviation_next = 0.8 * deviation + 0.002 * signed_flow / 1000 + event_effect`. Quotes cannot pin the mark forever. Prices are updated before all portfolios and rewards. Event duration/cooldown uses simulation ticks; default event price drifts are interpreted as total-event magnitudes spread over duration. The core applies event drift, volatility, jumps and asset effects. Other catalogue fields (sector links, liquidity multipliers, correlations, news sentiment) are not active market mechanisms and should not be claimed as experimental features.

Synthetic tapes have symmetric bull/bear drifts, clustered volatility, correlated asset shocks and explicit seeds. Their parameters are illustrative, not calibrated to an exchange. Supplied historical prices require finite positive values and enough rows for a chronological 80/20 train/evaluation split. Each episode needs steps + 1 prices. CSV/Excel files need an explicit price/close column; timestamps, when present, are sorted and duplicates rejected. Multi-asset input files must be row-aligned by the user. No silent replacement of invalid files with synthetic data is performed.

The environment exposes a custom synchronous joint-action API: dict observations/rewards with scalar episode flags. It is not a PettingZoo AEC or single-agent Gym environment. The episode ends on insolvency or a 50% asset-price collapse; the configured time horizon truncates it. A 10% price rise no longer ends it.

## Evaluation and research use

Evaluation copies learned policies, freezes policy/preference updates, uses held-out synthetic seeds or historical windows, and restores the training RNG state afterwards. Curriculum evaluation uses fixed final-level difficulty so changing training levels does not change the evaluation distribution. Cash return is zero. The logged buy-and-hold reference is a **gross, uncosted primary-asset return**, not a like-for-like capital-constrained trading strategy.

`python -m scripts.validate_training` runs three seeds with zero and one adversary. The saved `docs/validation_results.json` records a short engineering validation: 120 episodes, 24,000 market transitions, and 68,773 fills. All runs completed; every policy made 140 updates. Ordinary-agent held-out returns improved in these short runs, but some remained negative or reached cash-like zero returns. Exploratory training rewards stayed negative in the longer validation. This is evidence of functioning learning and stable execution, not proof of profitability, convergence, robustness or statistical significance.

For a paper, run more seeds and longer training, reserve an untouched final test set, compare against cash/random/passive risk-matched traders with the same execution costs, report per-role net returns and maximum drawdown, and vary minority size, event budget, costs and liquidity. Include uncertainty across seeds and paired no-adversary controls. Do not select only positive runs or interpret a rising adversary reward as improved market health.

The headless trainer is the research entry point. The main GUI now reads the same environment ledger and realized prediction windows; experimental population/resource-editing controls are disabled in that live view. Direct execution of the standalone GUI retains a clearly labelled illustrative demo. Auxiliary world-model/temporal-head modules are not the transition dynamics used by the research trainer.

The validation environment used Python 3.14, NumPy 2.4.6, PyTorch 2.13.0, Gymnasium 1.3.0, Polars 1.33.1 and sortedcontainers 2.4.0. The GUI was connected to the live environment and syntax-checked, but not visually tested in this session.

Algorithm references: [policy-gradient actor-critic overview](https://spinningup.openai.com/en/latest/algorithms/vpg.html), [Gymnasium termination versus truncation](https://gymnasium.farama.org/api/env/).
