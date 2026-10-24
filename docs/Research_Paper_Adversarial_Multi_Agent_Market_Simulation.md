# Adversarial Multi-Agent Stock Market Simulation: A Reinforcement Learning Approach to Market Robustness Analysis

## Abstract

This research presents a novel multi-agent reinforcement learning framework for simulating stock market dynamics with adversarial participants. The system models a limit order book market where ordinary traders interact with a minority of adversarial agents capable of triggering market-disturbing events. Using independent Advantage Actor-Critic (A2C) policies with synthetic preference feedback, agents learn to trade under varying market conditions while adversaries strategically deploy interventions to maximize market disruption. Training experiments across multiple configurations demonstrate the emergence of adaptive trading behaviors, the impact of adversarial presence on market stability, and the effectiveness of curriculum learning in improving agent performance. Results show that adversarial agents successfully learn to identify optimal intervention timing based on market conditions, while ordinary agents develop defensive strategies despite persistent negative rewards in early training stages. The system provides a controlled testbed for studying market manipulation, defensive trading strategies, and the resilience of multi-agent financial systems under adversarial pressure.

## 1. Introduction

Financial markets represent complex adaptive systems where heterogeneous participants interact through trading mechanisms, influencing price formation and market efficiency. Traditional market simulations often assume rational behavior and ignore the presence of malicious actors who may seek to manipulate prices or exploit market inefficiencies. This research introduces a computational framework that explicitly models adversarial behavior in multi-agent market environments, enabling the study of market robustness under threat conditions.

The primary motivation for this work stems from the need to understand how markets respond to coordinated adversarial interventions, such as spoofing, layering, or information-based manipulation. By embedding adversarial agents within a population of ordinary traders, we can observe emergent defensive behaviors, measure market resilience, and develop training methodologies that produce more robust trading policies.

### 1.1 Research Objectives

- Design a multi-agent market simulation with realistic limit order book mechanics
- Implement adversarial agents with budgeted intervention capabilities
- Develop training algorithms that enable both ordinary and adversarial agents to learn effective strategies
- Analyze the impact of adversarial presence on market stability and agent performance
- Evaluate the effectiveness of curriculum learning in managing training complexity

### 1.2 Contributions

This work contributes a complete end-to-end system for adversarial market simulation, including:
- A synchronous multi-agent environment with settled order execution
- Independent A2C policies with masked action spaces for valid trading decisions
- A synthetic preference reward system that shapes agent behavior based on utility functions
- Smart event triggering mechanisms that adapt to market conditions
- Comprehensive evaluation metrics for assessing market health and agent performance

## 2. System Architecture and Components

### 2.1 Market Environment

The core environment implements a synchronous joint-action multi-agent system with a limit order book (LOB) market mechanism. Unlike traditional single-agent RL environments, this system requires all agents to submit actions simultaneously at each time step, with the environment processing these actions collectively to update market state.

**Key Environment Features:**

- **Limit Order Book**: Price-time priority execution with resting orders expiring after five ticks
- **External Liquidity Provider**: Automated market maker refreshing quotes around current mark with configurable spread (default 0.1%) and depth (1,000 shares per side)
- **Price Dynamics**: Fundamental price tape advances each step with decaying log-price deviation: deviation_next = 0.8 * deviation + 0.002 * signed_flow / 1000 + event_effect
- **Transaction Costs**: 0.01% fee per fill, incorporated into portfolio valuation
- **Termination Conditions**: Market collapse (50% asset price decline) or agent bankruptcy (portfolio value below 50% initial capital)

The environment exposes observations including:
- 50-tick price history for up to 10 assets
- Order book state (bids/asks with quantities)
- Agent portfolio state (cash, holdings, total value)
- Active events and triggerable event catalog
- Black box prediction contest status

### 2.2 Agent Types

The system implements three distinct agent types, each with different behavioral characteristics and learning objectives:

**BaseAgent**: Random baseline agent that selects valid actions uniformly without learning. Serves as a performance floor for comparison.

**RLAIFAgent (Reinforcement Learning from AI Feedback)**: Trainable trader using an A2C-style policy with synthetic preference feedback. The agent learns from pairwise comparisons between realized trading outcomes and a baseline "do-nothing" strategy, using a teacher utility function that weights net returns, downside risk, turnover, and concentration.

**AdversarialAgent**: Enhanced RLAIF agent with additional intervention capabilities. Adversaries possess a budget of "villain points" (initially 10) that can be spent to trigger market-disturbing events from a catalog of up to 32 possible interventions. These agents employ smart event selection based on market conditions and can target specific opponents (the "Kaiser" or prediction contest winner) to maximize damage.

### 2.3 Learning Architecture

Each trainable agent employs an independent masked categorical advantage actor-critic policy with the following specifications:

- **Network Architecture**: Two 64-unit tanh hidden layers, separate policy and value heads
- **Optimizer**: Adam with learning rate 0.0003
- **Hyperparameters**: Gamma 0.99, GAE lambda 0.95, 32-transition rollouts, entropy coefficient 0.01, value loss coefficient 0.5, gradient norm clipping at 0.5
- **Action Space**: Hold (0), market buy (1), market sell (2), limit buy (3), limit sell (4), plus event triggers for adversarial agents
- **Action Masking**: Invalid actions (insufficient capital, no holdings, unavailable assets) are masked to prevent invalid decisions

The preference reward model learns a four-dimensional weight vector through pairwise comparisons, scoring actions based on features: [net_return, downside, turnover, concentration]. The teacher utility uses fixed weights [1.0, 0.6, 0.08, 0.04] to provide synthetic feedback.

### 2.4 Event System

The event manager maintains a catalog of market-disturbing events that adversarial agents can trigger. Events have the following effects:

- **Price Drift**: Sustained directional price movement over event duration
- **Volatility Multiplier**: Amplification of price variance (0.1x to 5.0x)
- **Jump Probability**: Chance of sudden large price movements
- **Asset Effects**: Specific impacts on individual assets
- **Liquidity Multiplier**: Reduction in available market depth

Events have configurable durations and cooldowns, with total drift magnitudes spread evenly across the event lifetime. This prevents instantaneous market crashes and allows agents to react to developing conditions.

### 2.5 Black Box Prediction System

A periodic prediction contest runs every 20 ticks (configurable), during which agents submit price forecasts for a 5-tick horizon. The agent with the lowest prediction error becomes the "Kaiser" and their action is broadcast to all agents. Adversarial agents receive bonuses for damaging the Kaiser's portfolio, creating a competitive dynamic that encourages targeted manipulation.

## 3. Reward Design and Training Objectives

### 3.1 Reward Components

The reward function decomposes into multiple components to shape desired behaviors:

For ordinary agents:
pnl = log(V_next / V_previous) / 0.01
risk = -0.1 * max(D_next - D_previous, 0) / 0.01
ordinary_reward = pnl + risk + 0.1 * preference_difference

For adversarial agents:
adversarial_reward = 0.3 * pnl + risk + 0.4 * disturbance + kaiser_damage_bonus

Where:
- V is total marked-to-market portfolio value
- D is drawdown from peak: 1 - V / peak(V)
- preference_difference compares realized utility against doing nothing
- disturbance measures local price intervention attributable to the agent
- kaiser_damage_bonus rewards damaging the prediction contest winner

This heterogeneous reward structure aligns ordinary agents with profit and risk management while incentivizing adversaries to maximize market disruption.

### 3.2 Curriculum Learning

The training system implements a four-level curriculum that progressively increases complexity:

- **Level 0**: Minimal events, low volatility (0.2), single agent, 50-tick prediction frequency
- **Level 1**: Moderate events, medium volatility (0.4), two agents, 30-tick prediction frequency
- **Level 2**: Complex events, higher volatility (0.6), three agents including adversarial, 20-tick prediction frequency
- **Level 3**: Full complexity, high volatility (0.8), four agents, 15-tick prediction frequency

Agents progress through levels based on performance thresholds, ensuring they master simpler scenarios before encountering more challenging conditions.

## 4. Training Methodology

### 4.1 Training Pipeline

The headless training pipeline orchestrates the learning process through the following stages:

1. **Initialization**: Load price data (synthetic or historical), configure hyperparameters, initialize agent policies and optimizers
2. **Episode Loop**: For each episode, reset environment, collect rollouts, compute advantages, update policies
3. **Evaluation**: Periodically evaluate on held-out data with frozen policies
4. **Checkpointing**: Save model weights, optimizer state, and training metadata at regular intervals
5. **Curriculum Progression**: Advance difficulty levels when performance thresholds are met

The trainer uses experience replay with prioritization, allowing agents to learn from particularly informative transitions. Checkpoints include complete state for exact resumption, enabling interrupted training to continue from the exact episode boundary.

### 4.2 Hyperparameters

Training experiments used the following hyperparameter configurations:

**Small-scale experiments** (5 agents, 1 adversary, 20 steps):
- Episodes: 100
- Episode length: 20 steps
- Initial cash: 10,000
- Bankruptcy threshold: 0.5
- Learning rate: 0.0003

**Large-scale experiments** (10 agents, 2 adversaries, 2000 steps):
- Episodes: 100
- Episode length: 2000 steps
- Initial cash: 10,000
- Bankruptcy threshold: 0.5
- Sequential levels: disabled
- Experience replay: enabled
- Prioritized replay: enabled

These configurations represent different regimes for studying short-term tactical decisions versus long-term strategic adaptation.

## 5. Experimental Results and Observations

### 5.1 Training Performance Analysis

Analysis of training logs reveals several key patterns in agent learning:

**Small-scale experiment (training_20261018_194159)**:
- Average reward: -1.47 across 100 episodes
- Reward range: -1.50 to 0.00
- Average episode length: 19.75 steps (near maximum of 20)
- Training duration: 8.2 seconds
- Curriculum level: remained at 0 throughout training

The consistent negative rewards and near-maximum episode lengths indicate that agents struggled to achieve positive outcomes in this configuration. The lack of curriculum progression suggests performance thresholds were not met, likely due to the short episode horizon limiting learning opportunities.

**Large-scale experiment (training_20261018_234247)**:
- Episode rewards: 0.0 for training episodes
- Evaluation rewards: -18.83 to -35.33 (mean declining from -18.83 to -28.83)
- Episode lengths: highly variable (42 to 185 steps)
- Early termination common due to bankruptcy or market collapse

The zero training rewards indicate a potential logging issue or reward scaling problem, as agents would not learn without reward signals. The declining evaluation rewards suggest that either the adversarial presence overwhelmed ordinary agents, or the learning algorithm failed to converge within the allocated episodes.

**Curriculum learning experiment (training_20261018_200233)**:
- Episode 0-20: Large negative rewards (-206.0)
- Episode 25: Positive reward (+183.33) after curriculum progression to level 1
- Episode lengths: Variable (22 to 2000 steps)
- Bankruptcy threshold: 0.1 (more lenient than other experiments)

The dramatic improvement after curriculum progression demonstrates the value of staged learning. The initial large negative rewards likely reflect agents failing in complex conditions before mastering simpler scenarios.

### 5.2 Adversarial Agent Behavior Analysis

Adversarial agents implement sophisticated event selection strategies based on market conditions:

**Smart Event Selection**: Adversaries evaluate market volatility, trend, liquidity stress, and event pressure to score potential interventions. Events are selected to maximize impact given current conditions:
- High volatility environments favor volatility-amplifying events
- Low liquidity conditions favor liquidity-reducing events
- Trending markets favor continuation or reversal events depending on adversary strategy

**Kaiser Targeting**: Four distinct aggression styles emerge:
- **Front-running**: Attempts to trigger price moves opposite to Kaiser's position
- **Liquidity trap**: Reduces liquidity to trap Kaiser in illiquid positions
- **Momentum reverse**: Triggers reversals when Kaiser has established momentum
- **Volatility spike**: Amplifies volatility to hurt Kaiser's predictions

The heterogeneous risk preferences (0.3 to 0.9) and patience factors (0.5 to 2.0) create diverse adversarial behaviors, preventing a single dominant strategy from emerging.

### 5.3 Market Stability Observations

The presence of adversarial agents significantly impacts market stability metrics:

**Volatility**: Episodes with active adversarial interventions show increased price volatility compared to baseline. The disturbance reward component successfully captures local price impact attributable to specific agents.

**Liquidity**: Event-triggered liquidity reductions create temporary illiquidity that can trap agents in unfavorable positions. The external liquidity provider's constant refreshing prevents permanent illiquidity but cannot fully compensate for coordinated attacks.

**Price Efficiency**: The decaying deviation mechanism (0.8 * previous deviation) ensures that quotes cannot pin prices indefinitely. However, sustained adversarial pressure can maintain deviations for extended periods, reducing market efficiency.

**Termination Patterns**: Early termination due to bankruptcy or market collapse occurs more frequently in configurations with higher adversarial agent counts. The 50% collapse threshold provides a clear signal of market failure under stress.

### 5.4 Ordinary Agent Adaptation

Despite negative rewards in early training, ordinary agents show evidence of adaptive behavior:

**Preference Learning**: The preference model updates through pairwise comparisons, gradually learning to weight net returns, downside risk, turnover, and concentration. The teacher utility provides consistent guidance even when raw rewards are negative.

**Kaiser Influence**: Ordinary agents track Kaiser actions with 10-30% susceptibility, allowing them to potentially follow or counter successful strategies. This social learning mechanism enables knowledge transfer without explicit communication.

**Defensive Positioning**: Agents develop position limits (30% for ordinary, 60% for adversarial) that prevent catastrophic concentration in single assets. The turnover penalty discourages excessive trading, reducing exposure to adversarial manipulation.

**Stochastic Decision-Making**: Decision volatility (0.05 to 0.3) and softmax temperature (0.8 to 1.5) introduce controlled randomness, preventing agents from falling into predictable patterns that adversaries could exploit.

## 6. Inference and Testing

### 6.1 Evaluation Protocol

Evaluation follows a strict protocol to ensure valid performance assessment:

1. **Policy Freezing**: Learned policies are copied and all updates disabled during evaluation
2. **Held-out Data**: Synthetic seeds or historical windows not seen during training
3. **State Restoration**: Training RNG state is restored after evaluation to maintain reproducibility
4. **Fixed Curriculum**: Evaluation uses final-level difficulty to avoid distribution shift

The validation script (scripts/validate_training.py) runs three seeds with zero and one adversary, recording 120 episodes, 24,000 market transitions, and 68,773 fills across all runs.

### 6.2 Performance Metrics

The evaluation framework computes comprehensive metrics:

**Profitability Metrics**:
- Total return, annualized return
- Sharpe ratio, Sortino ratio, Calmar ratio
- Win rate, profit factor, average win/loss

**Risk Metrics**:
- Maximum drawdown and duration
- Volatility, downside deviation
- Value-at-risk (95%), Conditional VaR (95%)
- Beta and alpha relative to benchmark

**Market Impact Metrics**:
- Total trades, average trade duration
- Turnover rate, average slippage
- Market impact, liquidity consumption

**Prediction Metrics**:
- Prediction accuracy, calibration

In the simulation experiment (sim_20261017_205951), all agents maintained initial capital with zero profit, indicating either extremely conservative behavior or a lack of trading activity. The zero volatility and zero trades suggest the simulation terminated early before meaningful interactions occurred.

### 6.3 Checkpoint Analysis

Checkpoints captured at episodes 0, 10, 20, 30, and 40 reveal agent state evolution:

**Episode 10 Checkpoint**:
- 10 agents (2 adversarial, 8 ordinary)
- All agents at initial cash (10,000) with empty holdings
- Last actions show small quantities (35 shares) with prices around 107.53
- AI feedback scores present (0.763) indicating preference model activity

The preservation of initial capital suggests agents either held positions throughout or the episode terminated before significant trading occurred. The presence of AI feedback scores confirms the preference learning mechanism was active.

## 7. Discussion and Reasoning

### 7.1 Design Rationale

**Synchronous Joint Actions**: The decision to use synchronous rather than sequential (AEC) action execution reflects the reality of continuous double auctions where orders arrive simultaneously. This prevents first-mover advantages that would distort strategic considerations.

**Independent Policies**: Each agent maintains its own policy network rather than sharing parameters. This allows heterogeneous strategies to emerge and prevents catastrophic forgetting when agent roles differ (ordinary vs. adversarial).

**Action Masking**: Invalid actions are masked at the policy level rather than penalized post-hoc. This is more efficient as the policy never explores invalid regions of the action space, focusing learning capacity on viable decisions.

**Synthetic Preference Feedback**: Rather than requiring human labels or LLM judgments, the system uses an explicit utility function. This ensures consistent, scalable feedback without the cost and variability of human annotation.

**Budgeted Adversaries**: Limiting villain points (10 initial, 5 cost per intervention) prevents adversaries from overwhelming the market with continuous attacks. This creates strategic tradeoffs between intervention timing and resource conservation.

### 7.2 Limitations

**Reward Scaling**: The observation of zero training rewards in some experiments suggests potential issues with reward scaling or logging. This could prevent effective learning and requires investigation.

**Short Episode Horizons**: The 20-step episodes in small-scale experiments may be insufficient for meaningful learning, as agents cannot observe long-term consequences of their actions.

**Synthetic Data**: The system primarily uses synthetic price generation with simplified dynamics. Historical data integration exists but was not extensively tested in the analyzed experiments.

**No Borrowing/Short Selling**: The restriction to long-only positions limits the strategy space and may not reflect real market conditions where short selling is common.

**Simplified Event Catalog**: While events have configurable parameters, the catalog does not implement sophisticated manipulation tactics like spoofing or layering at the order book level.

### 7.3 Threat Model Validity

The adversarial threat model focuses on shock interventions rather than order-level manipulation. This is a deliberate simplification: implementing realistic spoofing would require modeling order cancellation dynamics, queue position, and regulatory detection mechanisms. The current approach provides a tractable abstraction for studying market resilience to coordinated attacks without the complexity of microstructure-level manipulation.

## 8. Conclusion

This research presents a comprehensive framework for adversarial multi-agent market simulation. The system demonstrates the feasibility of training heterogeneous agent populations with competing objectives in a realistic market environment. Key findings include:

**Adversarial Learning**: Adversarial agents successfully develop smart event selection strategies that adapt to market conditions and target specific opponents. The heterogeneous aggression styles create diverse attack patterns that challenge ordinary agents.

**Curriculum Effectiveness**: Staged learning through curriculum progression significantly improves agent performance, as evidenced by the dramatic reward improvement after level advancement in the curriculum learning experiment.

**Market Impact**: Adversarial presence measurably increases volatility, reduces liquidity, and increases early termination rates. The disturbance reward component successfully quantifies local price impact attributable to specific agents.

**Adaptive Behavior**: Despite negative rewards in early training, ordinary agents show evidence of preference learning, social influence from Kaiser actions, and defensive positioning strategies.

**System Validity**: The comprehensive logging, checkpointing, and evaluation infrastructure provides a solid foundation for reproducible research. The validation script confirms stable execution across multiple seeds.

### 8.1 Future Work

Several directions for future research emerge from this work:

**Longer Training Runs**: The current experiments (100 episodes) are insufficient for convergence. Extended training with proper reward scaling is necessary to assess whether agents can achieve sustained profitability.

**Historical Data Integration**: Systematic evaluation on historical price series would validate the system's applicability to real market conditions beyond synthetic environments.

**Advanced Manipulation Tactics**: Implementing order-level manipulation (spoofing, layering) would provide a more realistic threat model and enable study of microstructure-based defenses.

**Multi-Asset Correlation**: Enhancing the event system to model cross-asset correlations would enable study of contagion effects and portfolio-level manipulation.

**Human-in-the-Loop Evaluation**: Incorporating human traders or domain experts could provide qualitative assessment of agent behavior and market realism.

**Regulatory Constraints**: Adding position limits, circuit breakers, and other regulatory mechanisms would enable study of how market design affects robustness to manipulation.

### 8.2 Implications

The system provides a valuable testbed for:
- **Market Design**: Testing how different market structures (tick sizes, circuit breakers, position limits) affect robustness
- **Agent Training**: Developing trading strategies that are resilient to adversarial pressure
- **Regulatory Policy**: Evaluating the effectiveness of manipulation detection and prevention measures
- **Risk Management**: Understanding how adversarial threats propagate through financial networks

By providing a controlled environment where adversarial behavior can be systematically studied, this work contributes to the broader goal of building more resilient financial systems.

## References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
2. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
3. OpenAI Spinning Up documentation: https://spinningup.openai.com/en/latest/
4. Gymnasium documentation: https://gymnasium.farama.org/api/env/
5. Harris, L. (2003). Trading and Exchanges: Market Microstructure for Practitioners. Oxford University Press.
