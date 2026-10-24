"""Synchronous multi-agent market with settled orders and one reward per transition.

This is a custom joint-action API (dict observations/rewards, scalar episode
flags), not a PettingZoo AEC environment or a single-agent Gym environment.
"""
from dataclasses import asdict
import random
import numpy as np
from gymnasium import spaces
from src.utils.logging import LogCategory, get_logger
from src.market import OrderSide, MultiAssetOrderBook, MultiStockEnvironment
from src.events import EventManager
from src.agents import BaseAgent, RLAIFAgent, AdversarialAgent
from src.agents.adversarial_agent import MAX_TRIGGERABLE_EVENTS
from src.environment.observation_space import EnhancedObservationSpace
from src.simulation.black_box import BlackBoxPredictionSystem
from src.simulation.time_manager import TimeManager
from src.models.incremental_reward import IncrementalRewardCalculator
from src.models.preference_reward import PreferenceReward


class MarketEnvironment:
    metadata = {'name': 'WrlD_Zr0', 'render_modes': ['human']}

    def __init__(self, n_agents=5, max_bad_agents=1, initial_cash=10000., episode_length=200,
                 price_data=None, black_box_duration=5, black_box_frequency=20,
                 event_manager=None, use_enhanced_observation=False, agent_types=None,
                 logger=None, fee_rate=.0001, spread=.001, liquidity=1000,
                 impact_coefficient=.002, seed=None):
        if n_agents < 1 or episode_length < 1 or initial_cash <= 0:
            raise ValueError('Positive agents, episode length and cash required')
        if fee_rate < 0 or not 0 <= spread < 1 or liquidity < 1 or impact_coefficient < 0:
            raise ValueError('Invalid market execution parameters')
        if black_box_duration < 1 or black_box_frequency < 1:
            raise ValueError('Prediction period and frequency must be positive')
        self.logger = logger or get_logger()
        self.n_agents, self.initial_cash, self.episode_length = n_agents, initial_cash, episode_length
        self.max_bad_agents = min(max(0, max_bad_agents), n_agents // 5)
        self.agent_types = list(agent_types) if agent_types is not None else ['rlaif'] * n_agents
        if len(self.agent_types) != n_agents or any(t not in ('rlaif', 'adversarial', 'base') for t in self.agent_types):
            raise ValueError('Supply exactly one valid agent type per agent')
        if self.agent_types.count('adversarial') > self.max_bad_agents:
            raise ValueError('Adversarial population exceeds the configured cap (at most 20%)')
        self.rng = np.random.default_rng(seed)
        self.price_data = price_data if price_data is not None else {'ASSET0': self._generate_synthetic_prices(episode_length+1)}
        self._validate_prices()
        self.available_assets = list(self.price_data)
        self.order_book = MultiAssetOrderBook(self.available_assets)
        self.multi_stock_env = MultiStockEnvironment(self.available_assets, self.price_data)
        self.event_manager = event_manager if event_manager is not None else EventManager()
        self.black_box_duration, self.black_box_frequency = black_box_duration, black_box_frequency
        self.black_box_system = BlackBoxPredictionSystem(duration=black_box_duration, frequency=black_box_frequency, logger=self.logger)
        self.time_manager = TimeManager(enable_market_sessions=False, enable_dynamic_adjustment=False, logger=self.logger)
        self.use_enhanced_observation = use_enhanced_observation
        # Untrained random recurrent features are not fed to the trading policies.
        self.observation_space_enhanced = EnhancedObservationSpace(use_temporal_features=False) if use_enhanced_observation else None
        self.agents = [f'agent_{i}' for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        self.agent_objects = {}
        self.agent_states = {}
        for aid, kind in zip(self.agents, self.agent_types):
            cls = {'rlaif': RLAIFAgent, 'adversarial': AdversarialAgent, 'base': BaseAgent}[kind]
            kwargs = {} if kind == 'base' else {'logger': self.logger}
            self.agent_objects[aid] = cls(aid, initial_cash=initial_cash, **kwargs)
        self.fee_rate, self.spread, self.liquidity = fee_rate, spread, liquidity
        self.impact_coefficient = impact_coefficient
        self.bankruptcy_threshold = .5
        self.reward_calculator = IncrementalRewardCalculator(forecast_weight=0., preference_weight=.1, drawdown_penalty=.1)
        self.action_space = spaces.Box(np.array([0, 0, 0, 0], dtype=np.float32),
                                       np.array([4, len(self.available_assets)-1, 1000, np.inf], dtype=np.float32))
        self.render_mode = None
        # Passive company event parameters (MDP-like transitions)
        self.company_event_frequency = 20  # Every ~20 days
        self.company_event_counter = 0
        self.company_event_state = 'neutral'  # MDP state: neutral, bullish, bearish
        self.company_state_transition_prob = {
            'neutral': {'neutral': 0.6, 'bullish': 0.2, 'bearish': 0.2},
            'bullish': {'neutral': 0.3, 'bullish': 0.5, 'bearish': 0.2},
            'bearish': {'neutral': 0.3, 'bullish': 0.2, 'bearish': 0.5}
        }
        self.company_event_impact = {
            'neutral': 0.0,
            'bullish': 0.03,  # +3% price impact
            'bearish': -0.03  # -3% price impact
        }
        self.reset(seed=seed)

    def _validate_prices(self):
        if not self.price_data or len(self.price_data) > 10:
            raise ValueError('Supply 1 to 10 price series')
        for asset, prices in self.price_data.items():
            a = np.asarray(prices, dtype=float)
            if a.ndim != 1 or len(a) < self.episode_length + 1 or not np.isfinite(a).all() or np.any(a <= 0):
                raise ValueError(f'{asset}: need at least episode_length+1 finite positive prices')
            self.price_data[asset] = a.tolist()

    def _generate_synthetic_prices(self, n_steps):
        return (100 * np.exp(np.r_[0., np.cumsum(self.rng.normal(-.5e-4, .01, n_steps-1))])).tolist()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            random.seed(seed)
        self._validate_prices()
        self.current_step = self.current_day = self.current_movement = 0
        self.movements_per_day = 10
        self.agent_selection = self.agents[0]
        self.current_prices = {a: p[0] for a, p in self.price_data.items()}
        self.price_history = {a: [p] for a, p in self.current_prices.items()}
        self.market_cap = {a: p * 10000 for a, p in self.current_prices.items()}
        self.economy_collapsed = self.is_black_box = False
        self._ended = False
        self._prediction_actual = []
        self._prediction_origin = 0.
        self._deviations = {a: 0. for a in self.available_assets}
        self._event_owners = {}
        self._order_steps = {}
        self._trade_cursor = {a: 0 for a in self.available_assets}
        self.lp_cash = 0.
        self.lp_holdings = {a: 0 for a in self.available_assets}
        self.fees_collected = 0.
        self.order_book.reset()
        self.multi_stock_env.stock_data = self.price_data
        self.multi_stock_env.reset()
        self.event_manager.reset()
        self.black_box_system.reset()
        self.time_manager.reset()
        self.time_manager.start_simulation()
        if self.observation_space_enhanced:
            self.observation_space_enhanced.reset()
        for aid, agent in self.agent_objects.items():
            agent.reset(self.initial_cash)
            self.agent_states[aid] = agent.state  # One ledger, never a stale duplicate.
            if isinstance(agent, AdversarialAgent):
                agent.bind_event_manager(self.event_manager)
        self._turnover = {aid: 0. for aid in self.agents}
        self._flow = {aid: {a: 0 for a in self.available_assets} for aid in self.agents}
        self._refresh_liquidity()
        return self._get_observations(), {a: {} for a in self.agents}

    def _reserved(self, aid):
        cash, holdings = 0., {a: 0 for a in self.available_assets}
        for asset, book in self.order_book.order_books.items():
            for order in book.order_map.values():
                if order.agent_id == aid:
                    if order.side == OrderSide.BUY:
                        cash += order.price * order.qty * (1 + self.fee_rate)
                    else:
                        holdings[asset] += order.qty
        return cash, holdings

    def _settle(self, asset):
        book = self.order_book.get_order_book(asset)
        for trade in book.trades[self._trade_cursor[asset]:]:
            cost = trade.price * trade.qty
            for aid, direction in ((trade.buyer, 1), (trade.seller, -1)):
                if aid == '__liquidity__':
                    self.lp_cash -= direction * cost
                    self.lp_holdings[asset] += direction * trade.qty
                else:
                    state = self.agent_states[aid]
                    state.cash -= direction * cost + cost * self.fee_rate
                    state.holdings[asset] = state.holdings.get(asset, 0) + direction * trade.qty
                    self.fees_collected += cost * self.fee_rate
                    self._turnover[aid] += cost
                    self._flow[aid][asset] += direction * trade.qty
                    if state.cash < -1e-6 or state.holdings[asset] < 0:
                        raise RuntimeError('Settlement violated cash/inventory constraints')
                    state.cash = max(0., state.cash)
        self._trade_cursor[asset] = len(book.trades)

    def _refresh_liquidity(self):
        for asset, book in self.order_book.order_books.items():
            for oid, order in list(book.order_map.items()):
                if order.agent_id == '__liquidity__' or self.current_step - self._order_steps.get((asset, oid), 0) >= 5:
                    book.cancel_order(oid)
                    self._order_steps.pop((asset, oid), None)
            p = self.current_prices[asset]
            bid = p * (1-self.spread/2) if self.spread else np.nextafter(p, -np.inf)
            ask = p * (1+self.spread/2) if self.spread else np.nextafter(p, np.inf)
            book.add_order('__liquidity__', OrderSide.BUY, bid, self.liquidity)
            self._settle(asset)
            book.add_order('__liquidity__', OrderSide.SELL, ask, self.liquidity)
            self._settle(asset)
        self._order_steps = {key: step for key, step in self._order_steps.items()
                             if key[1] in self.order_book.get_order_book(key[0]).order_map}

    def _execute_action(self, aid, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('Action must contain four finite numbers')
        if not np.equal(action[:3], np.floor(action[:3])).all():
            raise ValueError('Action type, asset and quantity must be integers')
        kind, idx, quantity = map(int, action[:3])
        if kind not in range(5) or idx not in range(len(self.available_assets)) or quantity < 0 or quantity > 1000:
            raise ValueError('Invalid action type, asset index or quantity')
        if kind == 0 or quantity == 0:
            return
        asset = self.available_assets[idx]
        state = self.agent_states[aid]
        book = self.order_book.get_order_book(asset)
        # Replacing the agent's old quote prevents repeated unbounded reservations.
        for oid, order in list(book.order_map.items()):
            if order.agent_id == aid:
                book.cancel_order(oid)
        buying = kind in (1, 3)
        quote = book.get_best_ask() if buying else book.get_best_bid()
        price = quote[0] if kind in (1, 2) and quote else float(action[3])
        if price <= 0:
            return
        reserved_cash, reserved_holdings = self._reserved(aid)
        if buying:
            quantity = min(quantity, int(max(0., state.cash-reserved_cash) / (price * (1+self.fee_rate))))
        else:
            quantity = min(quantity, state.holdings.get(asset, 0)-reserved_holdings[asset])
        if quantity <= 0:
            return
        oid = book.add_order(aid, OrderSide.BUY if buying else OrderSide.SELL, price, quantity)
        self._order_steps[asset, oid] = self.current_step
        self._settle(asset)
        if kind in (1, 2):  # Market orders are IOC: never leave phantom resting orders.
            book.cancel_order(oid)

    def _market_state_for_events(self):
        p = np.asarray(self.price_history[self.available_assets[0]][-20:])
        return dict(step=self.current_step, current_price=float(p[-1]),
                    market_cap=float(p[-1])*10000,
                    volatility=float(np.std(np.diff(np.log(p)))) if len(p)>1 else .01)

    def _apply_adversarial_event_trigger(self, aid, agent):
        if not isinstance(agent, AdversarialAgent):
            return
        eid = agent.consume_pending_event()
        if eid is None or agent.state.villain_points < agent.event_trigger_cost:
            return
        if self.event_manager.trigger_event(eid, self._market_state_for_events()):
            agent.state.villain_points -= agent.event_trigger_cost
            event = self.event_manager.get_event(eid)
            self._event_owners[eid] = (aid, event.start_time)
            agent.record_event_trigger(eid, event.name, 0.)

    def _update_market(self):
        # Fundamental prices always advance. Quotes cannot freeze the tape.
        active = self.event_manager.get_active_events()
        damage = {aid: 0. for aid in self.agents}
        
        # Handle passive company events (MDP-like transitions)
        company_shift = self._apply_company_event()
        
        for asset in self.available_assets:
            t = self.current_step
            base_return = np.log(self.price_data[asset][t+1] / self.price_data[asset][t])
            disturbances = {}
            event_shift = 0.
            for event in active:
                e = event.effects
                # Default event drifts are TOTAL event magnitudes, spread over duration.
                contribution = np.log1p(max(-.95, e.price_drift)) / max(1, event.duration)
                contribution += base_return * (np.clip(e.volatility_multiplier, .1, 5.) - 1)
                if self.rng.random() < e.jump_probability / max(1, event.duration):
                    contribution += self.rng.normal(e.jump_mean, max(0., e.jump_std))
                contribution += float(e.asset_effects.get(asset, 0.)) / max(1, event.duration)
                event_shift += contribution
                owner = self._event_owners.get(event.id)
                if owner and owner[1] == event.start_time:
                    disturbances[owner[0]] = disturbances.get(owner[0], 0.) + contribution
            flow = sum(self._flow[aid][asset] for aid in self.agents)
            impact = self.impact_coefficient * flow / self.liquidity
            deviation = .8*self._deviations[asset] + impact + event_shift + company_shift
            self._deviations[asset] = deviation
            self.current_prices[asset] = float(self.price_data[asset][t+1] * np.exp(np.clip(deviation, -5, 5)))
            for aid in self.agents:
                # Local, same-shock price intervention proxy, not global volatility
                # or an entropy bonus merely for submitting random actions.
                own = self.impact_coefficient*self._flow[aid][asset]/self.liquidity + disturbances.get(aid, 0.)
                damage[aid] += min(1., abs(own)/.01)/len(self.available_assets)
        return damage
    
    def _apply_company_event(self):
        """Apply passive company events with MDP-like state transitions every ~20 days"""
        self.company_event_counter += 1
        
        # Check if we should trigger a company event (every ~20 days = 200 movements)
        if self.company_event_counter >= self.company_event_frequency * self.movements_per_day:
            self.company_event_counter = 0
            
            # MDP state transition based on current state
            current_state = self.company_event_state
            transition_probs = self.company_state_transition_prob[current_state]
            
            # Sample next state
            rand = self.rng.random()
            cumulative = 0.0
            next_state = current_state
            for state, prob in transition_probs.items():
                cumulative += prob
                if rand < cumulative:
                    next_state = state
                    break
            
            self.company_event_state = next_state
            
            # Apply price impact based on new state
            impact = self.company_event_impact[next_state]
            
            # Add some randomness to the impact
            impact += self.rng.normal(0, 0.01)  # ±1% additional variance
            
            return np.log1p(impact)
        
        return 0.0

    def _prediction_tick(self):
        if self.is_black_box:
            self._prediction_actual.append(self.current_prices[self.available_assets[0]])
            self.black_box_system.update()
            if not self.black_box_system.is_active:
                self.black_box_system.evaluate_predictions(self._prediction_actual)
                # Notify all agents about the new Kaiser (bbox winner)
                kaiser_id = self.black_box_system.curr_king
                if kaiser_id and kaiser_id in self.agent_objects:
                    kaiser_action = self.agent_states[kaiser_id].last_action
                    kaiser_state = {**self.agent_states[kaiser_id].__dict__, 'current_prices': self.current_prices}
                    for aid, agent in self.agent_objects.items():
                        if aid != kaiser_id:
                            if hasattr(agent, 'observe_kaiser_action'):
                                agent.observe_kaiser_action(kaiser_action)
                            if hasattr(agent, 'observe_kaiser_info'):
                                agent.observe_kaiser_info(kaiser_id, kaiser_state)
                self.is_black_box = False
        elif self.current_step % self.black_box_frequency == 0:
            self.black_box_system.is_active = self.is_black_box = True
            self.black_box_system.rem_dur = self.black_box_duration
            self.black_box_system.curr_pred.clear()
            self._prediction_actual = []
            for aid, agent in self.agent_objects.items():
                if hasattr(agent, 'generate_prediction'):
                    prediction = agent.generate_prediction(self._observe(aid), self.black_box_duration)
                    self.black_box_system.submit_prediction(aid, prediction)

    def step(self, action=None):
        if self._ended:
            raise RuntimeError('Episode ended; call reset before step')
        if not isinstance(action, dict):
            raise TypeError('Use a joint action dict with one action per agent')
        unknown = set(action) - set(self.agents)
        if unknown:
            raise ValueError(f'Unknown agents: {unknown}')
        old_values = {a: s.total_value for a, s in self.agent_states.items()}
        old_dd = {a: 1-s.total_value/max(s.peak, s.total_value) for a, s in self.agent_states.items()}
        old_holdings = {a: s.holdings.copy() for a, s in self.agent_states.items()}
        old_prices = self.current_prices.copy()
        self._turnover = {aid: 0. for aid in self.agents}
        self._flow = {aid: {a: 0 for a in self.available_assets} for aid in self.agents}
        self.event_manager.update(self._market_state_for_events())
        for aid in self.rng.permutation(self.agents):
            state = self.agent_states[aid]
            if state.is_blocked:
                state.block_remaining -= 1
                state.is_blocked = state.block_remaining > 0
                continue
            if not self.time_manager.can_agent_trade(aid):
                continue
            self._execute_action(aid, action.get(aid, np.zeros(4)))
            self._apply_adversarial_event_trigger(aid, self.agent_objects[aid])
        damage = self._update_market()
        self.current_step += 1
        # Resting limit orders can fill against refreshed external quotes.
        self._refresh_liquidity()
        for asset, price in self.current_prices.items():
            self.price_history[asset].append(price)
            self.market_cap[asset] = price*10000
        self.multi_stock_env.update_prices(self.current_prices)
        for state in self.agent_states.values():
            state.update_portfolio_value(self.current_prices, self.initial_cash)
        self.time_manager.advance_movement()
        self.current_movement, self.current_day = self.time_manager.curr_movement, self.time_manager.curr_day
        self.economy_collapsed = any(self.current_prices[a] < self.price_data[a][0]*.5 for a in self.available_assets)
        terminated = self.economy_collapsed or any(s.total_value < self.initial_cash*self.bankruptcy_threshold for s in self.agent_states.values())
        truncated = self.current_step >= self.episode_length and not terminated
        reason = 'market_collapse' if self.economy_collapsed else 'bankruptcy' if terminated else 'time_limit' if truncated else None
        self._prediction_tick()
        rewards, infos = {}, {}
        
        # Calculate Kaiser damage bonuses for adversarial agents
        kaiser_id = self.black_box_system.curr_king
        kaiser_damage_bonuses = {}
        if kaiser_id and kaiser_id in old_values:
            kaiser_old = old_values[kaiser_id]
            kaiser_new = self.agent_states[kaiser_id].total_value
            for aid, agent in self.agent_objects.items():
                if isinstance(agent, AdversarialAgent) and aid != kaiser_id:
                    if hasattr(agent, 'calculate_kaiser_damage_bonus'):
                        my_old = old_values[aid]
                        my_new = self.agent_states[aid].total_value
                        bonus = agent.calculate_kaiser_damage_bonus(kaiser_old, kaiser_new, my_old, my_new)
                        kaiser_damage_bonuses[aid] = bonus
        
        for aid, state in self.agent_states.items():
            agent = self.agent_objects[aid]
            wealth = old_values[aid]
            pnl = np.log(state.total_value/wealth)/.01
            # Compare realized net P&L against doing nothing with the old holdings.
            no_trade = sum(old_holdings[aid].get(a, 0)*(self.current_prices[a]-old_prices[a]) for a in self.available_assets)/wealth/.01
            weights = np.array([state.holdings.get(a, 0)*self.current_prices[a]/state.total_value for a in self.available_assets])
            baseline_wealth = wealth * (1 + no_trade * .01)
            baseline_weights = np.array([old_holdings[aid].get(a, 0)*self.current_prices[a]/baseline_wealth for a in self.available_assets])
            # Both alternatives use the preference teacher;
            # turnover counts fills, not passive weight changes from price moves.
            net = (state.total_value-wealth)/wealth/.01
            features = np.array([net, -max(0., -net), -self._turnover[aid]/wealth, -weights@weights])
            baseline = np.array([no_trade, -max(0., -no_trade), 0., -baseline_weights@baseline_weights])
            pref = 0.
            if isinstance(agent, RLAIFAgent) and not isinstance(agent, AdversarialAgent):
                pref = agent.preference_model.score(features) - agent.preference_model.score(baseline)
                if agent.training_enabled and not np.allclose(features, baseline):
                    better = PreferenceReward.teacher_utility(features) > PreferenceReward.teacher_utility(baseline)
                    agent.preference_model.fit_pairs(features if better else baseline, baseline if better else features)
            
            # Get agent characteristics for heterogeneous rewards
            agent_chars = {}
            if hasattr(agent, 'risk_preference'):
                agent_chars['risk_preference'] = agent.risk_preference
            if hasattr(agent, 'aggression_style'):
                agent_chars['aggression_style'] = agent.aggression_style
            
            kaiser_bonus = kaiser_damage_bonuses.get(aid, 0.0)
            reward, comp = self.reward_calculator.calculate_step_reward(
                wealth, state.total_value, old_dd[aid], 1-state.total_value/state.peak,
                preference_score=pref, is_adversarial=isinstance(agent, AdversarialAgent), 
                market_damage=damage[aid], kaiser_damage_bonus=kaiser_bonus,
                agent_characteristics=agent_chars)
            rewards[aid] = float(reward)
            infos[aid] = dict(reward_components=asdict(comp), market_damage=damage[aid],
                              profit_delta=(state.total_value-wealth)/wealth, total_value=state.total_value,
                              termination_reason=reason, policy_updates=getattr(getattr(agent, 'policy', None), 'updates', 0))
            agent.update_state(float(reward), terminated or truncated, infos[aid])
        observations = self._get_observations()
        for aid, agent in self.agent_objects.items():
            if hasattr(agent, 'learn'):
                agent.learn(rewards[aid], observations[aid], terminated, truncated)
        self._ended = terminated or truncated
        return observations, rewards, terminated, truncated, infos
    def _observe(self, agent_id: str) -> Dict:
        agent_state = self.agent_states[agent_id]
        price_hist_matrix = np.zeros((50, 10), dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            asset_hist = self.price_history[asset][-50:]
            if len(asset_hist) < 50:
                asset_hist = [asset_hist[0]] * (50 - len(asset_hist)) + asset_hist
            price_hist_matrix[:, i] = np.array(asset_hist, dtype=np.float32)
        bids_matrix = np.zeros((10, 10, 2), dtype=np.float32)
        asks_matrix = np.zeros((10, 10, 2), dtype=np.float32)
        
        for i, asset in enumerate(self.available_assets[:10]):
            order_book_state = self.order_book.get_order_book_state(asset)
            bids_list = order_book_state.get('bids', [])[:10]
            asks_list = order_book_state.get('asks', [])[:10]
            
            if len(bids_list) > 0:
                bids = np.array(bids_list, dtype=np.float32)
                if bids.ndim == 1:
                    bids = bids.reshape(-1, 1)
                    bids = np.hstack([bids, np.zeros((bids.shape[0], 1), dtype=np.float32)])
                for j, bid in enumerate(bids[:10]):
                    bids_matrix[i, j] = bid
            
            if len(asks_list) > 0:
                asks = np.array(asks_list, dtype=np.float32)
                if asks.ndim == 1:
                    asks = asks.reshape(-1, 1)
                    asks = np.hstack([asks, np.zeros((asks.shape[0], 1), dtype=np.float32)])
                for j, ask in enumerate(asks[:10]):
                    asks_matrix[i, j] = ask
        
        holdings_array = np.zeros(10, dtype=np.int32)
        for i, asset in enumerate(self.available_assets[:10]):
            holdings_array[i] = agent_state.holdings.get(asset, 0)
        
        prices_array = np.zeros(10, dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            prices_array[i] = self.current_prices.get(asset, 100.0)
        
        market_cap_array = np.zeros(10, dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            market_cap_array[i] = self.market_cap.get(asset, 1000000.0)
        
        available_assets_array = np.zeros(10, dtype=np.float32)
        for i, asset in enumerate(self.available_assets[:10]):
            available_assets_array[i] = 1.0
        
        active_events = self.event_manager.get_active_events()
        event_vector = np.zeros(10, dtype=np.float32)
        for i, event in enumerate(active_events[:10]):
            event_vector[i] = 1.0
        market_state = self._market_state_for_events()
        triggerable_ids, triggerable_mask = self.event_manager.agent_trigger_catalog(market_state, MAX_TRIGGERABLE_EVENTS)
        padded_ids = triggerable_ids + [-1] * (MAX_TRIGGERABLE_EVENTS - len(triggerable_ids))
        base_observation = {'price_history': price_hist_matrix, 'order_book_bids': bids_matrix, 'order_book_asks': asks_matrix, 'agent_cash': np.array(agent_state.cash, dtype=np.float32), 'agent_holdings': holdings_array, 'agent_total_value': np.array(agent_state.total_value, dtype=np.float32), 'is_black_box': 1 if self.is_black_box else 0, 'current_price': prices_array, 'market_cap': market_cap_array, 'active_events': event_vector, 'available_assets': available_assets_array, 'asset_names': self.available_assets[:10], 'triggerable_event_ids': padded_ids[:MAX_TRIGGERABLE_EVENTS], 'triggerable_event_mask': triggerable_mask}

        reserved_cash, reserved_holdings = self._reserved(agent_id)
        base_observation['available_cash'] = max(0., agent_state.cash-reserved_cash)
        base_observation['available_holdings'] = np.array([
            max(0, agent_state.holdings.get(a, 0)-reserved_holdings[a]) for a in self.available_assets
        ] + [0]*(10-len(self.available_assets)), dtype=np.int32)
        base_observation['villain_points'] = agent_state.villain_points
        base_observation['can_trade'] = not agent_state.is_blocked and self.time_manager.can_agent_trade(agent_id)

        if self.use_enhanced_observation and self.observation_space_enhanced:
            primary_asset = self.available_assets[0]
            agent_state_dict = {'cash': agent_state.cash, 'holdings': agent_state.holdings, 'total_value': agent_state.total_value}
            primary_order_book_state = self.order_book.get_order_book_state(primary_asset)
            best_bid = primary_order_book_state.get('best_bid')
            best_ask = primary_order_book_state.get('best_ask')
            market_state_dict = {
                'current_price': self.current_prices[primary_asset],
                'best_bid': best_bid[0] if best_bid else self.current_prices[primary_asset] * 0.99,
                'best_ask': best_ask[0] if best_ask else self.current_prices[primary_asset] * 1.01,
                'spread': primary_order_book_state.get('spread', 0.02)
            }
            enhanced_obs = self.observation_space_enhanced.create_observation(base_observation, agent_state_dict, market_state_dict)
            base_observation.update(enhanced_obs)
        return base_observation
    
    def _get_observations(self) -> Dict:
        return {agent: self._observe(agent) for agent in self.agents}
    
    def render(self, mode: str = "human"):
        if mode == "human":
            print(f"Step: {self.current_step}, Day: {self.current_day}")
            print(f"Available Assets: {self.available_assets}")
            print(f"Current Prices: {self.current_prices}")
            print(f"Market Caps: {self.market_cap}")
            print(f"Black Box: {self.is_black_box}")
            print("\nAgent States:")
            for agent_id, state in self.agent_states.items():
                print(f"  {agent_id}: Cash={state.cash:.2f}, Holdings={state.holdings}, Total={state.total_value:.2f}, Bad={state.is_bad}")
            print("\nOrder Books:")
            for asset in self.available_assets:
                ob_state = self.order_book.get_order_book_state(asset)
                print(f"  {asset}: Bid={ob_state.get('best_bid')}, Ask={ob_state.get('best_ask')}, Spread={ob_state.get('spread')}")
            print("-" * 50)
    
    def close(self):
        pass
