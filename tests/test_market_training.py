import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import polars as pl
import torch
from src.agents.learning import ActorCritic
from src.data.augment import augment_stock_data
from src.data.synthetic import generate_market_prices
from src.environment.market_env import MarketEnvironment
from src.events.event_manager import EventManager, MarketEvent, EventType, EventEffects
from src.ingestors.csv import ingest_csv
from src.market.order_book import OrderBook, OrderSide
from src.models.incremental_reward import IncrementalRewardCalculator
from src.models.robust_ensemble import RobustEnsemble
from src.simulation.black_box import BlackBoxPredictionSystem, Prediction, PredictionType
from src.training.trainer import TrainConf, TrainingPipeline, ExperienceReplay
from src.utils.logging import JSONLogger


def holds(env):
    return {a: np.zeros(4) for a in env.agents}


class MarketTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logger = JSONLogger(self.tmp.name)

    def env(self, prices=None, n=1, kinds=None, **kwargs):
        prices = prices if prices is not None else [100.] * 7
        return MarketEnvironment(n_agents=n, episode_length=len(prices)-1,
            price_data={'A': prices}, agent_types=kinds or ['base']*n,
            event_manager=EventManager(False), logger=self.logger, seed=7, **kwargs)

    def test_logger_buffer_boundary_does_not_deadlock(self):
        code = "from src.utils.logging import JSONLogger,LogCategory; import tempfile; d=tempfile.mkdtemp(); l=JSONLogger(d,buffer_size=2); [l.info(LogCategory.TRAINING,str(i)) for i in range(101)]; l.flush(); assert len(l.log_file.read_text().splitlines())==101"
        subprocess.run([sys.executable, '-c', code], check=True, timeout=10)

    def test_hold_has_zero_reward_and_market_advances(self):
        env = self.env([100., 110., 120.])
        obs, rewards, terminated, truncated, _ = env.step(holds(env))
        self.assertEqual(rewards['agent_0'], 0.)
        self.assertAlmostEqual(obs['agent_0']['current_price'][0], 110.)
        self.assertFalse(terminated or truncated)  # 10% growth is not an episode ending.
        _, _, terminated, truncated, _ = env.step(holds(env))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            env.step(holds(env))

    def test_market_execution_conserves_cash_and_inventory(self):
        env = self.env(fee_rate=.001, impact_coefficient=0.)
        env.step({'agent_0': [1, 0, 10, 100]})
        state = env.agent_states['agent_0']
        self.assertEqual(state.holdings['A'], 10)
        self.assertAlmostEqual(state.cash + env.lp_cash + env.fees_collected, 10000.)
        self.assertEqual(state.holdings['A'] + env.lp_holdings['A'], 0)
        self.assertIs(state, env.agent_objects['agent_0'].state)
        self.assertFalse(any(o.agent_id == 'agent_0' for o in env.order_book.get_order_book('A').order_map.values()))
        _, rewards, _, _, _ = env.step({'agent_0': [2, 0, 10, 100]})
        self.assertLess(state.total_value, 10000.)
        self.assertLess(rewards['agent_0'], 0.)
        self.assertAlmostEqual(state.cash+env.lp_cash+env.fees_collected, 10000.)

    def test_mark_to_market_when_holding(self):
        env = self.env([100., 100., 110., 90.], fee_rate=0, spread=0, impact_coefficient=0)
        env.step({'agent_0': [1, 0, 10, 100]})
        _, up, _, _, _ = env.step(holds(env))
        self.assertAlmostEqual(env.agent_states['agent_0'].total_value, 10100.)
        _, down, _, _, _ = env.step(holds(env))
        self.assertGreater(up['agent_0'], 0)
        self.assertLess(down['agent_0'], 0)
        self.assertAlmostEqual(env.agent_states['agent_0'].total_value, 9900.)

    def test_unfilled_limits_reserve_cash_and_later_settle(self):
        env = self.env([100., 100., 90., 90.], impact_coefficient=0)
        env.step({'agent_0': [3, 0, 100, 95]})
        self.assertEqual(env.agent_states['agent_0'].holdings, {})
        self.assertLess(env._observe('agent_0')['available_cash'], 501)
        env.step(holds(env))
        state = env.agent_states['agent_0']
        self.assertEqual(state.holdings['A'], 100)
        self.assertAlmostEqual(state.cash+env.lp_cash+env.fees_collected, 10000.)

    def test_peer_limit_fill_transfers_both_sides(self):
        env = self.env(n=2, impact_coefficient=0, fee_rate=0)
        env._execute_action('agent_0', [1, 0, 10, 100])
        env._execute_action('agent_0', [4, 0, 10, 100.01])
        before = env.agent_states['agent_0'].cash
        env._execute_action('agent_1', [1, 0, 10, 100])
        self.assertEqual(env.agent_states['agent_0'].holdings['A'], 0)
        self.assertEqual(env.agent_states['agent_1'].holdings['A'], 10)
        self.assertAlmostEqual(env.agent_states['agent_0'].cash-before, 1000.1)
        self.assertAlmostEqual(sum(s.cash for s in env.agent_states.values())+env.lp_cash, 20000.)

    def test_no_overspending_or_short_sales(self):
        env = self.env(initial_cash=100.)
        env._execute_action('agent_0', [1, 0, 1000, 100])
        env._execute_action('agent_0', [2, 0, 1000, 100])
        self.assertGreaterEqual(env.agent_states['agent_0'].cash, 0)
        self.assertTrue(all(q>=0 for q in env.agent_states['agent_0'].holdings.values()))
        for action in ([1, -1, 1, 100], [1, 0, -5, 100], [1, 0, 1, float('nan')]):
            with self.assertRaises(ValueError):
                env.step({'agent_0': action})

    def test_adversarial_reward_requires_intervention_and_reset_keeps_role(self):
        env = self.env(n=5, kinds=['rlaif']*4+['adversarial'])
        _, rewards, _, _, infos = env.step(holds(env))
        self.assertEqual(rewards['agent_4'], 0.)
        self.assertEqual(infos['agent_4']['market_damage'], 0.)
        actions = holds(env); actions['agent_4'] = [1, 0, 10, 100]
        _, _, _, _, infos = env.step(actions)
        self.assertGreater(infos['agent_4']['market_damage'], 0.)
        self.assertEqual(infos['agent_0']['market_damage'], 0.)
        agent = env.agent_objects['agent_4']
        env.reset()
        self.assertIs(agent, env.agent_objects['agent_4'])
        self.assertTrue(agent.state.is_bad)
        self.assertEqual(agent.state.villain_points, 10)
        with self.assertRaises(ValueError):
            self.env(n=2, kinds=['rlaif','adversarial'])

    def test_event_duration_uses_steps_and_event_budget(self):
        env = self.env(n=5, kinds=['rlaif']*4+['adversarial'])
        event = MarketEvent(1, 'test', list(EventType)[0], 0., 2, 5, 1, effects=EventEffects(price_drift=-.02))
        env.event_manager.add_event(event)
        env.agent_objects['agent_4'].pending_event_id = 1
        _, _, _, _, info = env.step(holds(env))
        self.assertTrue(event.is_active)
        self.assertEqual(env.agent_states['agent_4'].villain_points, 5)
        self.assertGreater(info['agent_4']['market_damage'], 0)
        env.step(holds(env))
        _, _, _, _, info = env.step(holds(env))
        self.assertFalse(event.is_active)
        self.assertEqual(info['agent_4']['market_damage'], 0)

    def test_forecast_scores_only_realized_future_window(self):
        env = self.env([100.,101.,102.,103.,104.,999.], kinds=['rlaif'], black_box_frequency=1, black_box_duration=2)
        for _ in range(3):
            env.step(holds(env))
        result = env.black_box_system.pred_results[0]
        self.assertEqual(result.actual, [102.,103.])
        self.assertNotIn(999., result.actual)

    def test_padding_is_persistence_not_artificial_zero_jump(self):
        env = self.env()
        obs = env._observe('agent_0')
        np.testing.assert_array_equal(obs['price_history'][:,0], 100.)
        self.assertTrue(np.isfinite(ActorCritic.features(obs)).all())

    def test_reward_components_sum_and_loss_is_not_repeated(self):
        calc = IncrementalRewardCalculator()
        r, c = calc.calculate_step_reward(100, 90, 0, .1, is_adversarial=True, market_damage=.2)
        self.assertAlmostEqual(r, sum(c.__dict__.values()))
        r, _ = calc.calculate_step_reward(90, 90, .1, .1)
        self.assertEqual(r, 0)
        with self.assertRaises(ValueError):
            calc.calculate_step_reward(0, 0, 0, 0)

    def test_preference_does_not_penalize_passive_weight_changes(self):
        env = self.env([100.,100.,110.], kinds=['rlaif'], fee_rate=0, impact_coefficient=0)
        env.agent_objects['agent_0'].preference_model.weights[:] = 1.
        env.step({'agent_0': [1, 0, 10, 100]})
        _, _, _, _, info = env.step(holds(env))
        self.assertAlmostEqual(info['agent_0']['reward_components']['preference'], 0.)

    def test_orderbook_rejects_invalid_orders_and_self_trades(self):
        book = OrderBook()
        book.add_order('a', OrderSide.BUY, 100, 5)
        book.add_order('a', OrderSide.SELL, 99, 5)
        self.assertEqual(len(book.trades), 0)
        with self.assertRaises(ValueError):
            book.add_order('a', OrderSide.BUY, 100, 0)

    def test_robust_ensemble_shape_and_no_identity_oracle(self):
        model = BlackBoxPredictionSystem()
        preds = [Prediction(str(i), PredictionType.PRICE_LEVEL, [100.,101.], (90.,110.), .7) for i in range(5)]
        self.assertEqual(np.asarray(model._robust_ensemble(preds)).shape, (2,))
        reports = np.array([[[1.,2.]],[[1.,2.]],[[1.,2.]],[[1.,2.]],[[99.,99.]]])
        a = RobustEnsemble(5,1,2, ['rlaif']*5)
        b = RobustEnsemble(5,1,2, ['rlaif']*4+['adversarial'])
        np.testing.assert_equal(a.combine(reports)[0], b.combine(reports)[0])

    def test_metrics_use_consistent_covariance_and_compounding(self):
        from src.evaluation.metrics import MetricsCalculator
        calc = MetricsCalculator(logger=self.logger, periods_per_year=2)
        beta, alpha = calc._calculate_beta_alpha([.1,-.1], [.1,-.1])
        self.assertAlmostEqual(beta, 1.)
        self.assertAlmostEqual(alpha, 0.)
        self.assertAlmostEqual(calc._calculate_annualized_return([.1,.1]), .21)

    def test_multi_asset_orders_reserve_shared_cash(self):
        env = MarketEnvironment(n_agents=1, episode_length=2, initial_cash=1000,
            price_data={'A':[100.]*3, 'B':[100.]*3}, agent_types=['base'],
            event_manager=EventManager(False), logger=self.logger)
        env._execute_action('agent_0',[3,0,8,95])
        env._execute_action('agent_0',[3,1,8,95])
        reserved, _ = env._reserved('agent_0')
        self.assertLessEqual(reserved, 1000.)
        self.assertEqual(sum(o.qty for b in env.order_book.order_books.values()
                             for o in b.order_map.values() if o.agent_id == 'agent_0'),10)

    def test_data_validation_and_chronology(self):
        path = Path(self.tmp.name)/'prices.csv'
        path.write_text('date,Close,Volume\n2024-01-03,103,1000\n2024-01-01,101,999\n2024-01-02,102,1001\n')
        frame = ingest_csv(path.as_posix())
        self.assertEqual(frame['price'].to_list(), [101.,102.,103.])
        self.assertEqual(augment_stock_data(frame, 2, 1, 2), [[101.,102.,103.]]*2)
        with self.assertRaises(ValueError):
            augment_stock_data(frame,1,100,10)
        with self.assertRaises(ValueError):
            self.env([100.,float('nan')])
        a = generate_market_prices(100,42,3)
        self.assertEqual(a, generate_market_prices(100,42,3))
        self.assertNotEqual(a, generate_market_prices(100,43,3))

    def test_zero_priority_replay(self):
        replay = ExperienceReplay(3)
        for _ in range(3):
            replay.add({}, np.zeros(4), 0, {}, priority=0)
        self.assertEqual(len(replay.sample(2)), 2)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def pipeline(self, name, n=2, **kwargs):
        folder = str(Path(self.tmp.name)/name)
        conf = TrainConf(n_episodes=n, episode_len=35, checkpoint_dir=folder+'/ckpt',
                         log_dir=folder, eval_freq=0, save_freq=0, **kwargs)
        return TrainingPipeline(conf, logger=JSONLogger(folder))

    def test_all_policies_learn_and_survive_resets(self):
        pipe = self.pipeline('learn')
        ids = {a: id(obj) for a,obj in pipe.env.agent_objects.items()}
        before = {a: copy.deepcopy(obj.policy.state_dict()) for a,obj in pipe.env.agent_objects.items()}
        pipe.train()
        self.assertEqual(pipe.curr_ep, 2)
        for a,obj in pipe.env.agent_objects.items():
            self.assertEqual(id(obj), ids[a])
            self.assertEqual(obj.policy.updates, 4)
            self.assertTrue(any(not torch.equal(before[a][k],v) for k,v in obj.policy.state_dict().items()))
        for result in pipe.train_log:
            self.assertAlmostEqual(result['episode_reward'], sum(result['agent_rewards'].values()))
            for aid in ids:
                self.assertAlmostEqual(result['agent_rewards'][aid], sum(result['reward_components'][aid].values()))
        before = {a: copy.deepcopy(obj.snapshot()) for a,obj in pipe.env.agent_objects.items()}
        rng_before = torch.get_rng_state().clone()
        pipe.evaluate(n_episodes=2)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        for a,obj in pipe.env.agent_objects.items():
            self.assertEqual(before[a]['preference'], obj.snapshot()['preference'])
            for k,v in obj.policy.state_dict().items():
                self.assertTrue(torch.equal(before[a]['policy']['model'][k],v))

    def test_resume_matches_uninterrupted_training(self):
        full = self.pipeline('full', 3)
        full.train()
        part = self.pipeline('part', 1)
        part.train()
        resumed = self.pipeline('resumed',3,resume=str(Path(part.conf.checkpoint_dir)/'latest.json'))
        resumed.train()
        self.assertEqual(full.ep_reward, resumed.ep_reward)
        for a,obj in full.env.agent_objects.items():
            for k,v in obj.policy.state_dict().items():
                self.assertTrue(torch.equal(v, resumed.env.agent_objects[a].policy.state_dict()[k]))

    def test_sequential_curriculum_and_no_adversary(self):
        pipe = self.pipeline('sequential', n=5, sequential_levels=True, episodes_per_level=1, adversial_train=False)
        pipe.train()
        self.assertEqual(pipe.scheduler.curr_lvl_idx, 3)
        self.assertEqual(pipe.curr_ep, 5)
        self.assertNotIn('adversarial', pipe.env.agent_types)

    def test_policy_learns_reward_direction_on_controlled_task(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)
        policy = ActorCritic(2, learning_rate=.003)
        obs = {'price_history':np.full((50,10),100.), 'current_price':np.full(10,100.),
               'available_assets':np.ones(10),'agent_holdings':np.zeros(10),'agent_cash':100.,'agent_total_value':100.}
        for _ in range(160):
            action = policy.select(obs, np.ones(2,dtype=bool))
            policy.observe(1. if action == 1 else -1., obs, True, False)
        with torch.no_grad():
            logits, _ = policy(torch.from_numpy(policy.features(obs)))
        self.assertGreater(float(torch.softmax(logits, -1)[1]), .95)

    def test_historical_train_evaluation_windows_do_not_overlap(self):
        folder = self.tmp.name
        conf = TrainConf(episode_len=10, checkpoint_dir=folder+'/ckpt', log_dir=folder)
        data = {'A': list(range(100, 300))}
        pipe = TrainingPipeline(conf, logger=JSONLogger(folder), price_data=data)
        train = pipe._gen_price_data(seed=8)
        test = pipe._gen_price_data(seed=8, evaluation=True)
        self.assertLess(max(train['A']), min(test['A']))

    def test_bootstrap_uses_termination_not_truncation(self):
        policy = ActorCritic(2)
        with torch.no_grad():
            for p in policy.parameters(): p.zero_()
            policy.critic.bias.fill_(2)
        obs = {'price_history':np.full((50,10),100.), 'current_price':np.full(10,100.),
               'available_assets':np.ones(10),'agent_holdings':np.zeros(10),'agent_cash':100.,'agent_total_value':100.}
        policy.update = lambda: None
        policy.select(obs, np.ones(2,dtype=bool))
        policy.observe(0,obs,False,True)
        self.assertEqual(policy.rollout[-1][-2], 2.)
        policy.select(obs, np.ones(2,dtype=bool))
        policy.observe(0,obs,True,False)
        self.assertEqual(policy.rollout[-1][-2], 0.)



class FixTests(MarketTests):
    """Regression tests for the issues found while auditing runs against the code."""
    # Re-running every MarketTests case here is unnecessary; keep only the new ones.
    for _name in [n for n in dir(MarketTests) if n.startswith('test_')]:
        locals()[_name] = None
    del _name

    def adv_env(self, prices, **kwargs):
        env = self.env(prices, n=5, kinds=['rlaif']*4+['adversarial'], **kwargs)
        env.black_box_system.curr_king = 'agent_0'
        return env

    def test_news_arrives_at_onset_and_is_not_a_predictable_ramp(self):
        env = self.env([100.]*30)
        env.event_manager.add_event(MarketEvent(1, 'shock', list(EventType)[0], 0., 10, 50, 1, effects=EventEffects(price_drift=-.15)))
        env.event_manager.trigger_event(1, {'step': 0})
        path = [env.current_prices['A']]
        for _ in range(8):
            env.step(holds(env)); path.append(env.current_prices['A'])
        first_drop = path[1] / path[0] - 1
        later = [path[i+1] / path[i] - 1 for i in range(1, 8)]
        self.assertLess(first_drop, -.04)                 # the information arrives at once
        self.assertTrue(all(abs(r) < .005 for r in later))  # and does not keep trending
        legacy = self.env([100.]*30, event_impact='ramp')
        legacy.event_manager.add_event(MarketEvent(1, 'shock', list(EventType)[0], 0., 10, 50, 1, effects=EventEffects(price_drift=-.15)))
        legacy.event_manager.trigger_event(1, {'step': 0})
        legacy.step(holds(legacy))
        self.assertGreater(legacy.current_prices['A'], 98.)  # legacy ramp: ~1%/step, a predictable trend

    def test_own_flow_costs_slippage_and_cash_is_conserved(self):
        env = self.env(initial_cash=100000., fee_rate=.0001, impact_coefficient=.002)
        env._execute_action('agent_0', [1, 0, 500, 100])
        state = env.agent_states['agent_0']
        self.assertAlmostEqual(state.cash + env.lp_cash + env.fees_collected, 100000.)
        self.assertGreater(env.lp_cash, 500 * 100 * 1.0004)  # paid more than the quote: slippage is a cost

    def test_kaiser_bonus_is_causal(self):
        prices = [100., 100., 90., 90., 90., 90., 90.]
        passive = self.adv_env(prices)
        passive._execute_action('agent_0', [1, 0, 50, 100])
        _, _, _, _, info = passive.step(holds(passive))
        self.assertEqual(info['agent_4']['reward_components']['kaiser_damage'], 0.)
        active = self.adv_env(prices)
        active._execute_action('agent_0', [1, 0, 50, 100])
        active.event_manager.add_event(MarketEvent(1, 'noop', list(EventType)[0], 0., 4, 5, 1, effects=EventEffects()))
        active.agent_objects['agent_4'].pending_event_id = 1
        active.step(holds(active))
        active.agent_objects['agent_4'].pending_event_id = None
        _, _, _, _, info = active.step(holds(active))
        self.assertGreater(info['agent_4']['reward_components']['kaiser_damage'], 0.)
        self.assertTrue(active._intervened['agent_4'])

    def test_adversary_reward_components_sum_to_reward(self):
        calc = IncrementalRewardCalculator(forecast_weight=0., preference_weight=.1, drawdown_penalty=.1)
        for kaiser in (0., .4, 5.):
            total, comp = calc.calculate_step_reward(1000., 990., .0, .01, is_adversarial=True, market_damage=2.,
                kaiser_damage_bonus=kaiser, agent_characteristics={'risk_preference': .6, 'aggression_style': 'front_run'})
            self.assertAlmostEqual(total, sum(v for v in comp.__dict__.values()))
        total, comp = calc.calculate_step_reward(1000., 1000., 0., 0., is_adversarial=True)
        self.assertEqual(total, 0.)  # no survival bonus: doing nothing pays nothing
        total, comp = calc.calculate_step_reward(1000., 990., .0, .01, agent_characteristics={'risk_preference': .2})
        self.assertAlmostEqual(total, sum(comp.__dict__.values()))  # ordinary drawdown is logged as charged

    def test_exactly_one_deterministic_king(self):
        system = BlackBoxPredictionSystem(duration=2, frequency=1)
        system.is_active = True
        same = Prediction('a', PredictionType.PRICE_LEVEL, [100., 100.], (90., 110.), .6)
        twin = Prediction('b', PredictionType.PRICE_LEVEL, [100., 100.], (90., 110.), .6)
        system.submit_prediction('a', same); system.submit_prediction('b', twin)
        results = system.evaluate_predictions([100., 100.], step=7)
        self.assertEqual(sum(r.is_king for r in results), 1)
        self.assertEqual(system.curr_king, 'a')           # tie -> submission order, not "last one wins"
        self.assertEqual(system.episode_summary()['rounds'], 1)

    def test_villain_points_from_bbox_are_on_by_default_and_capped(self):
        env = self.env([100.]*30, n=5, kinds=['rlaif']*4+['adversarial'], black_box_frequency=2, black_box_duration=1)
        self.assertTrue(env.bbox_villain_points)
        off = self.env([100.]*30, n=5, kinds=['rlaif']*4+['adversarial'], black_box_frequency=2, black_box_duration=1, bbox_villain_points=False)
        for _ in range(20):
            off.step(holds(off))
        self.assertEqual(off.agent_states['agent_4'].villain_points, 10)
        on = self.env([100.]*60, n=5, kinds=['rlaif']*4+['adversarial'], black_box_frequency=2, black_box_duration=1,
                      bbox_villain_points=True, max_villain_points=11, villain_points_per_win=3)
        for _ in range(50):
            on.step(holds(on))
        self.assertLessEqual(on.agent_states['agent_4'].villain_points, 11)

    def test_event_liquidity_effects_are_applied(self):
        env = self.env([100.]*10, liquidity=1000)
        env.event_manager.add_event(MarketEvent(1, 'stress', list(EventType)[0], 0., 5, 5, 1, effects=EventEffects(liquidity_multiplier=.5, spread_multiplier=2.)))
        env.event_manager.trigger_event(1, {'step': 0})
        depth, spread = env._liquidity_conditions()
        self.assertEqual(depth, 500)
        self.assertAlmostEqual(spread, env.spread * 2)

    def test_gap_through_a_resting_limit_fills_at_the_market_not_the_stale_limit(self):
        env = self.env([100., 90., 90., 90.], spread=.001, fee_rate=0., impact_coefficient=0.)
        env._execute_action('agent_0', [3, 0, 10, 99.9])           # resting buy limit below the market
        self.assertEqual(env.agent_states['agent_0'].holdings.get('A', 0), 0)
        env.step(holds(env))                                        # the market gaps to ~90
        state = env.agent_states['agent_0']
        self.assertEqual(state.holdings['A'], 10)
        spent = 10000. - state.cash
        self.assertLess(spent, 10 * 90.1)                           # paid the 90.045 ask, not the stale 99.9
        self.assertAlmostEqual(state.cash + env.lp_cash + env.fees_collected, 10000.)

    def test_curriculum_gate_is_scale_free(self):
        from src.training.trainer import TrainingScheduler
        scheduler = TrainingScheduler(logger=self.logger)
        self.assertFalse(any(scheduler.should_progress(0.) for _ in range(5)))
        self.assertFalse(any(scheduler.should_progress(1.) for _ in range(2)))
        self.assertTrue(any(scheduler.should_progress(1.) for _ in range(5)))

    def test_episode_log_contains_mechanism_records(self):
        env = self.env([100.]*30, n=5, kinds=['rlaif']*4+['adversarial'], black_box_frequency=5, black_box_duration=2)
        for _ in range(20):
            env.step(holds(env))
        log = env.episode_log()
        self.assertGreater(log['bbox']['rounds'], 0)
        self.assertIn('wealth', log['liquidity_provider'])
        self.assertIn('agent_4', log['adversary_profiles'])

    def test_liquidity_provider_inventory_is_finite(self):
        env = self.env([100.]*30, initial_cash=1e9, lp_max_inventory=30)
        env._execute_action('agent_0', [1, 0, 1000, 100])
        self.assertLessEqual(abs(env.lp_holdings['A']), 30)       # the provider cannot be drained without limit
        env.step(holds(env))
        self.assertLessEqual(abs(env.lp_holdings['A']), 30)
        self.assertGreater(env.lp_limit_hits, 0)

    def test_kaiser_effects_can_be_switched_off(self):
        env = self.adv_env([100., 100., 90., 90., 90.], kaiser_effects=False, black_box_frequency=1, black_box_duration=1)
        env._execute_action('agent_0', [1, 0, 50, 100])
        env.event_manager.add_event(MarketEvent(1, 'noop', list(EventType)[0], 0., 4, 5, 1, effects=EventEffects()))
        env.agent_objects['agent_4'].pending_event_id = 1
        for _ in range(3):
            _, _, _, _, info = env.step(holds(env))
        self.assertEqual(info['agent_4']['reward_components']['kaiser_damage'], 0.)
        self.assertEqual(env._observe('agent_1')['kaiser_direction'], 0.)

    def test_policy_input_includes_consensus_and_kaiser_direction(self):
        from src.agents.learning import ActorCritic, FEATURE_DIM
        env = self.adv_env([100.]*30)
        env.agent_states['agent_0'].last_action = {'type': 1}
        obs = env._observe('agent_1')
        self.assertEqual(obs['kaiser_direction'], 1.)
        obs['bbox_consensus'] = .02
        x = ActorCritic.features(obs)
        self.assertEqual(len(x), FEATURE_DIM)
        self.assertEqual(x[65], 1.); self.assertAlmostEqual(float(x[64]), 2.)

    def test_event_prior_is_part_of_the_credited_policy(self):
        import numpy as np
        env = self.adv_env([100.]*30)
        agent = env.agent_objects['agent_4']
        self.assertEqual(agent.event_choice, 'prior')
        obs = env._observe('agent_4')
        agent.decide_action(obs)
        x, choice, mask, value, bias = agent.policy.pending
        self.assertIsNotNone(bias)                               # stored so the update uses the same distribution
        self.assertEqual(len(bias), len(mask))
        agent.policy.observe(1., obs, False, False)
        self.assertEqual(len(agent.policy.rollout), 1)

    def test_execution_noise_is_off_and_forecast_skill_differs(self):
        env = self.env([100.]*80, n=4, kinds=['rlaif']*4)
        agents = [a for a in env.agent_objects.values()]
        self.assertFalse(any(a.execution_noise for a in agents))
        self.assertGreater(len({a.forecast_lookback for a in agents}), 1)
        self.assertTrue(all(5 <= a.forecast_lookback <= 60 for a in agents))

    def test_collapse_driver_is_attributed(self):
        env = self.env([100.] + [40.]*10)  # the tape itself halves
        _, _, terminated, _, _ = env.step(holds(env))
        self.assertTrue(terminated)
        self.assertEqual(env.episode_log()['collapse_driver'], 'tape')

    def test_old_64_feature_checkpoints_still_load(self):
        import torch
        from src.agents.learning import ActorCritic, FEATURE_DIM
        net = ActorCritic(41)
        old = {k: v.clone() for k, v in net.state_dict().items()}
        old['body.0.weight'] = old['body.0.weight'][:, :64].clone()
        fresh = ActorCritic(41)
        fresh.restore({'model': old, 'optimizer': fresh.optimizer.state_dict(), 'updates': 3})
        self.assertEqual(fresh.body[0].weight.shape[1], FEATURE_DIM)
        self.assertTrue(torch.equal(fresh.body[0].weight[:, 64:], torch.zeros(64, FEATURE_DIM - 64)))


class BiasAuditTests(unittest.TestCase):
    """Things that would silently bias an experiment."""

    def pipeline(self, **kw):
        import tempfile, os
        from src.training.trainer import TrainConf, TrainingPipeline
        from src.utils.logging import JSONLogger
        tmp = tempfile.mkdtemp()
        conf = TrainConf(n_episodes=2, episode_len=60, eval_freq=0, save_freq=0, checkpoint_dir=tmp + '/ck', log_dir=tmp + '/lg', **kw)
        os.makedirs(conf.checkpoint_dir)
        return TrainingPipeline(conf, logger=JSONLogger(conf.log_dir))

    def test_derived_seeds_do_not_overlap_between_runs(self):
        from src.training.trainer import derive_seed
        a = {derive_seed(1, 'episode', i) for i in range(200)}
        b = {derive_seed(2, 'episode', i) for i in range(200)}
        self.assertEqual(len(a), 200)
        self.assertFalse(a & b)  # seed + episode made run 2's episode 1 equal run 1's episode 2

    def test_same_seed_same_training(self):
        rewards = []
        for _ in range(2):
            pipe = self.pipeline(seed=11)
            pipe.train()
            rewards.append([r['agent_rewards'] for r in pipe.train_log])
        self.assertEqual(rewards[0], rewards[1])

    def test_evaluation_agents_keep_their_traits(self):
        import numpy as np
        pipe = self.pipeline(seed=3)
        trained = pipe.env.agent_objects
        np.random.seed(999)  # a different global RNG state, as at any later evaluation
        fresh = pipe.create_environment(seed=1_000_000, evaluation=True).agent_objects
        for aid in trained:
            fresh[aid].restore(trained[aid].snapshot())
            for name in trained[aid].TRAIT_FIELDS:
                self.assertEqual(getattr(trained[aid], name), getattr(fresh[aid], name), name)

    def test_adversary_has_the_same_position_cap_by_default(self):
        pipe = self.pipeline(seed=3)
        caps = {a.__class__.__name__: a.traits.max_position_size for a in pipe.env.agent_objects.values()}
        self.assertEqual(len(set(caps.values())), 1)
        wide = self.pipeline(seed=3, adversary_max_position=.6)
        self.assertEqual(max(a.traits.max_position_size for a in wide.env.agent_objects.values()), .6)

    def test_bootstrap_tester_does_not_touch_global_rng(self):
        import numpy as np
        from src.evaluation.metrics import StatisticalSignificanceTester
        np.random.seed(5); expected = np.random.random()
        np.random.seed(5); StatisticalSignificanceTester(n_bootstrap=10, random_seed=77)
        self.assertEqual(np.random.random(), expected)

    def test_random_play_conserves_cash_and_shares(self):
        import numpy as np
        for mode in ('jump', 'ramp'):
            pipe = self.pipeline(seed=4, event_impact=mode, lp_max_inventory=40)
            env = pipe.create_environment(seed=4, evaluation=True)
            obs, _ = env.reset(seed=4)
            for _ in range(250):
                actions = {aid: a.decide_action(obs[aid], env.action_space) for aid, a in env.agent_objects.items()}
                obs, rewards, terminated, truncated, _ = env.step(actions)
                n = len(env.agents)
                total = sum(s.cash for s in env.agent_states.values()) + env.lp_cash + env.fees_collected
                self.assertAlmostEqual(total, n * env.initial_cash, places=4)
                for asset in env.available_assets:
                    self.assertEqual(sum(s.holdings.get(asset, 0) for s in env.agent_states.values()) + env.lp_holdings[asset], 0)
                    self.assertLessEqual(abs(env.lp_holdings[asset]), 40)
                self.assertTrue(all(s.cash >= -1e-6 and all(v >= 0 for v in s.holdings.values()) for s in env.agent_states.values()))
                self.assertTrue(all(np.isfinite(r) for r in rewards.values()))
                if terminated or truncated:
                    break

if __name__ == '__main__':
    unittest.main()
