import copy
import json
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


if __name__ == '__main__':
    unittest.main()
