import tempfile
import unittest

import numpy as np

from src.agents.learning import ActorCritic
from src.environment.market_env import MarketEnvironment
from src.events.event_manager import EventManager
from src.training.trainer import TrainConf, TrainingPipeline
from src.utils.logging import JSONLogger


class BBoxToggleTests(unittest.TestCase):
    def make_env(self, enabled):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        return MarketEnvironment(
            n_agents=1,
            max_bad_agents=0,
            episode_length=8,
            price_data={"A": [100.0] * 9},
            agent_types=["rlaif"],
            black_box_frequency=2,
            black_box_duration=2,
            bbox_enabled=enabled,
            event_manager=EventManager(False),
            logger=JSONLogger(self.tmp.name),
            seed=7,
        )

    @staticmethod
    def hold(env):
        return {agent_id: np.zeros(4, dtype=np.float32) for agent_id in env.agents}

    def test_enabled_bbox_generates_forecasts_and_trains_forecaster(self):
        env = self.make_env(True)
        first = env._observe("agent_0")
        self.assertEqual(first["is_black_box"], 0)
        for _ in range(8):
            observations, _, terminated, truncated, _ = env.step(self.hold(env))
            if terminated or truncated:
                break

        self.assertEqual(env.episode_log()["bbox"]["rounds"], 2)
        self.assertEqual(len(env.black_box_system.pred_results), 2)
        self.assertEqual(env.agent_objects["agent_0"].forecaster.updates, 2)

    def test_disabled_bbox_suppresses_contests_rewards_and_policy_features(self):
        env = self.make_env(False)
        observation = env._observe("agent_0")
        features = ActorCritic.features(observation)
        self.assertEqual(observation["is_black_box"], 0)
        self.assertEqual(observation["bbox_consensus"], 0.0)
        self.assertEqual(observation["kaiser_direction"], 0.0)
        np.testing.assert_array_equal(features[[51, 64, 65]], np.zeros(3, dtype=np.float32))

        for _ in range(8):
            observations, _, terminated, truncated, infos = env.step(self.hold(env))
            if terminated or truncated:
                break

        self.assertEqual(env.episode_log()["bbox"]["rounds"], 0)
        self.assertEqual(env.black_box_system.pred_results, [])
        self.assertEqual(env.agent_objects["agent_0"].forecaster.updates, 0)
        self.assertTrue(all(info["reward_components"]["forecast"] == 0.0 for info in infos.values()))
        self.assertTrue(all(ActorCritic.features(obs)[[51, 64, 65]].sum() == 0.0 for obs in observations.values()))

    def test_training_pipeline_propagates_disabled_setting(self):
        with tempfile.TemporaryDirectory() as folder:
            conf = TrainConf(
                n_episodes=1,
                episode_len=8,
                n_agents=1,
                mx_bad_agents=0,
                bbox_enabled=False,
                use_trainer=False,
                checkpoint_dir=folder + "/checkpoints",
                log_dir=folder + "/tracker",
            )
            pipeline = TrainingPipeline(conf, logger=JSONLogger(folder))

        self.assertFalse(pipeline.env.bbox_enabled)
        self.assertEqual(pipeline.env.reward_calculator.forecast_weight, 0.0)
        self.assertFalse(pipeline.env.agent_objects["agent_0"].forecast_training_enabled)

    def test_no_rlaif_training_uses_base_agents_and_can_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            conf = TrainConf(
                n_episodes=1,
                episode_len=8,
                n_agents=5,
                mx_bad_agents=1,
                rlaif_enabled=False,
                bbox_enabled=False,
                use_trainer=False,
                eval_freq=0,
                save_freq=0,
                checkpoint_dir=folder + "/checkpoints",
                log_dir=folder + "/tracker",
            )
            pipeline = TrainingPipeline(conf, logger=JSONLogger(folder))
            self.assertEqual(
                [agent.agent_type.value for agent in pipeline.env.agent_objects.values()],
                ["base", "base", "base", "base", "adversarial"],
            )
            pipeline.train()

        self.assertEqual(pipeline.curr_ep, 1)
        self.assertTrue(pipeline.env.agent_objects["agent_0"].snapshot())


if __name__ == "__main__":
    unittest.main()
