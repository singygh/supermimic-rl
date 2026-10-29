from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from supermimic.configs import AlgorithmConfig, RunnerConfig, TaskSpec
from supermimic.envs import ToyAdapter
from supermimic.models import WorldModel
from supermimic.models.heads import ToyObservation, ToyReward, ToyTermination
from supermimic.runners import SuperMimicRunner
from supermimic.utils import TrainingLogger


def make_runner(log_dir=None, backend="json"):
    torch.manual_seed(4)
    cfg = RunnerConfig(algorithm=AlgorithmConfig(model_batch_size=4, imagination_batch_size=4,
        critic_epochs=1, critic_mini_batches=1), hidden_dims=(16,), replay_capacity=32,
        console_log=False, logger=backend)
    env = ToyAdapter(2, max_episode_steps=10)
    model = WorldModel(env.spec, (16,), ToyObservation(), ToyReward(), ToyTermination())
    return SuperMimicRunner(env, cfg, log_dir, world_model=model)


class LoggingTests(unittest.TestCase):
    def test_episode_returns_and_lengths_across_batches(self):
        logger = TrainingLogger(None, RunnerConfig(console_log=False), 2, "cpu", TaskSpec())
        def step(rewards, terminated, truncated, ids, extras=None):
            logger.process_env_step(SimpleNamespace(reward=torch.tensor(rewards),
                terminated=torch.tensor(terminated), truncated=torch.tensor(truncated), extras=extras or {}),
                torch.tensor(ids))
        step([1., 2.], [False, False], [False, False], [0, 0])
        # An aggregate in extras must not be retained by reference.
        extra = torch.tensor([1., 3.])
        step([2., 2.], [False, True], [False, False], [0, 0], {"log": {"tracking": extra}})
        extra.fill_(99)
        step([3., 5.], [False, False], [True, False], [0, 1])
        metrics = logger.log({"iteration": 1, "collect_seconds": .1, "update_seconds": .2},
            total_iterations=3, call_elapsed=.4, call_iterations=1, total_env_steps=6,
            num_updates=0, replay_transitions=6)
        self.assertEqual(metrics["mean_episode_return"], 5.)  # (4 + 6) / 2
        self.assertEqual(metrics["mean_episode_length"], 2.5)
        self.assertEqual(metrics["episode_failure_rate"], .5)
        self.assertEqual(metrics["episode_timeout_rate"], .5)
        self.assertEqual(metrics["Episode/tracking"], 2.)
        self.assertAlmostEqual(metrics["eta_seconds"], .8)
        # External reset: discard the unfinished old episode instead of splicing it.
        step([1., 1.], [True, True], [False, False], [1, 2])
        self.assertEqual(list(logger.returns), [4., 6., 1., 1.])

    def test_logger_state_exact_and_physx_restart(self):
        cfg = RunnerConfig(console_log=False)
        logger = TrainingLogger(None, cfg, 2, "cpu", TaskSpec())
        logger.current_returns[:] = torch.tensor([3., 4.])
        logger.current_lengths[:] = torch.tensor([2, 3])
        logger.episode_ids[:] = 0
        logger.returns.extend([10., 12.])
        logger.lengths.extend([5, 6])
        state = logger.state_dict()
        restored = TrainingLogger(None, cfg, 2, "cpu", TaskSpec())
        restored.load_state_dict(state, iteration=7)
        torch.testing.assert_close(restored.current_returns, torch.tensor([3., 4.]))
        self.assertEqual(list(restored.returns), [10., 12.])
        restored.load_state_dict(state, iteration=7, exact_restore=False)
        self.assertTrue((restored.current_returns == 0).all())
        self.assertTrue((restored.current_lengths == 0).all())
        self.assertEqual(list(restored.returns), [10., 12.])

    def test_json_console_and_checkpoint_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = make_runner(directory)
            runner.cfg.console_log = True
            capture = io.StringIO()
            with redirect_stdout(capture):
                runner.learn(1)
            self.assertIn("SuperMimic learning iteration 1/1", capture.getvalue())
            self.assertIn("Mean episode return", capture.getvalue())
            self.assertIn("World model loss", capture.getvalue())
            self.assertIn("ETA", capture.getvalue())
            checkpoint = str(Path(directory) / "first.pt")
            runner.save(checkpoint)
            original_returns = runner.logger.current_returns.clone()
            runner.learn(1)
            expected = deepcopy(runner.alg.actor.state_dict())
            restored = make_runner(directory)
            restored.cfg.log_interval = 5  # Logging preferences may change on resume.
            restored.load(checkpoint)
            torch.testing.assert_close(restored.logger.current_returns, original_returns)
            restored.learn(1)
            for key, value in expected.items():
                torch.testing.assert_close(value, restored.alg.actor.state_dict()[key], rtol=0, atol=0)
            rows = [json.loads(line) for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(rows[-1]["iteration"], 2)
            self.assertEqual(rows[-1]["total_env_steps"], 64)
            self.assertIn("mean_episode_return", rows[-1])
            self.assertIn("actor_learning_rate", rows[-1])
            self.assertTrue((Path(directory) / "training.log").is_file())
            self.assertTrue((Path(directory) / "task_spec.json").is_file())

    @unittest.skipUnless(importlib.util.find_spec("tensorboard"), "optional TensorBoard dependency")
    def test_tensorboard_scalars_and_logging_does_not_change_updates(self):
        baseline = make_runner()
        baseline.learn(2)
        expected = deepcopy(baseline.alg.actor.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            logged = make_runner(directory, "tensorboard")
            logged.learn(2)
            self.assertIsNone(logged.logger.writer)  # closed and flushed on learn exit
            for key, value in expected.items():
                torch.testing.assert_close(value, logged.alg.actor.state_dict()[key], rtol=0, atol=0)
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            events = EventAccumulator(directory).Reload()
            for tag in ("Loss/actor", "Loss/world_model", "Grad/actor", "Train/mean_reward",
                        "Train/mean_episode_length", "Imagination/return", "Perf/total_fps",
                        "LearningRate/world_model"):
                self.assertIn(tag, events.Tags()["scalars"])
                self.assertEqual([event.step for event in events.Scalars(tag)], [1, 2])

    def test_legacy_checkpoint_load(self):
        original = make_runner()
        original.learn(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            original.save(path)
            state = torch.load(path, weights_only=False)
            state.pop("logger")
            for field in ("logger", "console_log", "log_interval", "episode_stats_window", "log_flush_secs"):
                state["cfg"].pop(field)
            torch.save(state, path)
            restored = make_runner()
            restored.load(path)
            self.assertEqual(restored.current_learning_iteration, 1)
            self.assertEqual(restored.learn(1)["total_env_steps"], 64)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
