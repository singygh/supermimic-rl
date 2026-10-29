from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from supermimic.configs import AlgorithmConfig, RunnerConfig
from supermimic.envs import ToyAdapter
from supermimic.runners import SuperMimicRunner
from supermimic.storage import ReplayBuffer


class CheckpointMemoryTests(unittest.TestCase):
    def test_atomic_save_and_error_log(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = RunnerConfig(algorithm=AlgorithmConfig(model_batch_size=2, imagination_batch_size=2,
                critic_epochs=1, critic_mini_batches=1), hidden_dims=(16,), replay_capacity=16,
                logger="json", console_log=False, save_interval=1)
            runner = SuperMimicRunner(ToyAdapter(2), cfg, directory)
            path = Path(directory) / "model_1.pt"
            runner.save(path)
            previous = path.read_bytes()
            def failed_write(state, stream):
                stream.write(b"incomplete checkpoint")
                raise OSError("test: disk write failed")
            with patch("supermimic.runners.supermimic_runner.torch.save", side_effect=failed_write):
                with self.assertRaisesRegex(OSError, "disk write failed"):
                    runner.learn(1)
            self.assertEqual(path.read_bytes(), previous)
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            self.assertIn("disk write failed", (Path(directory) / "error.log").read_text())

    def test_cpu_snapshot_order_and_independence(self):
        for size, cursor in ((0, 0), (3, 3), (5, 0), (5, 2)):
            replay = ReplayBuffer(5, 2)
            if size:
                replay.data = {"value": torch.arange(10).reshape(5, 2)}
            replay.size, replay.cursor = size, cursor
            state = replay.state_dict()
            for key, value in state["data"].items():
                expected = replay.data[key][replay._indices()].clone()
                self.assertEqual(value.device.type, "cpu")
                replay.data[key].zero_()
                torch.testing.assert_close(value, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA memory regression")
    def test_cuda_snapshot_and_restore_allocate_no_extra_replay(self):
        replay = ReplayBuffer(16, 2, "cuda:0")
        replay.data = {"value": torch.arange(16, device="cuda:0", dtype=torch.float32)
                       .reshape(16, 1, 1).expand(16, 2, 262144).clone()}  # 32 MiB
        replay.size, replay.cursor = 16, 7
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        state = replay.state_dict()
        torch.cuda.synchronize()
        self.assertEqual(state["data"]["value"].device.type, "cpu")
        self.assertLess(torch.cuda.max_memory_allocated() - baseline, 1024**2)
        torch.testing.assert_close(state["data"]["value"][:, 0, 0], replay._indices().cpu().float())
        restored = ReplayBuffer(16, 2, "cuda:0")
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        restored.load_state_dict(state)
        torch.cuda.synchronize()
        expected_bytes = replay.data["value"].numel() * replay.data["value"].element_size()
        self.assertLessEqual(torch.cuda.max_memory_allocated() - baseline, expected_bytes + 1024**2)
        torch.testing.assert_close(restored.data["value"].cpu(), state["data"]["value"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
