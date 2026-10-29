from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from supermimic.configs import TaskSpec
from supermimic.models import Actor
from supermimic.utils.checkpoints import (build_inference_actor, compatible_policy_spec,
                                        latest_checkpoint, load_policy_checkpoint)


class CheckpointTests(unittest.TestCase):
    def test_reward_version_policy_compatibility(self):
        old = TaskSpec(version="g1-isaaclab-tracking-v1")
        new = TaskSpec(version="g1-isaaclab-tracking-v2")
        self.assertTrue(compatible_policy_spec(old, new))
        self.assertFalse(compatible_policy_spec(old, TaskSpec(version=new.version, action_dim=2)))

    def test_latest_across_runs_by_modification_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "older", root / "newer"
            first.mkdir()
            second.mkdir()
            for path, stamp in ((first / "model_100.pt", 100), (second / "model_2.pt", 200),
                                (second / "model_9.pt", 200), (first / "not_a_model.pt", 300)):
                path.write_bytes(b"checkpoint")
                os.utime(path, (stamp, stamp))
            self.assertEqual(latest_checkpoint(root), second / "model_9.pt")
            (second / "model_latest.pt").write_bytes(b"checkpoint")
            os.utime(second / "model_latest.pt", (200, 200))
            self.assertEqual(latest_checkpoint(root), second / "model_latest.pt")
            (first / "model_999.pt").touch()  # ignore empty/incomplete initial files
            self.assertEqual(latest_checkpoint(root), second / "model_latest.pt")

    def test_missing_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                latest_checkpoint(directory)
            with self.assertRaises(FileNotFoundError):
                latest_checkpoint(Path(directory) / "missing")

    def test_actor_only_load_new_and_legacy_formats(self):
        torch.manual_seed(7)
        for g1 in (False, True):
            spec = TaskSpec(version="g1-test-v1" if g1 else "toy-test-v1")
            actor = Actor(spec, (16, 8), (12, 6) if g1 else None, 32 if g1 else None)
            history = torch.randn(2, spec.history_length, spec.obs_dim)
            goal = torch.randn(2, spec.goal_dim)
            state = {"format_version": 1, "algorithm": {"actor": actor.state_dict(),
                     "critic": {"unused": torch.ones(5)}}, "task_spec": asdict(spec),
                     "iteration": 10, "cfg": {"seed": 3}, "env": {"motion_file": "motion.npz",
                     "task_id": "Unitree-test", "motion_digest": "fixture", "robot_urdf": "g1.urdf"}}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model_latest.pt"
                for legacy in (False, True):
                    if legacy:
                        state.pop("policy_config", None)
                    else:
                        state["policy_config"] = actor.get_config()
                    torch.save(state, path)
                    payload = load_policy_checkpoint(path)
                    self.assertNotIn("algorithm", payload)
                    self.assertEqual(payload["task"], "Unitree-test")
                    self.assertEqual(payload["robot_urdf"], "g1.urdf")
                    restored = build_inference_actor(payload)
                    torch.testing.assert_close(restored(history, goal), actor(history, goal), rtol=0, atol=0)
                    self.assertFalse(restored.training)
                    self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
                metadata = {"task": "Unitree-recorded", "motion_file": "/tmp/recorded.npz"}
                (Path(directory) / "run_metadata.json").write_text(json.dumps(metadata))
                self.assertEqual(load_policy_checkpoint(path)["task"], "Unitree-recorded")

    def test_rejects_unrelated_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_latest.pt"
            torch.save({"model_state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "SuperMimic checkpoint"):
                load_policy_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
