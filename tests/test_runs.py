from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest

from supermimic.utils.checkpoints import latest_checkpoint
from supermimic.utils.runs import create_run_directory, resolve_resume_checkpoint


class RunDirectoryTests(unittest.TestCase):
    def test_unique_timestamped_directories_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            stamp = datetime(2026, 10, 29, 12, 30, 45, 123456, tzinfo=timezone.utc)
            first = create_run_directory(directory, "dance", started_at=stamp)
            (first / "training.log").write_text("keep this run")
            second = create_run_directory(directory, "dance", started_at=stamp, resume_from=first / "model_250.pt")
            self.assertEqual(first.name, "2026-10-29_12-30-45_123456_dance")
            self.assertNotEqual(first, second)
            self.assertEqual((first / "training.log").read_text(), "keep this run")
            metadata = json.loads((second / "run_info.json").read_text())
            self.assertEqual(metadata["started_at"], stamp.isoformat())
            self.assertEqual(metadata["resume_from"], str(first / "model_250.pt"))
            self.assertEqual(metadata["log_root"], str(Path(directory).resolve()))

    def test_latest_run_precedes_checkpoint_modification_time(self):
        with tempfile.TemporaryDirectory() as directory:
            older = create_run_directory(directory, started_at=datetime(2026, 10, 28, 23, 59, 59))
            newer = create_run_directory(directory, started_at=datetime(2026, 10, 29, 0, 0, 0))
            old_model, new_model = older / "model_10000.pt", newer / "model_1.pt"
            old_model.write_bytes(b"old policy")
            new_model.write_bytes(b"new policy")
            os.utime(new_model, (100, 100))
            os.utime(old_model, (200, 200))  # touching an older run must not select it
            self.assertEqual(latest_checkpoint(directory), new_model)
            self.assertEqual(latest_checkpoint(older), old_model)
            self.assertEqual(resolve_resume_checkpoint("latest", directory), new_model)
            self.assertEqual(resolve_resume_checkpoint(str(older), directory), old_model)
            self.assertEqual(resolve_resume_checkpoint(str(old_model), directory), old_model)

    def test_newest_empty_run_reports_instead_of_using_older_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            older = create_run_directory(directory, started_at=datetime(2026, 10, 28))
            (older / "model_latest.pt").write_bytes(b"old policy")
            newer = create_run_directory(directory, started_at=datetime(2026, 10, 29))
            with self.assertRaisesRegex(FileNotFoundError, "Latest training run has no completed checkpoint"):
                latest_checkpoint(directory)
            # Half-written atomic save files are never considered playable.
            (newer / "model_250.pt.random.tmp").write_bytes(b"partial")
            with self.assertRaises(FileNotFoundError):
                latest_checkpoint(directory)

    def test_invalid_name_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("", ".", "..", "../outside", "a/b", "a\\b"):
                with self.assertRaises(ValueError):
                    create_run_directory(directory, name)


if __name__ == "__main__":
    unittest.main()
