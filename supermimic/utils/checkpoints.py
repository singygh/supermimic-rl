"""Checkpoint selection and actor-only loading, independent of Isaac Sim."""
import json
from datetime import datetime
from pathlib import Path
import re

import torch

from supermimic.configs import TaskSpec
from supermimic.models import Actor


def compatible_policy_spec(trained, current):
    """v2 changes G1 reward units only; v1 actors retain their inference contract."""
    from dataclasses import asdict

    left, right = asdict(trained), asdict(current)
    if (left["version"], right["version"]) == ("g1-isaaclab-tracking-v1", "g1-isaaclab-tracking-v2"):
        left["version"] = right["version"]
    return left == right


def _run_time(path):
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_(\d{6}))?(?:_.*)?", path.name)
    if match is None:
        return None
    try:
        return datetime.strptime(match[1] + (match[2] or "000000"), "%Y-%m-%d_%H-%M-%S%f")
    except ValueError:
        return None


def latest_checkpoint(log_dir):
    """Choose newest timestamped run first, then its newest completed checkpoint.

    A newest run without checkpoints raises explicitly, rather than silently
    playing an older policy. Explicit run directories and legacy flat logs work.
    """
    root = Path(log_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {root}. Set --log-dir or --checkpoint.")
    if _run_time(root) is not None or (root / "run_info.json").is_file():
        selected = root
    else:
        runs = [(stamp, str(path), path) for path in root.rglob("*") if path.is_dir()
                and (stamp := _run_time(path)) is not None]
        selected = max(runs)[2] if runs else None
    search = selected.glob("model_*.pt") if selected is not None else root.rglob("model_*.pt")
    candidates = [p for p in search
                  if re.fullmatch(r"model_(latest|\d+)\.pt", p.name) and p.is_file() and p.stat().st_size > 0]
    if not candidates:
        if selected is not None:
            raise FileNotFoundError(f"Latest training run has no completed checkpoint: {selected}. "
                                    "Wait for its first save or use --checkpoint to select an older policy.")
        raise FileNotFoundError(f"No model_latest.pt or model_<iteration>.pt under {root}")
    def order(path):
        suffix = path.stem.removeprefix("model_")
        return path.stat().st_mtime_ns, float("inf") if suffix == "latest" else int(suffix), str(path)
    return max(candidates, key=order)


def load_policy_checkpoint(path):
    """Extract policy payload on CPU and release replay/optimizer tensors.

    Full training checkpoints contain Python objects; only load trusted files.
    Legacy Actor architecture is inferred from Linear weight shapes.
    """
    path = Path(path).expanduser().resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1 or "actor" not in state.get("algorithm", {}):
        raise ValueError(f"Not a supported SuperMimic checkpoint: {path}")
    spec = TaskSpec(**state["task_spec"])
    weights = state["algorithm"]["actor"]
    config = state.get("policy_config")
    if config is None:
        def widths(prefix):
            layers = [(int(key.split(".")[1]), value.shape[0]) for key, value in weights.items()
                      if re.fullmatch(prefix + r"\.\d+\.weight", key)]
            return [width for _, width in sorted(layers)]
        decoder, encoder = widths("decoder"), widths("goal_encoder")
        if len(decoder) < 2 or len(encoder) < 2 or decoder[-1] != spec.action_dim:
            raise ValueError("Cannot infer Actor architecture; checkpoint needs a supported policy_config")
        config = {"hidden_dims": tuple(decoder[:-1]), "latent_dim": encoder[-1],
                  "encoder_dims": tuple(encoder[:-1]) if spec.version.startswith("g1-") else None}
    metadata_path = path.parent / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    env = state.get("env", {})
    return {"weights": weights, "config": config, "spec": spec, "iteration": state["iteration"],
            "seed": state["cfg"].get("seed", 1), "metadata": metadata,
            "task": metadata.get("task") or env.get("task_id"),
            "motion_file": metadata.get("motion_file") or env.get("motion_file"),
            "motion_digest": env.get("motion_digest", metadata.get("motion_sha256")),
            "robot_urdf": env.get("robot_urdf")}


def build_inference_actor(payload, device="cpu"):
    actor = Actor(payload["spec"], **payload["config"])
    actor.load_state_dict(payload["weights"], strict=True)
    return actor.to(device).eval().requires_grad_(False)
