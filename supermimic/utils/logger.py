"""Console, JSONL and TensorBoard metrics for world-model BPTT training."""
from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import warnings

import torch


LOG_CONFIG_FIELDS = {"logger", "console_log", "log_interval", "episode_stats_window", "log_flush_secs"}


def scalar_tag(key):
    exact = {
        "actor_loss": "Loss/actor", "critic_loss": "Loss/critic", "world_model_loss": "Loss/world_model",
        "real_reward": "Train/mean_step_reward", "mean_episode_return": "Train/mean_reward",
        "mean_episode_length": "Train/mean_episode_length",
        "mean_episode_seconds": "Train/mean_episode_seconds",
        "episode_failure_rate": "Train/episode_failure_rate", "episode_timeout_rate": "Train/episode_timeout_rate",
        "imagined_return": "Imagination/return", "survival_H": "Imagination/survival_H",
        "valid_window_fraction": "Replay/valid_window_fraction", "replay_transitions": "Replay/transitions",
        "total_steps_per_second": "Perf/total_fps", "collect_transitions_per_second": "Perf/collection_fps",
        "collect_seconds": "Perf/collection_time", "update_seconds": "Perf/learning_time",
        "iteration_seconds": "Perf/iteration_time", "elapsed_seconds": "Perf/elapsed_seconds",
        "eta_seconds": "Perf/eta_seconds", "total_env_steps": "Perf/total_env_steps",
        "learning_rate": "LearningRate/actor", "actor_learning_rate": "LearningRate/actor",
        "critic_learning_rate": "LearningRate/critic", "world_model_learning_rate": "LearningRate/world_model",
        "action_noise_std": "Policy/exploration_noise_std", "num_updates": "Train/updates",
    }
    if key in exact:
        return exact[key]
    if key.endswith("_grad_norm"):
        return "Grad/" + key.removesuffix("_grad_norm")
    if key.startswith(("model_", "replay_state_", "replay_normalized_", "termination_brier")):
        return "WorldModel/" + key
    return key if "/" in key else "Train/" + key


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingLogger:
    """Track completed episodes independently of backend availability.

    Episode returns are undiscounted sums of real simulator rewards. Truncation
    ends an episode for reporting, while remaining distinct from task failure.
    """

    def __init__(self, log_dir, cfg, num_envs, device, task_spec):
        self.log_dir = Path(log_dir) if log_dir else None
        self.cfg, self.task_spec = cfg, task_spec
        self.device = torch.device(device)
        self.current_returns = torch.zeros(num_envs, device=self.device)
        self.current_lengths = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.episode_ids = torch.full((num_envs,), -1, device=self.device, dtype=torch.long)
        self.returns = deque(maxlen=cfg.episode_stats_window)
        self.lengths = deque(maxlen=cfg.episode_stats_window)
        self.failures = deque(maxlen=cfg.episode_stats_window)
        self.timeouts = deque(maxlen=cfg.episode_stats_window)
        self.episode_extras = {}
        self.total_time = 0.
        self.writer = None
        self._purge_step = None
        self._warned_missing_tensorboard = False

    def start(self):
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "train_config.json").write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")
        (self.log_dir / "task_spec.json").write_text(json.dumps(asdict(self.task_spec), indent=2), encoding="utf-8")
        if self.cfg.logger == "tensorboard" and self.writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                if not self._warned_missing_tensorboard:
                    warnings.warn("TensorBoard is unavailable; using console/JSONL. Install supermimic-rl[logging] "
                                  "or select --logger json.", stacklevel=2)
                    self._warned_missing_tensorboard = True
            else:
                self.writer = SummaryWriter(str(self.log_dir), flush_secs=self.cfg.log_flush_secs,
                                            purge_step=self._purge_step)
                self._purge_step = None

    @torch.no_grad()
    def process_env_step(self, step, episode_ids):
        episode_ids = episode_ids.to(self.device)
        changed = self.episode_ids != episode_ids
        self.current_returns[changed] = 0
        self.current_lengths[changed] = 0
        self.episode_ids.copy_(episode_ids)
        self.current_returns += step.reward.detach().to(self.device)
        self.current_lengths += 1
        terminated = step.terminated.to(self.device)
        truncated = step.truncated.to(self.device)
        done = terminated | truncated
        if done.any():
            self.returns.extend(self.current_returns[done].cpu().tolist())
            self.lengths.extend(self.current_lengths[done].cpu().tolist())
            self.failures.extend(terminated[done].float().cpu().tolist())
            self.timeouts.extend((truncated & ~terminated)[done].float().cpu().tolist())
            self.current_returns[done] = 0
            self.current_lengths[done] = 0
            # Copy scalar aggregates now: Isaac Lab may reuse extras next step.
            extra = step.extras.get("episode", step.extras.get("log", {}))
            for key, value in extra.items():
                if not isinstance(value, (int, float, torch.Tensor)):
                    continue
                values = torch.as_tensor(value).detach().float().reshape(-1)
                values = values[torch.isfinite(values)]
                if values.numel():
                    count, total = self.episode_extras.get(key, (0, 0.))
                    self.episode_extras[key] = (count + values.numel(), total + values.sum().item())

    def log(self, metrics, *, total_iterations, call_elapsed, call_iterations, total_env_steps,
            num_updates, replay_transitions, final=False):
        metrics = dict(metrics)
        iteration = metrics["iteration"]
        elapsed = metrics["collect_seconds"] + metrics["update_seconds"]
        self.total_time += elapsed
        metrics.update(iteration_seconds=elapsed, elapsed_seconds=self.total_time,
                       eta_seconds=call_elapsed / max(call_iterations, 1) * max(total_iterations - iteration, 0),
                       total_env_steps=total_env_steps, num_updates=num_updates, replay_transitions=replay_transitions,
                       total_steps_per_second=self.cfg.num_steps_per_env * len(self.current_returns) / max(elapsed, 1e-9))
        if self.returns:
            metrics.update(mean_episode_return=statistics.mean(self.returns),
                           mean_episode_length=statistics.mean(self.lengths),
                           mean_episode_seconds=statistics.mean(self.lengths) * self.task_spec.dt,
                           episode_failure_rate=statistics.mean(self.failures),
                           episode_timeout_rate=statistics.mean(self.timeouts))
        for key, (count, total) in self.episode_extras.items():
            metrics[key if "/" in key else f"Episode/{key}"] = total / count
        self.episode_extras.clear()
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values()):
            raise ValueError("Training metrics must be finite numeric scalars")
        if self.log_dir:
            with (self.log_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        if self.writer is not None:
            for key, value in metrics.items():
                if key not in ("iteration", "learning_rate"):  # actor LR has its own explicit key
                    self.writer.add_scalar(scalar_tag(key), value, iteration)
        if iteration == 1 or iteration % self.cfg.log_interval == 0 or final:
            message = self.format_console(metrics, total_iterations)
            if self.cfg.console_log:
                print(message, flush=True)
            if self.log_dir:
                with (self.log_dir / "training.log").open("a", encoding="utf-8") as stream:
                    stream.write(message + "\n")
        return metrics

    def format_console(self, metrics, total_iterations):
        lines = ["#" * 88, f" SuperMimic learning iteration {metrics['iteration']}/{total_iterations} ".center(88), ""]
        def row(label, value):
            lines.append(f"{label + ':':>38} {value}")
        row("Status", "training" if "actor_loss" in metrics else
            "world model warmup" if "world_model_loss" in metrics else "replay warmup")
        row("Total environment steps", f"{metrics['total_env_steps']:,}")
        row("Replay transitions / updates", f"{metrics['replay_transitions']:,} / {metrics['num_updates']:,}")
        row("Completed / failed / timeout", " / ".join(str(int(metrics.get(key, 0)))
            for key in ("completed_episodes", "real_terminations", "real_truncations")))
        row("Steps per second (collect + update)", f"{metrics['total_steps_per_second']:,.0f}")
        row("Collection / learning time", f"{metrics['collect_seconds']:.3f}s / {metrics['update_seconds']:.3f}s")
        for key, label in (("world_model_loss", "World model loss"), ("actor_loss", "Actor loss"),
                           ("critic_loss", "Critic loss"), ("real_reward", "Mean real step reward"),
                           ("mean_episode_return", "Mean episode return"), ("mean_episode_length", "Mean episode length (steps)"),
                           ("imagined_return", "Imagined return"), ("survival_H", "Survival at H"),
                           ("valid_window_fraction", "Valid replay window fraction"),
                           ("episode_failure_rate", "Episode failure rate"),
                           ("actor_grad_norm", "Actor gradient norm"), ("critic_grad_norm", "Critic gradient norm"),
                           ("world_model_grad_norm", "World model gradient norm")):
            if key in metrics:
                row(label, f"{metrics[key]:.5g}")
        for key in sorted(metrics, key=lambda k: (k.rsplit("_", 1)[0],
                                                  int(k.rsplit("_", 1)[-1]) if k.rsplit("_", 1)[-1].isdigit() else 0, k)):
            if key.startswith(("replay_state_mse_", "replay_normalized_state_mse_")):
                row(key, f"{metrics[key]:.5g}")
            elif key.startswith(("Episode/", "Episode_", "Metrics/")):
                row(key, f"{metrics[key]:.5g}")
        row("Learning rates (actor/critic/model)", " / ".join(
            f"{metrics[k]:.3g}" for k in ("actor_learning_rate", "critic_learning_rate", "world_model_learning_rate")
            if k in metrics))
        row("Exploration action noise std", f"{metrics.get('action_noise_std', self.cfg.algorithm.action_noise_std):.3g}")
        lines.append("-" * 88)
        row("Elapsed / ETA", f"{duration(metrics['elapsed_seconds'])} / {duration(metrics['eta_seconds'])}")
        return "\n".join(lines)

    def state_dict(self):
        return {"current_returns": self.current_returns.clone(), "current_lengths": self.current_lengths.clone(),
                "episode_ids": self.episode_ids.clone(), "returns": list(self.returns), "lengths": list(self.lengths),
                "failures": list(self.failures), "timeouts": list(self.timeouts), "total_time": self.total_time}

    def load_state_dict(self, state, iteration, exact_restore=True):
        self.close()
        for key in ("current_returns", "current_lengths", "episode_ids"):
            getattr(self, key).copy_(state[key].to(self.device))
        for key in ("returns", "lengths", "failures", "timeouts"):
            target = getattr(self, key)
            target.clear()
            target.extend(state[key])
        if not exact_restore:
            self.current_returns.zero_()
            self.current_lengths.zero_()
            self.episode_ids.fill_(-1)
        self.total_time = state["total_time"]
        self.episode_extras.clear()
        self._purge_step = iteration + 1

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
