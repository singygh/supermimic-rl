from dataclasses import asdict
import os
from pathlib import Path
import random
import time
import tempfile
import traceback

import torch

from supermimic.algorithms import SuperMimicAlgorithm
from supermimic.configs import AlgorithmConfig, RunnerConfig
from supermimic.envs import ToyAdapter, stationary_goal
from supermimic.models import Actor, Critic, WorldModel
from supermimic.storage import ReplayBuffer
from supermimic.utils.logger import LOG_CONFIG_FIELDS, TrainingLogger


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


class SuperMimicRunner:
    """RSL-RL-style lifecycle, backed by replay/model learning rather than PPO.

    Inject task-specific WorldModel and goal_provider for robot/motion tasks.
    Exact resume requires the adapter to serialize its simulator state as well.
    """

    def __init__(self, env, train_cfg=None, log_dir=None, device="cpu", *,
                 world_model=None, actor=None, critic=None, goal_provider=None, observation_noise=None):
        if isinstance(train_cfg, dict):
            train_cfg = dict(train_cfg)
            algorithm = train_cfg.get("algorithm", {})
            train_cfg["algorithm"] = AlgorithmConfig(**algorithm) if isinstance(algorithm, dict) else algorithm
            train_cfg = RunnerConfig(**train_cfg)
        self.cfg = train_cfg or RunnerConfig()
        self.env, self.device = env, torch.device(device)
        self.log_dir = Path(log_dir) if log_dir else None
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        env.reset()
        if goal_provider is None:
            if not isinstance(env, ToyAdapter):
                raise ValueError("Non-toy tasks must supply a reference goal_provider(seed, offset)")
            goal_provider = stationary_goal
        spec = env.spec
        storage = ReplayBuffer(self.cfg.replay_capacity, env.num_envs, device,
                               goal_resolver=getattr(env, "replay_goal", None))
        self.alg = SuperMimicAlgorithm(
            actor if actor is not None else Actor(spec, self.cfg.hidden_dims),
            critic if critic is not None else Critic(spec, self.cfg.hidden_dims),
            world_model if world_model is not None else WorldModel(spec, self.cfg.hidden_dims),
            storage, goal_provider, self.cfg.algorithm, device, observation_noise,
        )
        self.current_learning_iteration = 0
        self.last_metrics = {}
        self.logger = TrainingLogger(self.log_dir, self.cfg, env.num_envs, device, env.spec)

    def learn(self, num_learning_iterations):
        if num_learning_iterations < 0:
            raise ValueError("Iteration count must be nonnegative")
        if not num_learning_iterations:
            return self.last_metrics
        self.alg.train_mode()
        total_iterations = self.current_learning_iteration + num_learning_iterations
        try:
            self.logger.start()
            call_started = time.perf_counter()
            for call_iteration in range(1, num_learning_iterations + 1):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                metrics = self.alg.collect(self.env, self.cfg.num_steps_per_env, self.logger.process_env_step)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                collected = time.perf_counter()
                valid = self.alg.storage.valid_starts(self.cfg.algorithm.model_horizon)
                candidates = max(0, self.alg.storage.size - self.cfg.algorithm.model_horizon + 1) * self.env.num_envs
                metrics["valid_window_fraction"] = len(valid) / max(candidates, 1)
                if len(valid):
                    metrics.update(self.alg.update())
                    metrics.update(self.alg.evaluate_model())
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                metrics["collect_seconds"] = collected - started
                metrics["update_seconds"] = time.perf_counter() - collected
                metrics["collect_transitions_per_second"] = (
                    self.cfg.num_steps_per_env * self.env.num_envs / max(collected - started, 1e-9))
                for name in ("actor", "critic", "world_model"):
                    metrics[f"{name}_learning_rate"] = getattr(self.alg, f"{name}_optimizer").param_groups[0]["lr"]
                metrics["action_noise_std"] = self.cfg.algorithm.action_noise_std
                self.current_learning_iteration += 1
                metrics["iteration"] = self.current_learning_iteration
                self.last_metrics = self.logger.log(metrics, total_iterations=total_iterations,
                    call_elapsed=time.perf_counter() - call_started, call_iterations=call_iteration,
                    total_env_steps=self.alg.num_collected_steps, num_updates=self.alg.num_updates,
                    replay_transitions=self.alg.storage.size * self.env.num_envs,
                    final=call_iteration == num_learning_iterations)
                if self.log_dir and self.current_learning_iteration % self.cfg.save_interval == 0:
                    self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")
            if self.log_dir:
                self.save(self.log_dir / "model_latest.pt")
        except BaseException:
            if self.log_dir:
                try:
                    self.log_dir.mkdir(parents=True, exist_ok=True)
                    with (self.log_dir / "error.log").open("a", encoding="utf-8") as stream:
                        stream.write(traceback.format_exc() + "\n")
                except OSError:
                    pass  # Preserve the original exception if logging also fails.
            raise
        finally:
            self.logger.close()
        return self.last_metrics

    def save(self, path, infos=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving checkpoint to {path} (replay snapshot on CPU)...", flush=True)
        started = time.perf_counter()
        state = {"format_version": 1, "algorithm": self.alg.state_dict(),
                 "replay": self.alg.storage.state_dict(), "env": self.env.state_dict(),
                 "rng": rng_state(), "iteration": self.current_learning_iteration,
                 "logger": self.logger.state_dict(),
                 "policy_config": self.alg.actor.get_config() if hasattr(self.alg.actor, "get_config") else None,
                 "cfg": asdict(self.cfg), "task_spec": asdict(self.env.spec), "infos": infos}
        # Atomically publish only complete files; a failed write leaves any
        # previous checkpoint intact and cannot confuse automatic play loading.
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary_path = Path(stream.name)
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        if self.logger.writer is not None:
            self.logger.writer.flush()
        print(f"[INFO] Checkpoint saved: {path} ({time.perf_counter() - started:.1f}s)", flush=True)

    def load(self, path, load_optimizer=True):
        """Load trusted local checkpoints only (optimizer and Python RNG payload)."""
        # Keep replay on CPU until copying into its destination buffer, instead
        # of loading it onto CUDA then allocating another complete GPU replay.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["format_version"] != 1 or state["task_spec"] != asdict(self.env.spec):
            raise ValueError("Checkpoint format/task spec mismatch")
        saved_cfg = {key: value for key, value in state["cfg"].items() if key not in LOG_CONFIG_FIELDS}
        saved_cfg["algorithm"] = asdict(AlgorithmConfig(**saved_cfg["algorithm"]))
        current_cfg = {key: value for key, value in asdict(self.cfg).items() if key not in LOG_CONFIG_FIELDS}
        if saved_cfg != current_cfg:
            raise ValueError("Training configuration mismatch; use actor.state_dict() for policy transfer")
        self.alg.load_state_dict(state["algorithm"], load_optimizer)
        self.alg.storage.load_state_dict(state["replay"])
        self.env.load_state_dict(state["env"])
        self.current_learning_iteration = state["iteration"]
        logger_state = state.get("logger")
        if logger_state is None:  # backward compatible with pre-logger checkpoints
            logger_state = TrainingLogger(self.log_dir, self.cfg, self.env.num_envs,
                                          self.device, self.env.spec).state_dict()
        self.logger.load_state_dict(logger_state, self.current_learning_iteration,
                                    exact_restore=getattr(self.env, "supports_exact_restore", True))
        restore_rng(state["rng"])
        return state["infos"]

    def get_inference_policy(self, device=None):
        """Return an independent deployable actor accepting (history, goal)."""
        from copy import deepcopy

        policy = deepcopy(self.alg.actor).to(device or self.device).eval()
        policy.requires_grad_(False)
        return policy.act_inference

    @torch.no_grad()
    def evaluate(self, n_steps=128, seed=0):
        """Deterministic actions; noise/DR settings remain explicit adapter choices.

        Restores simulator and RNG state so evaluation does not change training.
        Reports finite-horizon reward, not a benchmark motion-tracking score.
        """
        if n_steps < 1:
            raise ValueError("Evaluation steps must be positive")
        if not getattr(self.env, "supports_exact_restore", True):
            raise RuntimeError("This simulator cannot restore solver state; evaluate in a separate environment")
        env_state, random_state = self.env.state_dict(), rng_state()
        actor_mode = self.alg.actor.training
        try:
            torch.manual_seed(seed)
            self.env.reset()
            self.alg.actor.eval()
            reward_sum, completed = 0.0, 0
            for _ in range(n_steps):
                view = self.env.get_step_view().to(self.device)
                step = self.env.step(self.alg.act(view.obs_history, view.goal).to(self.env.device))
                reward_sum += step.reward.mean().item()
                completed += (step.terminated | step.truncated).sum().item()
            return {"mean_reward": reward_sum / n_steps, "completed_episodes": completed}
        finally:
            self.env.load_state_dict(env_state)
            restore_rng(random_state)
            self.alg.actor.train(actor_mode)
