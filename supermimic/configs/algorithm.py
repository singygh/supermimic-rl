from dataclasses import dataclass, field


@dataclass
class AlgorithmConfig:
    model_horizon: int = 8
    imagination_horizon: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    model_batch_size: int = 64
    imagination_batch_size: int = 64
    updates_per_iteration: int = 1
    world_model_batches_per_update: int = 1
    actor_warmup_updates: int = 0
    critic_epochs: int = 5
    critic_mini_batches: int = 4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    world_model_lr: float = 1e-3
    max_grad_norm: float = 1.0
    actor_betas: tuple[float, float] = (0.9, 0.999)
    actor_grad_value: float | None = None
    lr_decay_updates: int | None = None
    min_lr_ratio: float = .03
    action_noise_std: float = 0.1
    observation_noise_std: float = 0.01
    state_loss_weight: float = 1.0
    observation_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    termination_loss_weight: float = 1.0

    def __post_init__(self):
        for key in ("model_horizon", "imagination_horizon", "model_batch_size", "imagination_batch_size",
                    "updates_per_iteration", "world_model_batches_per_update", "critic_epochs", "critic_mini_batches"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.lam <= 1:
            raise ValueError("gamma and lam must be in [0, 1]")
        if self.actor_warmup_updates < 0:
            raise ValueError("actor_warmup_updates must be nonnegative")
        if len(self.actor_betas) != 2 or any(not 0 <= b < 1 for b in self.actor_betas):
            raise ValueError("Invalid Adam betas")
        if self.actor_grad_value is not None and self.actor_grad_value <= 0:
            raise ValueError("actor_grad_value must be positive")
        if self.lr_decay_updates is not None and self.lr_decay_updates < 1:
            raise ValueError("lr_decay_updates must be positive")
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in (0, 1]")
        for key in ("actor_lr", "critic_lr", "world_model_lr", "max_grad_norm"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in ("action_noise_std", "observation_noise_std", "state_loss_weight",
                    "observation_loss_weight", "reward_loss_weight", "termination_loss_weight"):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")


@dataclass
class RunnerConfig:
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    num_steps_per_env: int = 16
    replay_capacity: int = 2048  # vector steps (total capacity is this * num_envs)
    hidden_dims: tuple[int, ...] = (128, 128)
    seed: int = 1
    save_interval: int = 100
    logger: str = "tensorboard"
    console_log: bool = True
    log_interval: int = 1
    episode_stats_window: int = 100
    log_flush_secs: int = 10

    def __post_init__(self):
        if self.num_steps_per_env < 1 or self.replay_capacity < self.algorithm.model_horizon:
            raise ValueError("Collect steps must be positive and replay must fit a model window")
        if self.save_interval < 1 or not self.hidden_dims or min(self.hidden_dims) < 1:
            raise ValueError("Invalid save interval or hidden dimensions")
        if self.logger not in ("tensorboard", "json"):
            raise ValueError("logger must be tensorboard or json")
        if min(self.log_interval, self.episode_stats_window, self.log_flush_secs) < 1:
            raise ValueError("Log interval, episode window and flush interval must be positive")
