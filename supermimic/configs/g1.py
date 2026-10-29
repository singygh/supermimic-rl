"""Practical G1 defaults and explicit paper-sized network/configuration reference."""
from .algorithm import AlgorithmConfig, RunnerConfig


def g1_config(profile="debug", seed=1):
    if profile not in ("debug", "single_motion", "paper"):
        raise ValueError("profile must be debug, single_motion or paper")
    if profile == "single_motion":
        # Engineering starting point for a 16 GB GPU, not a tuned convergence
        # claim. Fit dynamics before exposing the actor to imagined gradients.
        cfg = RunnerConfig(algorithm=AlgorithmConfig(model_batch_size=512, imagination_batch_size=128,
            world_model_batches_per_update=10, actor_warmup_updates=100,
            actor_lr=1e-4, critic_lr=2e-4, world_model_lr=3e-4,
            actor_betas=(.9, .95), actor_grad_value=.1, observation_noise_std=0.),
            num_steps_per_env=16, replay_capacity=256, hidden_dims=(512, 256, 128),
            seed=seed, save_interval=250, log_interval=10)
        return cfg, {"actor": cfg.hidden_dims, "encoder": (512, 256, 128),
                     "critic": (256, 128), "world_model": (512, 512, 512)}
    paper = profile == "paper"
    cfg = RunnerConfig(
        algorithm=AlgorithmConfig(model_batch_size=2048 if paper else 128,
            imagination_batch_size=4096 if paper else 64,
            world_model_batches_per_update=10 if paper else 2,
            actor_lr=2e-4, critic_lr=2e-4, world_model_lr=2e-4,
            actor_betas=(.9, .95), actor_grad_value=.1, observation_noise_std=0.,
            lr_decay_updates=250000 if paper else None),
        num_steps_per_env=16, replay_capacity=1000 if paper else 128,
        hidden_dims=(2048, 1024, 512, 256) if paper else (256, 128), seed=seed,
    )
    architecture = {"actor": cfg.hidden_dims,
                    "encoder": (2048, 1024, 512, 256) if paper else (256, 128),
                    "critic": (256, 128), "world_model": (2048, 2048, 2048) if paper else (256, 256, 256)}
    return cfg, architecture
