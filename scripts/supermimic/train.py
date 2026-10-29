"""Launch Isaac Sim before importing robot tasks, then train G1 via model BPTT."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/unitree_rl_lab"))


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Unitree-G1-29dof-SuperMimic-Dance-102")
    parser.add_argument("--motion-file", required=True, help="Official converter NPZ at 50 Hz")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--profile", choices=("debug", "single_motion", "paper"), default="debug")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-dir", default="runs/g1_supermimic", help="Log root; each launch creates a timestamped run")
    parser.add_argument("--run-name", help="Optional suffix for the timestamped run directory")
    parser.add_argument("--resume", help="Checkpoint file, run directory, or 'latest'")
    parser.add_argument("--logger", choices=("tensorboard", "json"), default="tensorboard")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--quiet", action="store_true", help="Disable console training summaries")
    parser.add_argument("--no-dr", action="store_true", help="Disable startup DR for initial alignment checks")
    parser.add_argument("--no-observation-noise", action="store_true")
    parser.add_argument("--no-reset-perturbations", action="store_true", help="Initialize exactly on sampled reference frames")
    parser.add_argument("--robot-urdf", help="Override G1 asset; keep meshes adjacent to URDF")
    parser.add_argument("--checkpoint-dir", default="/tmp/supermimic-isaac-assets",
                        help="Writable USD conversion cache")
    parser.add_argument("--check-only", action="store_true", help="Exercise simulator/reset/model/gradients on a small run")
    parser.add_argument("--episode-length-s", type=float, help="Override time limit; short values exercise reset capture")
    parser.add_argument("--model-batch-size", type=int, help="Override world-model fitting batch size")
    parser.add_argument("--imagination-batch-size", type=int, help="Override BPTT batch size")
    parser.add_argument("--world-model-batches", type=int, help="Model optimizer steps per algorithm update")
    parser.add_argument("--actor-warmup-updates", type=int, help="Model-only updates before enabling actor/critic")
    parser.add_argument("--actor-lr", type=float)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--world-model-lr", type=float)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    motion = Path(args.motion_file).expanduser().resolve()
    if not motion.is_file():
        parser.error(f"Motion NPZ not found: {motion}. Run scripts/mimic/csv_to_npz.py first.")
    if "SuperMimic" not in args.task:
        parser.error("Use a registered SuperMimic G1 task, not a PPO task")
    from supermimic.utils.runs import create_run_directory, resolve_resume_checkpoint
    try:
        resume_path = resolve_resume_checkpoint(args.resume, args.log_dir)
        log_dir = create_run_directory(args.log_dir, args.run_name, resume_from=resume_path)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        import unitree_rl_lab.tasks  # noqa: F401

        from supermimic.configs.g1 import g1_config
        from supermimic.envs.isaaclab_adapter import IsaacLabEnvAdapter
        from supermimic.models import Actor, Critic
        from supermimic.models.g1_world_model import G1WorldModel
        from supermimic.runners import SuperMimicRunner

        torch.manual_seed(args.seed)
        cfg, architecture = g1_config(args.profile, args.seed)
        cfg.logger, cfg.console_log = args.logger, not args.quiet
        cfg.log_interval = args.log_interval
        for argument, field in (("model_batch_size", "model_batch_size"),
                ("imagination_batch_size", "imagination_batch_size"),
                ("world_model_batches", "world_model_batches_per_update"),
                ("actor_warmup_updates", "actor_warmup_updates"), ("actor_lr", "actor_lr"),
                ("critic_lr", "critic_lr"), ("world_model_lr", "world_model_lr")):
            value = getattr(args, argument)
            if value is not None:
                setattr(cfg.algorithm, field, value)
        if args.check_only:
            cfg.algorithm.actor_warmup_updates = 0  # short check must exercise BPTT too
        cfg.algorithm.__post_init__()
        cfg.__post_init__()
        env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
        env_cfg.seed = args.seed
        env_cfg.commands.motion.motion_file = str(motion)
        env_cfg.scene.robot.spawn.usd_dir = str(Path(args.checkpoint_dir).resolve())
        if args.robot_urdf:
            env_cfg.scene.robot.spawn.asset_path = str(Path(args.robot_urdf).expanduser().resolve())
        if args.episode_length_s is not None:
            env_cfg.episode_length_s = args.episode_length_s
        if args.no_reset_perturbations:
            env_cfg.commands.motion.pose_range = {}
            env_cfg.commands.motion.velocity_range = {}
            env_cfg.commands.motion.joint_position_range = (0., 0.)
        if args.no_dr:
            env_cfg.events.physics_material = None
            env_cfg.events.add_joint_default_pos = None
            env_cfg.events.base_com = None
        env = gym.make(args.task, cfg=env_cfg)
        # Save the resolved simulator configuration alongside the algorithm config.
        from isaaclab.utils.io import dump_yaml
        dump_yaml(str(log_dir / "env_config.yaml"), env_cfg)
        # With DR off, force the same valid context contract (mu=1, restitution=0).
        if args.no_dr:
            robot = env.unwrapped.scene["robot"]
            materials = robot.root_physx_view.get_material_properties().clone()
            materials[..., :2] = 1
            materials[..., 2] = 0
            robot.root_physx_view.set_material_properties(materials, torch.arange(args.num_envs, device="cpu"))
        adapter = IsaacLabEnvAdapter(env, observation_noise=not args.no_observation_noise)
        spec = adapter.spec
        actor = Actor(spec, architecture["actor"], architecture["encoder"], latent_dim=32)
        critic = Critic(spec, architecture["critic"])
        model = G1WorldModel(adapter.task, architecture["world_model"])
        runner = SuperMimicRunner(adapter, cfg, log_dir, adapter.device, world_model=model,
            actor=actor, critic=critic, goal_provider=adapter.reference,
            observation_noise=adapter.add_observation_noise)
        if resume_path:
            print(f"[INFO] Resuming checkpoint: {resume_path}", flush=True)
            runner.load(resume_path)
        (log_dir / "task_spec.json").write_text(json.dumps(asdict(spec), indent=2))
        (log_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))
        import importlib.metadata as metadata
        versions = {name: metadata.version(name) for name in ("torch", "isaaclab", "isaacsim")
                    if metadata.packages_distributions().get(name)}
        (log_dir / "run_metadata.json").write_text(json.dumps({"task": args.task,
            "motion_file": str(motion), "motion_sha256": adapter.runtime.motion_digest,
            "profile": args.profile, "no_dr": args.no_dr, "versions": versions,
            "no_reset_perturbations": args.no_reset_perturbations,
            "architecture": architecture, "exact_simulator_resume": False}, indent=2))
        metrics = runner.learn(2 if args.check_only else args.iterations)
        if args.check_only:
            if runner.alg.num_updates == 0:
                raise RuntimeError("No valid 8-step windows; increase the episode length")
            if not metrics.get("actor_grad_norm", 0) > 0:
                raise RuntimeError("Actor received no BPTT gradient")
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Actor/critic leaked gradients into world model")
        print(json.dumps(metrics, indent=2))
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
