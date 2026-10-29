"""Play the newest trained G1 policy; restores actor weights, not training state."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/unitree_rl_lab"))


def main():
    from isaaclab.app import AppLauncher
    from supermimic.utils.checkpoints import (build_inference_actor, compatible_policy_spec,
                                            latest_checkpoint, load_policy_checkpoint)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="runs/g1_supermimic", help="Log root: select latest timestamped training run, then its latest model")
    parser.add_argument("--checkpoint", help="Explicit checkpoint overrides automatic selection")
    parser.add_argument("--task", help="Default: task recorded by training")
    parser.add_argument("--motion-file", help="Default: reference NPZ recorded by training")
    parser.add_argument("--robot-urdf", help="Override recorded/default G1 URDF")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0, help="0: until GUI closes (headless defaults to 1000)")
    parser.add_argument("--start-frame", type=int, default=0, help="Reference frame used on each reset")
    parser.add_argument("--randomize", action="store_true", help="Enable training DR and reset perturbations")
    parser.add_argument("--observation-noise", action="store_true")
    parser.add_argument("--real-time", action="store_true", help="Pace playback at the 50 Hz control rate")
    parser.add_argument("--fixed-camera", action="store_true", help="Disable following the first robot")
    parser.add_argument("--asset-cache", default="/tmp/supermimic-isaac-assets")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 0 or args.start_frame < 0:
        parser.error("num-envs must be positive; steps and start-frame must be nonnegative")
    try:
        checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else latest_checkpoint(args.log_dir)
        payload = load_policy_checkpoint(checkpoint)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    task = args.task or payload["task"]
    motion_arg = args.motion_file or payload["motion_file"]
    if not task or not motion_arg:
        parser.error("Checkpoint has no task/reference metadata; supply --task and --motion-file")
    if "SuperMimic" not in task or not payload["spec"].version.startswith("g1-"):
        parser.error("This play script requires a G1 SuperMimic task/checkpoint")
    motion = Path(motion_arg).expanduser()
    if not motion.is_absolute():
        motion = checkpoint.parent / motion
    motion = motion.resolve()
    if not motion.is_file():
        parser.error(f"Recorded motion file is unavailable: {motion}. Supply --motion-file.")
    if not args.motion_file and payload["motion_digest"]:
        import hashlib
        if hashlib.sha256(motion.read_bytes()).hexdigest() != payload["motion_digest"]:
            parser.error("Recorded motion file changed; use --motion-file to explicitly select a new reference")
    print(f"[INFO] Checkpoint: {checkpoint}\n[INFO] Training iteration: {payload['iteration']}\n"
          f"[INFO] Task: {task}\n[INFO] Motion: {motion}", flush=True)
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch
        import unitree_rl_lab.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from supermimic.envs.isaaclab_adapter import IsaacLabEnvAdapter

        torch.manual_seed(payload["seed"])
        env_cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
        env_cfg.seed = payload["seed"]
        env_cfg.commands.motion.motion_file = str(motion)
        env_cfg.supermimic_start_frame = args.start_frame
        env_cfg.scene.robot.spawn.usd_dir = str(Path(args.asset_cache).resolve())
        urdf = args.robot_urdf or payload["robot_urdf"]
        if urdf:
            env_cfg.scene.robot.spawn.asset_path = str(Path(urdf).expanduser().resolve())
        if not args.randomize:
            env_cfg.events.physics_material = None
            env_cfg.events.add_joint_default_pos = None
            env_cfg.events.base_com = None
            env_cfg.commands.motion.pose_range = {}
            env_cfg.commands.motion.velocity_range = {}
            env_cfg.commands.motion.joint_position_range = (0., 0.)
        env = gym.make(task, cfg=env_cfg)
        if not args.randomize:
            robot = env.unwrapped.scene["robot"]
            materials = robot.root_physx_view.get_material_properties().clone()
            materials[..., :2] = 1
            materials[..., 2] = 0
            robot.root_physx_view.set_material_properties(materials, torch.arange(args.num_envs, device="cpu"))
        adapter = IsaacLabEnvAdapter(env, observation_noise=args.observation_noise)
        if not compatible_policy_spec(payload["spec"], adapter.spec):
            raise ValueError("Playback task differs from training TaskSpec (dimensions/joint order/action scales/codec)")
        policy = build_inference_actor(payload, adapter.device)
        del payload  # No replay buffer, optimizer, world model or critic is constructed.
        steps = args.steps or (1000 if args.headless else 0)
        step_count, reward_sum, completed = 0, 0., 0
        failures, timeouts = 0, 0
        while app.is_running() and (steps == 0 or step_count < steps):
            started = time.perf_counter()
            with torch.inference_mode():
                view = adapter.get_step_view()
                result = adapter.step(policy(view.obs_history, view.goal))
                reward_sum += result.reward.mean().item()
                completed += (result.terminated | result.truncated).sum().item()
                failures += result.terminated.sum().item()
                timeouts += (result.truncated & ~result.terminated).sum().item()
            step_count += 1
            if not args.headless and not args.fixed_camera:
                position = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                env.unwrapped.sim.set_camera_view(position + [3., 3., 1.5], position)
            if args.real_time:
                remaining = adapter.spec.dt - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
        print(f"[INFO] Played {step_count} steps; mean reward {reward_sum / max(step_count, 1):.4f}; "
              f"completed episodes {completed}; failures {failures}; timeouts {timeouts}", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
