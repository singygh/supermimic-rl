import argparse
import json

import torch

from supermimic.configs import RunnerConfig
from supermimic.envs import ToyAdapter
from supermimic.models import WorldModel
from supermimic.models.heads import ToyObservation, ToyReward, ToyTermination
from supermimic.runners import SuperMimicRunner
from supermimic.utils.runs import create_run_directory, resolve_resume_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train SuperMimic on a vector toy task")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-dir", default="runs/toy", help="Log root; each launch creates a timestamped run")
    parser.add_argument("--run-name", help="Optional suffix for the timestamped run directory")
    parser.add_argument("--resume", help="Checkpoint file, run directory, or 'latest'")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--logger", choices=("tensorboard", "json"), default="tensorboard")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--quiet", action="store_true", help="Disable console training summaries")
    args = parser.parse_args()
    try:
        resume_path = resolve_resume_checkpoint(args.resume, args.log_dir)
        log_dir = create_run_directory(args.log_dir, args.run_name, resume_from=resume_path)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cfg = RunnerConfig(seed=args.seed, logger=args.logger, log_interval=args.log_interval,
                       console_log=not args.quiet)
    env = ToyAdapter(args.num_envs, args.device, noise_std=cfg.algorithm.observation_noise_std)
    model = WorldModel(env.spec, cfg.hidden_dims, ToyObservation(), ToyReward(), ToyTermination())
    runner = SuperMimicRunner(env, cfg, log_dir, args.device, world_model=model)
    if resume_path:
        print(f"[INFO] Resuming checkpoint: {resume_path}", flush=True)
        runner.load(resume_path)
    print(json.dumps(runner.learn(args.iterations), indent=2))
    print(json.dumps(runner.evaluate(), indent=2))


if __name__ == "__main__":
    main()
