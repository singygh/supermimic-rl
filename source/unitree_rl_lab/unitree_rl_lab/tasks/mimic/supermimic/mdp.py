"""Isaac Lab callbacks for the shared differentiable G1 task."""
import torch

from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand
from supermimic.envs.isaaclab_adapter import get_runtime


class SuperMimicMotionCommand(MotionCommand):
    def _adaptive_sampling(self, env_ids):
        play_frame = getattr(self._env.cfg, "supermimic_start_frame", -1)
        if play_frame >= 0:
            if play_frame >= self.motion.time_step_total - 1:
                raise ValueError("Playback start frame must be before the last reference frame")
            self.time_steps[env_ids] = play_frame
            return
        # Initial single-motion integration: uniform starts, explicitly not the
        # paper's return-based mixture or the official failure-count heuristic.
        self.time_steps[env_ids] = torch.randint(self.motion.time_step_total - 1,
                                                (len(env_ids),), device=self.device)

    def _update_command(self):
        # Motion-end timeout must reset before a wrap could teleport the robot.
        # Guard also covers initialization and a reset sampled near the last frame.
        self.time_steps.clamp_(max=self.motion.time_step_total - 2)
        super()._update_command()


def tracking_reward_and_capture(env):
    runtime = get_runtime(env)
    state = runtime.state()
    frame = runtime.command.time_steps.clone()
    action = env.action_manager.action.clone()
    runtime.terminal = (state.clone(), frame, action,
                        env.termination_manager.terminated.clone(), env.termination_manager.time_outs.clone())
    reward = runtime.task.reward(state, action, runtime.reference.at(frame))
    # RewardManager multiplies term weights by dt. Expose per-control-step
    # reward to both simulator collection and imagined trajectories.
    return reward / env.step_dt


def tracking_failure(env):
    runtime = get_runtime(env)
    return runtime.task.terminated(runtime.state(), runtime.reference.at(runtime.command.time_steps))


def motion_end(env):
    command = env.command_manager.get_term("motion")
    return command.time_steps >= command.motion.time_step_total - 1


def proprioception(env):
    # ObservationManager calls terms while determining dimensions, before startup
    # randomizers run. Do not cache a task/context from this initialization call.
    from supermimic.tasks.g1 import normalize_quat, quat_apply, quat_conjugate

    robot = env.scene["robot"]
    pelvis = robot.body_names.index("pelvis")
    inverse = quat_conjugate(normalize_quat(robot.data.body_quat_w[:, pelvis]))
    gravity = torch.zeros(env.num_envs, 3, device=env.device)
    gravity[:, 2] = -1
    return torch.cat((robot.data.joint_pos - robot.data.default_joint_pos, robot.data.joint_vel,
        quat_apply(inverse, robot.data.body_ang_vel_w[:, pelvis]), quat_apply(inverse, gravity),
        env.action_manager.action), -1)


def randomize_shared_friction(env, env_ids):
    """One mu per robot, common to all shapes, with material readback in runtime."""
    robot = env.scene["robot"]
    ids = torch.arange(env.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    material = robot.root_physx_view.get_material_properties().clone()
    # Bucketed to respect PhysX material limits at large environment counts.
    buckets = torch.linspace(.3, 1.2, 64, device="cpu")
    mu = buckets[torch.randint(64, (len(ids),))]
    material[ids, :, 0] = mu[:, None]
    material[ids, :, 1] = mu[:, None]
    material[ids, :, 2] = 0
    robot.root_physx_view.set_material_properties(material, ids)
