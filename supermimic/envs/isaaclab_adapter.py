"""G1 ManagerBasedRLEnv adapter. Importing this module does not launch Isaac Sim.

The dedicated reward callback records a transition before ManagerBasedRLEnv
resets actors or advances MotionCommand. Other task configurations are rejected.
"""
import hashlib
from pathlib import Path
import warnings

import torch

from supermimic.tasks.g1 import G1MotionReference, G1TrackingTask
from .protocol import EnvStep, StepView


def capture_state(env, task, command):
    robot = env.scene["robot"]
    bodies = command.body_indexes
    return task.pack_state(robot.data.joint_pos, robot.data.joint_vel,
        env.action_manager.action, env.action_manager.prev_action,
        robot.data.body_pos_w[:, bodies] - env.scene.env_origins[:, None],
        robot.data.body_quat_w[:, bodies], robot.data.body_lin_vel_w[:, bodies],
        robot.data.body_ang_vel_w[:, bodies])


class G1Runtime:
    """Initialized after simulator startup DR; actual physical values are read back."""

    def __init__(self, env):
        self.env = env
        self.command = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        if self.command.cfg.anchor_body_name != "pelvis":
            raise ValueError("SuperMimic G1 config must track pelvis as root")
        nominal = getattr(robot.data, "default_joint_pos_nominal", robot.data.default_joint_pos[0]).clone()
        if nominal.ndim == 2:
            nominal = nominal[0]
        scale = torch.as_tensor(env.action_manager.get_term("JointPositionAction")._scale,
                                device=env.device).expand(env.num_envs, 29)[0]
        self.task = G1TrackingTask(self.command.cfg.body_names, robot.joint_names, nominal,
                                   robot.data.soft_joint_pos_limits[0], scale, env.step_dt).to(env.device)
        material = robot.root_physx_view.get_material_properties().to(env.device)
        friction = material[:, :1, :1]
        if not torch.allclose(material[..., :2], friction.expand_as(material[..., :2])):
            raise ValueError("Expected shared static/dynamic friction per robot; use SuperMimic DR config")
        if not torch.allclose(material[..., 2], torch.zeros_like(material[..., 2])):
            raise ValueError("Restitution must be zero or explicitly added to context")
        torso = robot.body_names.index("torso_link")
        com = robot.root_physx_view.get_coms().to(env.device)[:, torso, :3]
        self.context = torch.cat((material[:, 0, :1], robot.data.default_joint_pos - nominal, com), -1).clone()
        motion = self.command.motion
        # Updated official converter records ordering; old NPZ files retain the
        # official articulation-order contract and cannot be independently audited.
        import numpy as np
        with np.load(self.command.cfg.motion_file) as data:
            for key, expected in (("joint_names", robot.joint_names), ("body_names", robot.body_names)):
                if key in data and list(data[key]) != list(expected):
                    raise ValueError(f"Reference {key} order differs from the G1 articulation; reconvert the CSV")
                if key not in data:
                    warnings.warn(f"Motion NPZ lacks {key}; assuming official converter articulation order",
                                  stacklevel=2)
        self.reference = G1MotionReference(self.task, motion.joint_pos, motion.joint_vel,
            motion.body_pos_w, motion.body_quat_w, motion.body_lin_vel_w, motion.body_ang_vel_w,
            float(motion.fps.reshape(-1)[0]))
        self.motion_digest = hashlib.sha256(Path(self.command.cfg.motion_file).read_bytes()).hexdigest()
        self.terminal = None

    def state(self):
        return capture_state(self.env, self.task, self.command)


def get_runtime(env):
    if not hasattr(env, "_supermimic_runtime"):
        env._supermimic_runtime = G1Runtime(env)
    return env._supermimic_runtime


class IsaacLabEnvAdapter:
    supports_exact_restore = False  # PhysX solver/contact caches are not serialized.

    def __init__(self, env, observation_noise=True):
        self.env = env.unwrapped
        if not getattr(self.env.cfg, "supermimic_capture", False):
            raise ValueError("Use the registered G1 SuperMimic task with the pre-reset reward capture hook")
        self.num_envs, self.device = self.env.num_envs, str(self.env.device)
        self.runtime = get_runtime(self.env)
        self.spec, self.task = self.runtime.task.spec, self.runtime.task
        self.reference = self.runtime.reference
        self.replay_goal = self.reference.at
        self.noise = observation_noise
        self.episode_id = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.history = torch.zeros(self.num_envs, 10, 93, device=self.device)
        self.reset()

    def add_observation_noise(self, clean):
        return self.task.add_observation_noise(clean) if self.noise else clean

    def _view(self, state, history, frame_id):
        clean = self.task.observe_clean(state, self.runtime.context)
        return StepView(state.clone(), clean, history.clone(), self.reference.at(frame_id),
                        self.runtime.context.clone(), self.episode_id.clone(),
                        torch.zeros_like(frame_id), frame_id.clone())

    def get_step_view(self):
        return self._view(self.runtime.state(), self.history, self.runtime.command.time_steps)

    @torch.no_grad()
    def reset(self, ids=None):
        # Public Isaac Lab reset supports env_ids. No private reset call is needed.
        self.env.reset(env_ids=ids)
        ids = torch.arange(self.num_envs, device=self.device) if ids is None else ids
        self.episode_id[ids] += 1
        clean = self.task.observe_clean(self.runtime.state(), self.runtime.context)
        self.history[ids] = self.add_observation_noise(clean[ids])[:, None].expand(-1, 10, -1)
        return self.get_step_view()

    @torch.no_grad()
    def step(self, action_exec):
        self.runtime.terminal = None
        action = action_exec.to(self.device).clamp(-1, 1)
        _, reward, terminated, truncated, extras = self.env.step(action)
        if self.runtime.terminal is None:
            raise RuntimeError("Pre-reset capture was not called; refuse corrupted replay transitions")
        state, frame, executed, captured_terminated, captured_truncated = self.runtime.terminal
        if not torch.equal(captured_terminated, terminated) or not torch.equal(captured_truncated, truncated):
            raise RuntimeError("Isaac Lab termination lifecycle differs from the supported contract")
        clean = self.task.observe_clean(state, self.runtime.context)
        terminal_history = torch.cat((self.history[:, 1:], self.add_observation_noise(clean)[:, None]), 1)
        terminal = self._view(state, terminal_history, frame)
        # This check audits RewardManager's dt scaling and reference timing every step.
        expected = self.task.reward(state, executed, terminal.goal)
        if not torch.allclose(reward, expected, atol=2e-5, rtol=2e-5):
            raise RuntimeError("Simulator/model reward mismatch (check reference timing and dt scaling)")
        self.history = terminal_history
        done = terminated | truncated
        self.episode_id[done] += 1
        if done.any():
            reset_clean = self.task.observe_clean(self.runtime.state(), self.runtime.context)
            self.history[done] = self.add_observation_noise(reset_clean[done])[:, None].expand(-1, 10, -1)
        return EnvStep(terminal, self.get_step_view(), executed, reward.clone(),
                       terminated.clone(), truncated.clone(), extras)

    def get_context(self):
        return self.runtime.context.clone()

    def get_full_state(self):
        return self.runtime.state().clone()

    def state_dict(self):
        return {"exact_restore": False, "episode_id": self.episode_id.clone(),
                "motion_digest": self.runtime.motion_digest, "context": self.runtime.context.clone(),
                "motion_file": self.runtime.command.cfg.motion_file,
                "task_id": getattr(getattr(self.env, "spec", None), "id", None),
                "robot_urdf": self.env.cfg.scene.robot.spawn.asset_path,
                "noise": self.noise}

    def load_state_dict(self, state):
        if state["motion_digest"] != self.runtime.motion_digest or state["noise"] != self.noise:
            raise ValueError("Motion file or observation noise configuration changed")
        if not torch.equal(state["context"].to(self.device), self.runtime.context):
            raise ValueError("Startup physics DR differs; reuse the same simulator seed/configuration")
        warnings.warn("PhysX resume starts new episodes; solver state is not restored", stacklevel=2)
        # Keep old replay valid, while preventing new data from joining old episodes.
        self.episode_id = state["episode_id"].to(self.device).clone()
        self.reset()
