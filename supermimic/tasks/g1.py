"""G1 tracking contract. Pure PyTorch: shared by Isaac Lab and imagination.

Reference feature layout and residual state codec are engineering choices, since
the paper does not publish their complete tensor layout. Quaternions are wxyz.
"""
import torch
from torch import nn

from supermimic.configs import TaskSpec


def normalize_quat(q):
    norm = q.norm(dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    normalized = torch.where(norm > 1e-8, q / norm.clamp_min(1e-8), identity)
    return torch.where(normalized[..., :1] < 0, -normalized, normalized)


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_mul(a, b):
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    return torch.cat((w, v), -1)


def quat_apply(q, v):
    qv, v = torch.broadcast_tensors(q[..., 1:], v)
    uv = 2 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * uv + torch.cross(qv, uv, dim=-1)


def quat_distance_squared(a, b):
    rel = quat_mul(normalize_quat(a), quat_conjugate(normalize_quat(b)))
    return (2 * torch.atan2(rel[..., 1:].norm(dim=-1), rel[..., 0].abs().clamp_min(1e-8))).square()


def yaw_quat(q):
    w, x, y, z = normalize_quat(q).unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    zero = torch.zeros_like(yaw)
    return torch.stack(((yaw / 2).cos(), zero, zero, (yaw / 2).sin()), -1)


class G1TrackingTask(nn.Module):
    num_joints = 29
    future_frames = 10

    def __init__(self, body_names, joint_names, nominal_joint_pos, joint_limits, action_scales, dt=.02):
        super().__init__()
        if len(joint_names) != 29 or not body_names or body_names[0] != "pelvis":
            raise ValueError("G1 requires 29 articulation-order joints and pelvis first in tracked bodies")
        self.body_names = tuple(body_names)
        self.num_bodies = len(body_names)
        sizes = {"q": 29, "qd": 29, "action": 29, "previous_action": 29,
                 "pos": 3 * self.num_bodies, "quat": 4 * self.num_bodies,
                 "lin_vel": 3 * self.num_bodies, "ang_vel": 3 * self.num_bodies}
        self.slices = {}
        start = 0
        for key, size in sizes.items():
            self.slices[key] = slice(start, start + size)
            start += size
        self.frame_dim = 58 + 13 * self.num_bodies
        self.spec = TaskSpec(state_dim=start, obs_dim=93, action_dim=29,
            goal_dim=self.frame_dim * self.future_frames, context_dim=33, dt=dt,
            version="g1-isaaclab-tracking-v2", state_layout=str(sizes),
            frame_convention="environment-local world positions, world velocities, wxyz quaternion; pelvis root",
            action_scale=1., joint_names=tuple(joint_names), body_names=tuple(body_names),
            action_scales=tuple(float(x) for x in action_scales))
        self.register_buffer("nominal_joint_pos", torch.as_tensor(nominal_joint_pos).clone())
        self.register_buffer("joint_limits", torch.as_tensor(joint_limits).clone())
        scale = torch.ones(start, device=self.nominal_joint_pos.device)
        scale[self.slices["qd"]] = 10.
        scale[self.slices["lin_vel"]] = 5.
        scale[self.slices["ang_vel"]] = 10.
        self.register_buffer("state_scale", scale)
        self.register_buffer("noise_amplitude", torch.tensor([.01] * 29 + [.5] * 29 + [.2] * 3
                                                              + [.05] * 3 + [0.] * 29,
                                                              device=scale.device))

    def pack_state(self, q, qd, action, previous_action, pos, quat, lin_vel, ang_vel):
        return torch.cat((q, qd, action, previous_action, pos.flatten(1),
                          normalize_quat(quat).flatten(1), lin_vel.flatten(1), ang_vel.flatten(1)), -1)

    def unpack_state(self, state):
        result = {key: state[..., index] for key, index in self.slices.items()}
        for key in ("pos", "quat", "lin_vel", "ang_vel"):
            result[key] = result[key].reshape(*state.shape[:-1], self.num_bodies, 4 if key == "quat" else 3)
        return result

    def unpack_goal(self, goal):
        frame = goal[..., :self.frame_dim]
        b = self.num_bodies
        return {"q": frame[..., :29], "qd": frame[..., 29:58],
                "pos": frame[..., 58:58+3*b].reshape(-1, b, 3),
                "quat": frame[..., 58+3*b:58+7*b].reshape(-1, b, 4),
                "lin_vel": frame[..., 58+7*b:58+10*b].reshape(-1, b, 3),
                "ang_vel": frame[..., 58+10*b:].reshape(-1, b, 3)}

    def observe_clean(self, state, context):
        s = self.unpack_state(state)
        inverse = quat_conjugate(normalize_quat(s["quat"][:, 0]))
        gravity = torch.zeros_like(s["ang_vel"][:, 0])
        gravity[:, 2] = -1
        # Calibration affects both action offset and measured relative joint pose.
        q0 = self.nominal_joint_pos + context[:, 1:30]
        return torch.cat((s["q"] - q0, s["qd"], quat_apply(inverse, s["ang_vel"][:, 0]),
                          quat_apply(inverse, gravity), s["action"]), -1)

    def add_observation_noise(self, clean):
        return clean + (2 * torch.rand_like(clean) - 1) * self.noise_amplitude

    def reward(self, next_state, action, goal):
        s, g = self.unpack_state(next_state), self.unpack_goal(goal)
        rotation = yaw_quat(quat_mul(s["quat"][:, 0], quat_conjugate(g["quat"][:, 0])))
        anchor = torch.cat((s["pos"][:, 0, :2], g["pos"][:, 0, 2:3]), -1)
        desired_pos = anchor[:, None] + quat_apply(rotation[:, None], g["pos"] - g["pos"][:, :1])
        desired_quat = quat_mul(rotation[:, None].expand(-1, self.num_bodies, -1), g["quat"])
        reward = .5 * torch.exp(-(s["pos"][:, 0] - g["pos"][:, 0]).square().sum(-1) / .3**2)
        reward = reward + .5 * torch.exp(-quat_distance_squared(s["quat"][:, 0], g["quat"][:, 0]) / .4**2)
        reward = reward + torch.exp(-(s["pos"] - desired_pos).square().sum(-1).mean(-1) / .3**2)
        reward = reward + torch.exp(-quat_distance_squared(s["quat"], desired_quat).mean(-1) / .4**2)
        reward = reward + torch.exp(-(s["lin_vel"] - g["lin_vel"]).square().sum(-1).mean(-1))
        reward = reward + torch.exp(-(s["ang_vel"] - g["ang_vel"]).square().sum(-1).mean(-1) / 3.14**2)
        violation = ((self.joint_limits[:, 0] - s["q"]).clamp_min(0)
                     + (s["q"] - self.joint_limits[:, 1]).clamp_min(0)).sum(-1)
        # Paper Table A.2 terms are multiplied by control period (Appendix G,
        # below Table A.6). Both real collection and imagination use this scale.
        return self.spec.dt * (reward - .1 * (action - s["previous_action"]).square().sum(-1) - 2 * violation)

    def termination_errors(self, next_state, goal):
        s, g = self.unpack_state(next_state), self.unpack_goal(goal)
        height = (s["pos"][:, 0, 2] - g["pos"][:, 0, 2]).abs()
        gravity = torch.zeros_like(s["pos"][:, 0])
        gravity[:, 2] = -1
        actual = quat_apply(quat_conjugate(normalize_quat(s["quat"][:, 0])), gravity)[:, 2]
        desired = quat_apply(quat_conjugate(normalize_quat(g["quat"][:, 0])), gravity)[:, 2]
        return height, (actual - desired).abs()

    def terminated(self, state, goal):
        height, tilt = self.termination_errors(state, goal)
        return (height > .25) | (tilt > .8)

    def termination_prob(self, state, goal):
        height, tilt = self.termination_errors(state, goal)
        # Engineering smoothing of the paper's hard thresholds, kept differentiable.
        return 1 - (1 - torch.sigmoid((height - .25) / .02)) * (1 - torch.sigmoid((tilt - .8) / .05))


class G1MotionReference:
    """Single-clip reference in articulation order, sampled at control frequency."""

    def __init__(self, task, joint_pos, joint_vel, pos, quat, lin_vel, ang_vel, fps):
        if abs(float(fps) * task.spec.dt - 1) > 1e-5:
            raise ValueError("Reference FPS must equal control frequency; convert CSV at 50 Hz")
        if joint_pos.ndim != 2 or joint_pos.shape[-1] != 29 or len(joint_pos) < 2:
            raise ValueError("Reference must contain at least 2 frames and 29 joints")
        count = len(joint_pos)
        for value, shape in ((joint_vel, (count, 29)), (pos, (count, task.num_bodies, 3)),
                             (quat, (count, task.num_bodies, 4)), (lin_vel, (count, task.num_bodies, 3)),
                             (ang_vel, (count, task.num_bodies, 3))):
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid motion tensor; expected {shape}")
        if not torch.isfinite(joint_pos).all() or (quat.norm(dim=-1) < 1e-6).any():
            raise ValueError("Motion contains invalid positions/quaternions")
        self.task = task
        self.frames = torch.cat((joint_pos, joint_vel, pos.flatten(1), normalize_quat(quat).flatten(1),
                                 lin_vel.flatten(1), ang_vel.flatten(1)), -1)

    def at(self, frame_id):
        offsets = torch.arange(self.task.future_frames, device=self.frames.device)
        indices = (frame_id[:, None] + offsets).clamp(0, len(self.frames) - 1)
        return self.frames[indices].flatten(1)

    def __call__(self, seed, offset):
        if (seed.motion_id != 0).any():
            raise ValueError("This initial integration supports one motion clip per run")
        return self.at(seed.frame_id + offset)
