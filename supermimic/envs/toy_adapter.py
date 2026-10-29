import torch

from supermimic.configs import TaskSpec
from supermimic.models.heads import ToyReward, toy_transition
from .protocol import EnvStep, StepView


def stationary_goal(seed, offset):
    """Toy goals are fixed per episode; motion tasks must replace this provider."""
    return seed.goal_0


class ToyAdapter:
    """Vector point masses with episode-constant randomized damping and targets."""

    def __init__(self, num_envs=16, device="cpu", max_episode_steps=64, noise_std=0.01):
        if num_envs < 1 or max_episode_steps < 1 or noise_std < 0:
            raise ValueError("Invalid toy environment settings")
        self.num_envs, self.device = num_envs, str(device)
        self.spec = TaskSpec()
        self.max_episode_steps, self.noise_std = max_episode_steps, noise_std
        self.state = torch.zeros(num_envs, 3, device=device)
        self.context = torch.zeros(num_envs, 1, device=device)
        self.goal = torch.zeros(num_envs, 1, device=device)
        self.history = torch.zeros(num_envs, self.spec.history_length, 3, device=device)
        self.episode_id = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.frame_id = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset()

    def get_step_view(self):
        return StepView(self.state.clone(), self.state.clone(), self.history.clone(), self.goal.clone(),
                        self.context.clone(), self.episode_id.clone(), torch.zeros_like(self.frame_id),
                        self.frame_id.clone())

    @torch.no_grad()
    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        self.state[ids] = 0
        self.state[ids, 0] = torch.rand(len(ids), device=self.device) - 0.5
        self.context[ids] = 0.1 + 0.4 * torch.rand(len(ids), 1, device=self.device)
        self.goal[ids] = 2 * torch.rand(len(ids), 1, device=self.device) - 1
        self.frame_id[ids] = 0
        self.episode_id[ids] += 1
        obs = self.state[ids] + self.noise_std * torch.randn_like(self.state[ids])
        self.history[ids] = obs[:, None].expand(-1, self.spec.history_length, -1)
        return self.get_step_view()

    @torch.no_grad()
    def step(self, action_exec):
        action = action_exec.to(self.device).clamp(-1, 1)
        if action.shape != (self.num_envs, self.spec.action_dim):
            raise ValueError("Action shape mismatch")
        self.state = toy_transition(self.state, action, self.context, self.spec.dt, self.spec.action_scale)
        reward = ToyReward()(self.state, action, self.goal)
        self.frame_id += 1
        obs = self.state + self.noise_std * torch.randn_like(self.state)
        self.history = torch.cat((self.history[:, 1:], obs[:, None]), dim=1)
        terminated = self.state[:, 0].abs() >= 3
        truncated = self.frame_id >= self.max_episode_steps
        terminal = self.get_step_view()
        ids = (terminated | truncated).nonzero().flatten()
        if len(ids):
            self.reset(ids)
        return EnvStep(terminal, self.get_step_view(), action.clone(), reward, terminated, truncated)

    def get_context(self):
        return self.context.clone()

    def get_full_state(self):
        return self.state.clone()

    def state_dict(self):
        return {"tensors": {name: getattr(self, name).clone() for name in (
            "state", "context", "goal", "history", "episode_id", "frame_id")},
            "max_episode_steps": self.max_episode_steps, "noise_std": self.noise_std}

    def load_state_dict(self, state):
        if state["max_episode_steps"] != self.max_episode_steps or state["noise_std"] != self.noise_std:
            raise ValueError("Toy environment configuration mismatch")
        for name, value in state["tensors"].items():
            if getattr(self, name).shape != value.shape:
                raise ValueError(f"Environment tensor shape mismatch: {name}")
            setattr(self, name, value.to(self.device).clone())
