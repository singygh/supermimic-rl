from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass
class TensorBatch:
    def to(self, device):
        return type(self)(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def detached(self):
        return type(self)(**{f.name: getattr(self, f.name).detach() for f in fields(self)})


@dataclass
class TransitionBatch(TensorBatch):
    state_t: Tensor
    clean_obs_t: Tensor
    obs_history_t: Tensor
    goal_t: Tensor
    action_exec_t: Tensor
    reward_tp1: Tensor
    next_state: Tensor
    next_clean_obs: Tensor
    terminated_tp1: Tensor
    truncated_tp1: Tensor
    context: Tensor
    episode_id: Tensor
    motion_id: Tensor
    frame_id: Tensor


@dataclass
class ReplayWindow(TensorBatch):
    state_0: Tensor
    actions: Tensor
    states: Tensor
    rewards: Tensor
    clean_obs: Tensor
    terminated: Tensor
    truncated: Tensor
    goals: Tensor
    context: Tensor
    valid_mask: Tensor


@dataclass
class ImaginationSeed(TensorBatch):
    state_0: Tensor
    obs_history_0: Tensor
    goal_0: Tensor
    context: Tensor
    motion_id: Tensor
    frame_id: Tensor


@dataclass
class ImaginedRollout(TensorBatch):
    states: Tensor  # [B, H+1, Ds]
    goals: Tensor  # [B, H+1, Dg]
    context: Tensor  # [B, Dxi]
    actions: Tensor
    rewards: Tensor  # [B, H], reward at t+1 produced by action at t
    termination: Tensor
    values: Tensor  # [B, H+1]


def lambda_returns(rewards, termination, values, gamma, lam):
    """GAE returns bootstrapped from V_H; detach before fitting critic."""
    result = []
    ret = values[:, -1]
    for t in reversed(range(rewards.shape[1])):
        ret = rewards[:, t] + gamma * (1 - termination[:, t]) * (
            (1 - lam) * values[:, t + 1] + lam * ret
        )
        result.append(ret)
    return torch.stack(result[::-1], dim=1)


def survival_objective(rewards, termination, terminal_value, gamma):
    """Reward at t+1 has c_t weight; terminal bootstrap has c_H weight."""
    survival = torch.cat((torch.ones_like(termination[:, :1]),
                          (1 - termination).cumprod(dim=1)), dim=1)
    discounts = gamma ** torch.arange(rewards.shape[1] + 1, device=rewards.device,
                                     dtype=rewards.dtype)
    objective = (survival[:, :-1] * discounts[:-1] * rewards).sum(dim=1)
    return objective + discounts[-1] * survival[:, -1] * terminal_value
