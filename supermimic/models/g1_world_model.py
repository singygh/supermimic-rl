import torch
from torch import nn

from .networks import mlp


class G1WorldModel(nn.Module):
    """Normalized residual state prediction, with analytic tracking heads."""
    learn_observation = learn_reward = learn_termination = False

    def __init__(self, task, hidden_dims=(512, 512, 512)):
        super().__init__()
        self.task, self.spec = task, task.spec
        self.dynamics = mlp(self.spec.state_dim + 29 + self.spec.context_dim,
                            self.spec.state_dim, hidden_dims)
        nn.init.normal_(self.dynamics[-1].weight, std=1e-3)
        nn.init.zeros_(self.dynamics[-1].bias)

    def transition(self, state, action, context):
        from supermimic.tasks.g1 import normalize_quat

        delta = self.dynamics(torch.cat((state / self.task.state_scale, action, context), -1))
        predicted = state + self.spec.dt * self.task.state_scale * delta
        s = self.task.unpack_state(predicted)
        old = self.task.unpack_state(state)
        return self.task.pack_state(s["q"], s["qd"], action, old["action"], s["pos"],
                                    normalize_quat(s["quat"]), s["lin_vel"], s["ang_vel"])

    forward = transition

    def state_error(self, predicted, target):
        return ((predicted - target) / self.task.state_scale).square().mean(-1)

    def observe_clean(self, state, context):
        return self.task.observe_clean(state, context)

    def reward(self, state, action, goal):
        return self.task.reward(state, action, goal)

    def termination_prob(self, state, goal):
        return self.task.termination_prob(state, goal)
