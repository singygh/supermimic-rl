import torch
from torch import nn

from .networks import mlp


class Critic(nn.Module):
    def __init__(self, spec, hidden_dims=(128, 128)):
        super().__init__()
        self.net = mlp(spec.state_dim + spec.goal_dim + spec.context_dim, 1, hidden_dims)

    def forward(self, state, goal, context):
        return self.net(torch.cat((state, goal, context), dim=-1)).squeeze(-1)

    value = forward
