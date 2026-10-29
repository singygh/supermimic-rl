"""Shared toy task mathematics used by the simulator and analytic model heads."""
import torch
from torch import nn


def toy_transition(state, action, context, dt=0.02, action_scale=4.0):
    velocity = state[:, 1:2] + dt * (action_scale * action - context * state[:, 1:2])
    position = state[:, :1] + dt * velocity
    return torch.cat((position, velocity, action), dim=-1)


class ToyObservation(nn.Module):
    def forward(self, state, context):
        return state


class ToyReward(nn.Module):
    def forward(self, next_state, action, goal):
        return 1.0 - (next_state[:, 0] - goal[:, 0]).square() - 0.01 * action.square().sum(-1)


class ToyTermination(nn.Module):
    def forward(self, next_state, goal):
        return torch.sigmoid(20 * (next_state[:, 0].abs() - 3.0))
