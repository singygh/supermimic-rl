import torch
from torch import nn

from .networks import mlp


class WorldModel(nn.Module):
    """Residual dynamics and optional analytic observation/reward/termination heads.

    Supplied heads must keep input gradients. Otherwise heads are learned and
    supervised in fit_world_model. No privileged input is passed to Actor.
    """

    def __init__(self, spec, hidden_dims=(128, 128), observation_head=None,
                 reward_head=None, termination_head=None):
        super().__init__()
        self.spec = spec
        self.dynamics = mlp(spec.state_dim + spec.action_dim + spec.context_dim,
                            spec.state_dim, hidden_dims)
        # Start near identity to keep the initial 32-step graph well conditioned.
        nn.init.normal_(self.dynamics[-1].weight, std=0.001)
        nn.init.zeros_(self.dynamics[-1].bias)
        self.learn_observation = observation_head is None
        self.learn_reward = reward_head is None
        self.learn_termination = termination_head is None
        self.observation_head = observation_head if observation_head is not None else mlp(
            spec.state_dim + spec.context_dim, spec.obs_dim, hidden_dims)
        self.reward_head = reward_head if reward_head is not None else mlp(
            spec.state_dim + spec.action_dim + spec.goal_dim, 1, hidden_dims)
        self.termination_head = termination_head if termination_head is not None else mlp(
            spec.state_dim + spec.goal_dim, 1, hidden_dims)

    def transition(self, state, action, context):
        return state + self.dynamics(torch.cat((state, action, context), dim=-1))

    forward = transition

    def observe_clean(self, state, context):
        if self.learn_observation:
            return self.observation_head(torch.cat((state, context), dim=-1))
        return self.observation_head(state, context)

    def reward(self, next_state, action, goal):
        if self.learn_reward:
            return self.reward_head(torch.cat((next_state, action, goal), dim=-1)).squeeze(-1)
        return self.reward_head(next_state, action, goal)

    def termination_logits(self, next_state, goal):
        if not self.learn_termination:
            raise RuntimeError("Analytic termination heads expose probabilities only")
        return self.termination_head(torch.cat((next_state, goal), dim=-1)).squeeze(-1)

    def termination_prob(self, next_state, goal):
        if self.learn_termination:
            return self.termination_logits(next_state, goal).sigmoid()
        return self.termination_head(next_state, goal)
