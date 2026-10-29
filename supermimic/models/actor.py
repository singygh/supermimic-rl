import torch
from torch import nn

from .networks import mlp


class Actor(nn.Module):
    """Deterministic deployable policy: observation history and goal only."""

    def __init__(self, spec, hidden_dims=(128, 128), encoder_dims=None, latent_dim=None):
        super().__init__()
        latent_dim = hidden_dims[0] if latent_dim is None else latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.encoder_dims = tuple(encoder_dims) if encoder_dims is not None else None
        self.latent_dim = latent_dim
        self.bound_latent = encoder_dims is not None
        self.goal_encoder = mlp(spec.goal_dim, latent_dim, encoder_dims or (hidden_dims[0],))
        self.decoder = mlp(spec.obs_dim * spec.history_length + latent_dim,
                           spec.action_dim, hidden_dims)

    def forward(self, obs_history, goal):
        latent = self.goal_encoder(goal)
        if self.bound_latent:
            latent = latent.tanh()
        features = torch.cat((obs_history.flatten(1), latent), dim=-1)
        return self.decoder(features).tanh()

    def act(self, obs_history, goal):
        return self(obs_history, goal)

    act_inference = act

    def get_config(self):
        return {"hidden_dims": self.hidden_dims, "encoder_dims": self.encoder_dims, "latent_dim": self.latent_dim}
