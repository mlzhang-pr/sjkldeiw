import torch
import torch.nn.functional as F
from torch import nn

from agent import utils


class StateEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        hidden_depth: int = 2,
        normalize_output: bool = False,
    ):
        super().__init__()
        self.normalize_output = normalize_output
        self.trunk = utils.mlp(obs_dim, hidden_dim, latent_dim, hidden_depth)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.trunk(obs)
        if self.normalize_output:
            z = F.normalize(z, dim=-1)
        return z


class TransitionEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        hidden_depth: int = 2,
        normalize_output: bool = False,
    ):
        super().__init__()
        self.normalize_output = normalize_output
        input_dim = obs_dim + action_dim
        self.trunk = utils.mlp(input_dim, hidden_dim, latent_dim, hidden_depth)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self.trunk(torch.cat([obs, action], dim=-1))
        if self.normalize_output:
            z = F.normalize(z, dim=-1)
        return z


class LatentTransitionEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        hidden_depth: int = 2,
        normalize_output: bool = False,
    ):
        super().__init__()
        self.normalize_output = normalize_output
        input_dim = obs_dim + action_dim
        self.trunk = utils.mlp(input_dim, hidden_dim, latent_dim, hidden_depth)
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:

        z = self.trunk(torch.cat([obs, action], dim=-1))
        if self.normalize_output:
            z = F.normalize(z, dim=-1)
        return z
