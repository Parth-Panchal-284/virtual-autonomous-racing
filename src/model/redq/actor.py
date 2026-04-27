import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from typing import Callable

class SquashedGaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, encoder: nn.Module, enc_dim: int, action_dim: int):
        super(SquashedGaussianActor, self).__init__()
        self.encoder = encoder
        self.trunk = nn.Sequential(
            nn.Linear(enc_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)

    def _dist(self, obs: tuple[torch.Tensor, torch.Tensor]):
        latent = self.encoder(obs[0], obs[1])
        h = self.trunk(latent)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        return Normal(mean, std)

    def sample(self, obs: tuple[torch.Tensor, torch.Tensor]):
        dist = self._dist(obs)
        x = dist.rsample()
        action = torch.tanh(x)
        logp_pi = dist.log_prob(x).sum(axis=-1, keepdim=True)
        logp_pi -= (2 * (np.log(2) - x - F.softplus(-2 * x))).sum(axis=1, keepdim=True)
        return action, logp_pi

    @torch.no_grad()
    def act(self, obs: tuple[torch.Tensor, torch.Tensor], deterministic: bool = False):
        dist = self._dist(obs)
        if deterministic:
            action = torch.tanh(dist.mean)
            return action
        x = dist.rsample()
        logp_pi = dist.log_prob(x).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - x - F.softplus(-2 * x))).sum(axis=1)
        return torch.tanh(x), logp_pi


class QNetwork(nn.Module):
    def __init__(self, inp_dim : int, encoder: type[nn.Module] | Callable[[], nn.Module]):
        super(QNetwork, self).__init__()
        self.encoder = encoder()
        self.model = nn.Sequential(
            nn.Linear(inp_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def forward(self, obs: tuple[torch.Tensor, torch.Tensor], action: torch.Tensor):
        latent = self.encoder(obs[0], obs[1])
        x = torch.cat([latent, action], dim=-1)
        return self.model(x)
        
class REDQCritic(nn.Module):
    def __init__(self, encoder: type[nn.Module] | Callable[[], nn.Module], enc_dim: int, action_dim: int, N: int):
        super(REDQCritic, self).__init__()
        self.encoder = encoder
        
        self.N = N
        self.qs = nn.ModuleList([
            QNetwork(enc_dim + action_dim, self.encoder)
            for _ in range(self.N)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
                
    def forward(self, obs: tuple[torch.Tensor | None, torch.Tensor | None], action: torch.Tensor):
        return [ model.forward(obs, action) for model in self.qs]