import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class SquashedGaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, encoder: nn.Module, enc_dim: int, action_dim: int):
        super().__init__()
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
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)

    def _dist(self, obs: tuple[torch.Tensor | None, torch.Tensor | None]):
        latent = self.encoder(obs[0], obs[1])
        h = self.trunk(latent)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        return Normal(mean, std)

    def sample(self, obs: tuple[torch.Tensor | None, torch.Tensor | None]):
        dist = self._dist(obs)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x).sum(-1, keepdim=True)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs: tuple[torch.Tensor | None, torch.Tensor | None], deterministic: bool = False):
        dist = self._dist(obs)
        if deterministic:
            action = torch.tanh(dist.mean)
            return action
        x = dist.sample()
        return torch.tanh(x)


class TwinQCritic(nn.Module):
    def __init__(self, encoder: nn.Module, enc_dim: int, action_dim: int):
        super().__init__()
        self.encoder = encoder

        self.q1 = nn.Sequential(
            nn.Linear(enc_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(enc_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: tuple[torch.Tensor | None, torch.Tensor | None], action: torch.Tensor):
        latent = self.encoder(obs[0], obs[1])
        x = torch.cat([latent, action], dim=-1)
        return self.q1(x), self.q2(x)
