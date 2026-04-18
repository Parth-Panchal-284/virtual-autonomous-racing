import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Normal

# ─────────────────────────────────────────────
# Actor-Critic
# ─────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Dual-stream Actor-Critic with tanh-squashed Gaussian policy.
    Actions are always in (-1, 1) — safe for tmrl's gamepad interface.
    """
    def __init__(self, encoder: nn.Module, enc_dim: int, action_dim: int):
        super().__init__()
        self.encoder = encoder

        self.actor_mean = nn.Sequential(
            nn.Linear(enc_dim, 256), nn.Tanh(),
            nn.Linear(256, action_dim),
        )
        # Start with std ≈ 0.6 for less chaotic early exploration
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        self.critic = nn.Sequential(
            nn.Linear(enc_dim, 256), nn.Tanh(),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias) # type: ignore
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01) # type: ignore
        nn.init.orthogonal_(self.critic[-1].weight,     gain=1.0) # type: ignore

    def forward(self, obs):
        img, vec = obs
        enc   = self.encoder(img, vec)
        mean  = self.actor_mean(enc)
        std   = self.log_std.exp().expand_as(mean)
        dist  = Normal(mean, std)
        value = self.critic(enc).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, obs):
        dist, value = self(obs)
        action_raw  = dist.sample()
        action      = torch.tanh(action_raw)
        log_prob    = dist.log_prob(action_raw).sum(-1) \
                      - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob, value