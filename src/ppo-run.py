"""
Proximal Policy Optimization (PPO) for TMRL (TrackMania Reinforcement Learning)

Implements clipped PPO with:
  - Actor-Critic network with shared CNN encoder (for image obs) or MLP
  - Generalized Advantage Estimation (GAE)
  - Multiple epochs of minibatch updates per rollout
  - Entropy bonus for exploration
  - Value function clipping

Compatible with tmrl's custom gym-like interface.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import time


# ─────────────────────────────────────────────
# Network definitions
# ─────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """Lightweight CNN for processing raw pixel observations from TrackMania."""
    def __init__(self, in_channels: int = 1, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Dynamically compute flatten size
        dummy = torch.zeros(1, in_channels, 64, 64)
        flat = self.net(dummy).shape[1]
        self.proj = nn.Linear(flat, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(self.net(x / 255.0)))


class MLPEncoder(nn.Module):
    """MLP encoder for flat/vector observations (speed, gear, lidar, etc.)."""
    def __init__(self, obs_dim: int, hidden: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),  nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Shared-encoder Actor-Critic for continuous action spaces.
    TMRL default actions: [gas, brake, steer] ∈ [-1, 1]^3
    """
    def __init__(self, encoder: nn.Module, enc_dim: int, action_dim: int):
        super().__init__()
        self.encoder = encoder

        # Actor head: outputs mean of Gaussian policy
        self.actor_mean = nn.Sequential(
            nn.Linear(enc_dim, 256), nn.Tanh(),
            nn.Linear(256, action_dim),  # raw mean; tanh squashing applied at sample time
        )
        # Log-std as learnable parameter (state-independent)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head: outputs state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(enc_dim, 256), nn.Tanh(),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # smaller gain for output layers
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        enc = self.encoder(obs)
        mean = self.actor_mean(enc)
        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(enc).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist, value = self(obs)
        action_raw = dist.sample()
        # Squash to [-1, 1] and correct log_prob for the tanh transform
        action = torch.tanh(action_raw)
        # log_prob correction: log|d(tanh)/dx| = log(1 - tanh^2 + eps)
        log_prob = dist.log_prob(action_raw).sum(-1) \
                   - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob, value


# ─────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────

class RolloutBuffer:
    """Stores one rollout (T steps × N envs) and computes GAE returns."""

    def __init__(self, steps: int, obs_shape, action_dim: int, device: torch.device):
        self.steps = steps
        self.device = device
        T = steps

        self.obs      = torch.zeros(T, *obs_shape)
        self.actions  = torch.zeros(T, action_dim)
        self.log_probs= torch.zeros(T)
        self.rewards  = torch.zeros(T)
        self.values   = torch.zeros(T)
        self.dones    = torch.zeros(T)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i]       = obs.cpu()
        self.actions[i]   = action.cpu()
        self.log_probs[i] = log_prob.cpu()
        self.rewards[i]   = torch.tensor(reward, dtype=torch.float32)
        self.values[i]    = value.cpu()
        self.dones[i]     = torch.tensor(done, dtype=torch.float32)
        self.ptr += 1

    def compute_returns(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute GAE advantages and discounted returns."""
        last_value = last_value.cpu().item()
        advantages = torch.zeros(self.steps)
        gae = 0.0
        for t in reversed(range(self.steps)):
            next_val  = last_value if t == self.steps - 1 else self.values[t + 1].item()
            next_done = 0.0        if t == self.steps - 1 else self.dones[t + 1].item()
            delta = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            gae   = delta + gamma * gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
        self.returns    = advantages + self.values
        self.advantages = advantages
        self.ptr = 0  # reset

    def get_minibatches(self, batch_size: int, device: torch.device):
        """Yield shuffled minibatches."""
        idx = torch.randperm(self.steps)
        for start in range(0, self.steps, batch_size):
            mb = idx[start : start + batch_size]
            yield (
                self.obs[mb].to(device),
                self.actions[mb].to(device),
                self.log_probs[mb].to(device),
                self.returns[mb].to(device),
                self.advantages[mb].to(device),
            )


# ─────────────────────────────────────────────
# PPO trainer
# ─────────────────────────────────────────────

class PPOTrainer:
    """
    Proximal Policy Optimization trainer compatible with tmrl environments.

    Usage
    -----
    >>> from tmrl import get_environment
    >>> env = get_environment()
    >>> trainer = PPOTrainer(obs_dim=..., action_dim=3)
    >>> trainer.train(env, total_steps=5_000_000)
    """

    def __init__(
        self,
        obs_dim: int | None = None,       # set for flat obs; None triggers CNN
        obs_channels: int = 1,            # channels for CNN (grayscale frames)
        action_dim: int = 3,              # [gas, brake, steer]
        enc_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "auto",
    ):
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )

        # Build encoder
        if obs_dim is not None:
            encoder  = MLPEncoder(obs_dim, out_dim=enc_dim)
            obs_shape = (obs_dim,)
        else:
            encoder  = CNNEncoder(obs_channels, out_dim=enc_dim)
            obs_shape = (obs_channels, 64, 64)

        self.policy = ActorCritic(encoder, enc_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(rollout_steps, obs_shape, action_dim, self.device)

        print(f"PPO on {self.device} | params: {sum(p.numel() for p in self.policy.parameters()):,}")

    # ── core PPO update ──────────────────────────────────────────────

    def _update(self):
        stats = dict(policy_loss=[], value_loss=[], entropy=[], approx_kl=[])

        for _ in range(self.n_epochs):
            for obs, actions, old_log_probs, returns, advantages in \
                    self.buffer.get_minibatches(self.batch_size, self.device):

                # Normalize advantages within minibatch
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                dist, values = self.policy(obs)
                # actions stored are already tanh-squashed; invert for log_prob
                actions_raw  = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
                log_probs    = dist.log_prob(actions_raw).sum(-1) \
                               - torch.log(1 - actions.pow(2) + 1e-6).sum(-1)
                entropy      = dist.entropy().sum(-1).mean()

                # Ratio for clipped surrogate
                ratio       = (log_probs - old_log_probs).exp()
                surr1       = ratio * advantages
                surr2       = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (optionally clipped)
                value_loss  = 0.5 * (values - returns).pow(2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((old_log_probs - log_probs)).mean().item()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(approx_kl)

        return {k: np.mean(v) for k, v in stats.items()}

    # ── obs helper ───────────────────────────────────────────────────

    def _to_tensor(self, obs) -> torch.Tensor:
        """Convert tmrl observation (tuple/array/dict) to tensor."""
        if isinstance(obs, (tuple, list)):
            obs = np.concatenate([np.array(o).flatten() for o in obs])
        obs = np.array(obs, dtype=np.float32)
        return torch.from_numpy(obs).unsqueeze(0).to(self.device)

    # ── main training loop ───────────────────────────────────────────

    def train(self, env, total_steps: int = 5_000_000, log_interval: int = 10):
        """
        Parameters
        ----------
        env          : tmrl gym-like environment (returned by tmrl.get_environment())
        total_steps  : total environment steps
        log_interval : print stats every N rollouts
        """
        obs, _ = env.reset()
        ep_reward, ep_len, ep_count = 0.0, 0, 0
        recent_rewards = deque(maxlen=20)
        global_step, rollout_idx = 0, 0
        start_time = time.time()

        while global_step < total_steps:
            # ── collect rollout ──────────────────────────────────────
            self.policy.eval()
            for _ in range(self.rollout_steps):
                obs_t = self._to_tensor(obs)
                with torch.no_grad():
                    action, log_prob, value = self.policy.act(obs_t)

                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated

                self.buffer.add(obs_t.squeeze(0), action.squeeze(0),
                                log_prob.squeeze(0), reward, value.squeeze(0), done)

                obs = next_obs
                ep_reward += reward
                ep_len    += 1
                global_step += 1

                if done:
                    recent_rewards.append(ep_reward)
                    ep_count  += 1
                    ep_reward  = 0.0
                    ep_len     = 0
                    obs, _     = env.reset()

                if global_step >= total_steps:
                    break

            # Bootstrap value for last obs
            with torch.no_grad():
                last_obs_t = self._to_tensor(obs)
                _, _, last_value = self.policy.act(last_obs_t)
            self.buffer.compute_returns(last_value.squeeze(0), self.gamma, self.gae_lambda)

            # ── update policy ────────────────────────────────────────
            self.policy.train()
            stats = self._update()
            rollout_idx += 1

            # ── logging ──────────────────────────────────────────────
            if rollout_idx % log_interval == 0:
                elapsed = time.time() - start_time
                mean_rew = np.mean(recent_rewards) if recent_rewards else float("nan")
                fps = global_step / elapsed
                print(
                    f"[step {global_step:>8d}] "
                    f"mean_rew={mean_rew:>8.2f}  "
                    f"ep={ep_count:>5d}  "
                    f"pi_loss={stats['policy_loss']:>7.4f}  "
                    f"v_loss={stats['value_loss']:>7.4f}  "
                    f"entropy={stats['entropy']:>6.3f}  "
                    f"kl={stats['approx_kl']:>6.4f}  "
                    f"fps={fps:>5.0f}"
                )

        env.close()
        print("Training complete.")

    # ── checkpointing ────────────────────────────────────────────────

    def save(self, path: str = "ppo_tmrl.pt"):
        torch.save({"policy": self.policy.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)
        print(f"Saved checkpoint → {path}")

    def load(self, path: str = "ppo_tmrl.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Loaded checkpoint ← {path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick-start example using tmrl's built-in environment.

    Install dependencies:
        pip install tmrl torch numpy

    Run:
        python ppo_tmrl.py
    """
    try:
        import tmrl
        env = tmrl.get_environment()

        # Inspect observation space to set obs_dim correctly
        obs, _ = env.reset()
        if isinstance(obs, (tuple, list)):
            obs_dim = sum(np.array(o).flatten().shape[0] for o in obs)
        else:
            obs_dim = int(np.prod(np.array(obs).shape))

        print(f"obs_dim={obs_dim}  action_dim={env.action_space.shape[0]}")

        trainer = PPOTrainer(
            obs_dim=obs_dim,
            action_dim=env.action_space.shape[0],
            rollout_steps=2048,
            n_epochs=10,
            batch_size=64,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            entropy_coef=0.005,
        )
        
        trainer.load("ppo_tmrl_final.pt")

        

        trainer.train(env, total_steps=1_000_000)
        trainer.save("ppo_tmrl_final.pt")

    except ImportError:
        print("tmrl not installed. Install with: pip install tmrl")
        print("Running self-test with a dummy env instead...\n")

        # Lightweight smoke test with a Pendulum-like dummy env
        import gymnasium as gym
        env = gym.make("Pendulum-v1")
        obs, _ = env.reset()
        trainer = PPOTrainer(obs_dim=obs.shape[0], action_dim=1)
        trainer.train(env, total_steps=50_000, log_interval=5)
        trainer.save("ppo_dummy.pt")