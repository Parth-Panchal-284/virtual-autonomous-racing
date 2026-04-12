"""
Proximal Policy Optimization (PPO) for TMRL (TrackMania Reinforcement Learning)

Architecture
------------
tmrl's default observation is a tuple:
  (speed, gear, rpm,  image_t, image_t-1, image_t-2, image_t-3)
  where each image is a grayscale (1, H, W) float array and
  speed/gear/rpm are scalars or short float arrays.

This file uses a DUAL-STREAM encoder:
  ┌─ CNN stream  ─────────────────────────────┐
  │  Stacked grayscale frames (4×H×W)         │──► 256-d visual features
  └───────────────────────────────────────────┘
  ┌─ MLP stream  ─────────────────────────────┐
  │  speed + gear + rpm (vector obs)          │──► 64-d state features
  └───────────────────────────────────────────┘
         │                   │
         └────── concat ─────┘
                    │
               fusion MLP
                    │
            Actor / Critic heads
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import time

from datetime import datetime

from util import CurrentRunFolder



# ─────────────────────────────────────────────
# Crash penalty wrapper
# ─────────────────────────────────────────────
 
class CrashPenaltyWrapper:
    """
    Wraps a tmrl environment and adds a speed-scaled penalty on crash.
 
    How it works
    ------------
    tmrl signals a crash via `terminated=True`.  At that moment we read
    the current speed from the observation and subtract:
 
        penalty = base_penalty + speed_coef * speed_kmh
 
    where speed_kmh is extracted from the first scalar in the obs tuple
    (tmrl's default layout puts speed first, in km/h).
 
    Parameters
    ----------
    env         : the raw tmrl environment
    base_penalty: flat penalty applied on any crash              (default 50)
    speed_coef  : extra penalty per km/h of speed at crash time  (default  1)
    max_penalty : ceiling so a single crash can't dominate       (default 200)
 
    Example penalty curve
    ---------------------
      0 km/h crash  →  -50
     50 km/h crash  → -100
    150 km/h crash  → -200  (capped)
    """
 
    def __init__(self, env, base_penalty: float = 50.0,
                 speed_coef: float = 1.0, max_penalty: float = 200.0):
        self.env          = env
        self.base_penalty = base_penalty
        self.speed_coef   = speed_coef
        self.max_penalty  = max_penalty
        # Forward standard gym attributes
        self.action_space      = env.action_space
        self.observation_space = env.observation_space
        
        self.last_speed = 0
        
        self.frames_since_last_crash = 10000
 
    def _get_speed(self, obs) -> float:
        """Extract speed (km/h) from obs. Assumes speed is the first scalar."""
        if isinstance(obs, (tuple, list)):
            speed_arr = np.array(obs[0], dtype=np.float32).flatten()
            return float(speed_arr[0])
        return float(np.array(obs, dtype=np.float32).flatten()[0])
 
    def reset(self, **kwargs):
        
        self.frames_since_last_crash = 10000
        self.last_speed = 0
        return self.env.reset(**kwargs)
    
    def determine_crash(self, speed):
        difference = (speed - self.last_speed) * 15
        return difference < -25 and 3 < speed and 30 < self.frames_since_last_crash
        
 
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
 

        speed     = self._get_speed(obs)
        if(self.determine_crash(speed)):
            self.frames_since_last_crash = 0
            penalty   = self.base_penalty + self.speed_coef * abs(speed)
            penalty   = min(penalty, self.max_penalty)
            print(f"\n\nCRASHED: {speed} mph, Penalty: {penalty}, Rewqard: {reward}")
            reward   -= penalty
            info["crash_penalty"] = penalty
            info["crash_speed"]   = speed
 
        self.last_speed = speed
        self.frames_since_last_crash += 1
        return obs, reward, terminated, truncated, info
 
    def close(self):
        return self.env.close()


# ─────────────────────────────────────────────
# Observation splitter
# ─────────────────────────────────────────────

def split_obs(obs):
    """
    Split a tmrl observation tuple into:
      - img_arrays : list of (1, H, W) or (H, W) numpy arrays  (images)
      - vec_arrays : list of flat numpy arrays                  (speed/gear/rpm etc.)

    Anything with more than 1000 elements is treated as an image.
    """
    if not isinstance(obs, (tuple, list)):
        return [], [np.array(obs, dtype=np.float32).flatten()]

    img_arrays, vec_arrays = [], []
    
    for o in obs:
        # print(o.shape)
        arr = np.array(o, dtype=np.float32)
        if arr.ndim >= 2:
            img_arrays.append(arr)
        else:
            vec_arrays.append(arr.flatten())
    return img_arrays, vec_arrays


def obs_to_tensors(obs, device):
    """
    Returns (img_tensor, vec_tensor) on `device`.
      img_tensor : (1, C, H, W)  stacked grayscale frames, or None
      vec_tensor : (1, D)        concatenated vector obs,   or None
    """
    img_arrays, vec_arrays = split_obs(obs)

    img_t = None
    if img_arrays:
        img_t = torch.from_numpy(np.array(img_arrays)).to(device)

    vec_t = None
    if vec_arrays:
        vec   = np.concatenate(vec_arrays)
        vec_t = torch.from_numpy(vec).unsqueeze(0).to(device)      # (1, D)

    return img_t, vec_t


def probe_obs_dims(obs):
    """
    Given a sample observation, return (n_img_channels, img_h, img_w, vec_dim).
    Returns None for streams that are absent.
    """
    img_arrays, vec_arrays = split_obs(obs)

    img_channels = img_h = img_w = None
    if img_arrays:
        img_channels = img_arrays[0].shape[0]
        first = np.array(img_arrays[0], dtype=np.float32)
        print(first.shape)
        img_h, img_w = first.shape[1], first.shape[2]

    vec_dim = None
    if vec_arrays:
        vec_dim = sum(a.flatten().size for a in vec_arrays)

    return img_channels, img_h, img_w, vec_dim


# ─────────────────────────────────────────────
# Network building blocks
# ─────────────────────────────────────────────

class CNNStream(nn.Module):
    """
    Deeper CNN for stacked grayscale frames from TrackMania.
    Wider filters (32->64->128->128) capture richer 3D perspective features
    compared to the original Nature-DQN (32->64->64).
    Input : (B, C, H, W)  pixel values in [0, 255]
    Output: (B, out_dim)
    """
    def __init__(self, in_channels: int, img_h: int, img_w: int, out_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32,  kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,          64,  kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,          128, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(128,         128, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_h, img_w)
            flat  = self.conv(dummy).shape[1]
        self.proj = nn.Sequential(
            nn.Linear(flat, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x / 255.0))
 
 
class VecStream(nn.Module):
    """
    Larger MLP for vector observations (speed, gear, lidar history ...).
    With 8 history frames of lidar (~152 floats), the original 128->64
    was a severe bottleneck. Now 256->256->128 to preserve lidar detail.
    Input : (B, D)
    Output: (B, out_dim)
    """
    def __init__(self, vec_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vec_dim, 256), nn.Tanh(),
            nn.Linear(256,     256), nn.Tanh(),
            nn.Linear(256, out_dim), nn.Tanh(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DualStreamEncoder(nn.Module):
    """
    Fuses CNN visual features with MLP state features using a deeper fusion MLP.
    Two fusion layers instead of one give the network room to learn non-trivial
    interactions between visual context and vehicle state (e.g. speed + corner ahead).
    Either stream can be absent (pass None).
    Output: (B, enc_dim)
    """
    def __init__(
        self,
        cnn,
        vec,
        cnn_out_dim: int,
        vec_out_dim: int,
        enc_dim: int = 512,
    ):
        super().__init__()
        self.cnn = cnn
        self.vec = vec
        fusion_in = (cnn_out_dim if cnn is not None else 0) + \
                    (vec_out_dim if vec is not None else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, enc_dim), nn.ReLU(),
            nn.Linear(enc_dim,   enc_dim), nn.ReLU(),
        )
 
    def forward(self, img, vec):
        parts = []
        if self.cnn is not None and img is not None:
            parts.append(self.cnn(img))
        if self.vec is not None and vec is not None:
            parts.append(self.vec(vec))
        return self.fusion(torch.cat(parts, dim=-1))


# ─────────────────────────────────────────────
# Actor-Critic
# ─────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Dual-stream Actor-Critic with tanh-squashed Gaussian policy.
    Actions are always in (-1, 1) — safe for tmrl's gamepad interface.
    """
    def __init__(self, encoder: DualStreamEncoder, enc_dim: int, action_dim: int):
        super().__init__()
        self.encoder = encoder
 
        # Deeper actor head: enc -> 512 -> 256 -> action_dim
        # Two hidden layers let the policy express more nuanced decisions
        self.actor_mean = nn.Sequential(
            nn.Linear(enc_dim, 512), nn.Tanh(),
            nn.Linear(512,     256), nn.Tanh(),
            nn.Linear(256, action_dim),
        )
        # Start with std ~ 0.6 for less chaotic early exploration
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
 
        # Deeper critic head: enc -> 512 -> 256 -> 1
        # Value estimation benefits from extra capacity to predict long-horizon returns
        self.critic = nn.Sequential(
            nn.Linear(enc_dim, 512), nn.Tanh(),
            nn.Linear(512,     256), nn.Tanh(),
            nn.Linear(256, 1),
        )
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight,     gain=1.0)
 
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
 


# ─────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────

class RolloutBuffer:
    """Stores one rollout and computes GAE advantages + discounted returns."""

    def __init__(self, steps: int, img_shape, vec_dim, action_dim: int, device):
        self.steps   = steps
        self.device  = device
        self.has_img = img_shape is not None
        self.has_vec = vec_dim is not None

        if self.has_img:
            self.imgs = torch.zeros(steps, *img_shape)
        if self.has_vec:
            self.vecs = torch.zeros(steps, vec_dim)

        self.actions   = torch.zeros(steps, action_dim)
        self.log_probs = torch.zeros(steps)
        self.rewards   = torch.zeros(steps)
        self.values    = torch.zeros(steps)
        self.dones     = torch.zeros(steps)
        self.ptr = 0

    def add(self, img, vec, action, log_prob, reward, value, done):
        i = self.ptr
        if self.has_img and img is not None:
            self.imgs[i] = img.squeeze(0).cpu()
        if self.has_vec and vec is not None:
            self.vecs[i] = vec.squeeze(0).cpu()
        self.actions[i]   = action.squeeze(0).cpu()
        self.log_probs[i] = log_prob.cpu()
        self.rewards[i]   = torch.tensor(reward, dtype=torch.float32)
        self.values[i]    = value.cpu()
        self.dones[i]     = torch.tensor(float(done))
        self.ptr += 1

    def compute_returns(self, last_value, gamma: float, lam: float):
        last_val   = last_value.cpu().item()
        advantages = torch.zeros(self.steps)
        gae = 0.0
        for t in reversed(range(self.steps)):
            nv    = last_val if t == self.steps - 1 else self.values[t + 1].item()
            nd    = 0.0      if t == self.steps - 1 else self.dones[t + 1].item()
            delta = self.rewards[t] + gamma * nv * (1 - nd) - self.values[t]
            gae   = delta + gamma * lam * (1 - nd) * gae
            advantages[t] = gae
        self.returns    = advantages + self.values
        self.advantages = advantages
        self.ptr = 0

    def get_minibatches(self, batch_size: int, device):
        idx = torch.randperm(self.steps)
        for start in range(0, self.steps, batch_size):
            mb  = idx[start : start + batch_size]
            img = self.imgs[mb].to(device) if self.has_img else None
            vec = self.vecs[mb].to(device) if self.has_vec else None
            yield (
                (img, vec),
                self.actions[mb].to(device),
                self.log_probs[mb].to(device),
                self.returns[mb].to(device),
                self.advantages[mb].to(device),
            )


# ─────────────────────────────────────────────
# PPO Trainer
# ─────────────────────────────────────────────

class PPOTrainer:
    """
    PPO trainer with dual-stream CNN + MLP encoder for tmrl.

    Quickstart
    ----------
    >>> import tmrl
    >>> env = tmrl.get_environment()
    >>> trainer, first_obs = PPOTrainer.from_env(env)
    >>> trainer.train(env, first_obs=first_obs, total_steps=2_000_000)
    """

    def __init__(
        self,
        img_channels: int | None = 4,   # number of stacked frames; None = no vision
        img_h: int = 64,
        img_w: int = 64,
        vec_dim: int | None = 3,        # dimension of vector obs; None = no vector
        cnn_out: int = 256,
        vec_out: int = 64,
        enc_dim: int = 256,
        action_dim: int = 3,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.001,
        max_grad_norm: float = 0.5,
        rollout_steps: int = 4096,
        n_epochs: int = 10,
        batch_size: int = 128,
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

        cnn = CNNStream(img_channels, img_h, img_w, cnn_out) if img_channels else None
        vec = VecStream(vec_dim, vec_out)                     if vec_dim      else None

        encoder = DualStreamEncoder(
            cnn, vec,
            cnn_out_dim = cnn_out if img_channels else 0,
            vec_out_dim = vec_out if vec_dim      else 0,
            enc_dim     = enc_dim,
        )

        self.policy    = ActorCritic(encoder, enc_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        img_shape = (img_channels, img_h, img_w) if img_channels else None
        self.buffer = RolloutBuffer(rollout_steps, img_shape, vec_dim, action_dim, self.device)

        n_params = sum(p.numel() for p in self.policy.parameters())
        cnn_str  = f"{img_channels}×{img_h}×{img_w}" if img_channels else "off"
        vec_str  = f"{vec_dim}-d"                     if vec_dim      else "off"
        print(f"PPO  device={self.device}  params={n_params:,}")
        print(f"     CNN stream : {cnn_str}")
        print(f"     Vec stream : {vec_str}")

    @classmethod
    def from_env(cls, env, **kwargs):
        """Auto-detect observation dimensions from a live tmrl environment."""
        obs, _ = env.reset()
        img_ch, img_h, img_w, vec_dim = probe_obs_dims(obs)
        action_dim = env.action_space.shape[0]
        print(f"Auto-detected: img={'%d×%d×%d' % (img_ch, img_h, img_w) if img_ch else None}  "
              f"vec={vec_dim}  actions={action_dim}")
        trainer = cls(
            img_channels = img_ch,
            img_h        = img_h or 64,
            img_w        = img_w or 64,
            vec_dim      = vec_dim,
            action_dim   = action_dim,
            **kwargs,
        )
        return trainer, obs   # return first obs so env isn't reset twice

    # ── PPO update ───────────────────────────────────────────────────

    def _update(self):
        stats = dict(policy_loss=[], value_loss=[], entropy=[], approx_kl=[])

        for _ in range(self.n_epochs):
            for obs_mb, actions, old_log_probs, returns, advantages in \
                    self.buffer.get_minibatches(self.batch_size, self.device):

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                dist, values = self.policy(obs_mb)
                actions_raw  = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
                log_probs    = dist.log_prob(actions_raw).sum(-1) \
                               - torch.log(1 - actions.pow(2) + 1e-6).sum(-1)
                entropy      = dist.entropy().sum(-1).mean()

                ratio       = (log_probs - old_log_probs).exp()
                surr1       = ratio * advantages
                surr2       = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = 0.5 * (values - returns).pow(2).mean()
                loss        = policy_loss + self.value_coef * value_loss \
                              - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs - log_probs).mean().item()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["approx_kl"].append(approx_kl)

        return {k: np.mean(v) for k, v in stats.items()}

    # ── training loop ────────────────────────────────────────────────

    def train(self, env, first_obs=None, total_steps: int = 2_000_000, log_interval: int = 5, current_run_folder: CurrentRunFolder = CurrentRunFolder("./run")):
        obs = first_obs if first_obs is not None else env.reset()[0]
        ep_reward, ep_count = 0.0, 0
        recent_rewards = deque(maxlen=20)
        global_step, rollout_idx = 0, 0
        start_time = time.time()
        
        number_of_crashes = 0
        crash_penalty_running_sum = 0
        crash_speed_running_sum = 0

        while global_step < total_steps:
            # ── collect rollout ──────────────────────────────────────
            self.policy.eval()
            for _ in range(self.rollout_steps):
                img_t, vec_t = obs_to_tensors(obs, self.device)
                with torch.no_grad():
                    action, log_prob, value = self.policy.act((img_t, vec_t))

                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated
                
                if("crash_penalty" in info):
                    number_of_crashes += 1
                    crash_penalty_running_sum += info["crash_penalty"]
                    crash_speed_running_sum += info["crash_speed"]

                self.buffer.add(img_t, vec_t, action, log_prob, reward, value, done)
                obs        = next_obs
                ep_reward += reward
                global_step += 1

                if done: 
                    recent_rewards.append(ep_reward)
                    ep_count  += 1
                    ep_reward  = 0.0
                    obs, _     = env.reset()

                if global_step >= total_steps:
                    break

            # Bootstrap value
            with torch.no_grad():
                img_t, vec_t = obs_to_tensors(obs, self.device)
                _, _, last_val = self.policy.act((img_t, vec_t))
            self.buffer.compute_returns(last_val.squeeze(0), self.gamma, self.gae_lambda)


            mean_crash_penalty = crash_penalty_running_sum / number_of_crashes if number_of_crashes > 0 else 0
            mean_crash_speed = crash_speed_running_sum / number_of_crashes if number_of_crashes > 0 else 0

            # ── update policy ────────────────────────────────────────
            self.policy.train()
            stats = self._update()
            rollout_idx += 1

            if rollout_idx % log_interval == 0:
                elapsed  = time.time() - start_time
                mean_rew = np.mean(recent_rewards) if recent_rewards else float("nan")
                
                checkpoint_file = current_run_folder.get_date_file_name("pt", "chkpts")
                self.save(checkpoint_file)
                
                with open(current_run_folder.get_file_name("data.csv"), "a+") as f:
                    f.write(f"{global_step},{mean_rew},{ep_count},{stats['policy_loss']},{stats['value_loss']},{stats['entropy']},{stats['approx_kl']},{global_step / elapsed},\"{checkpoint_file}\",{mean_crash_penalty},{mean_crash_speed},{number_of_crashes}\n")
                
                print(
                    f"\n\n[{global_step:>8d}] "
                    f"rew={mean_rew:>8.2f}  ep={ep_count:>4d}  "
                    f"pi={stats['policy_loss']:>7.4f}  V={stats['value_loss']:>7.4f}  "
                    f"H={stats['entropy']:>5.3f}  kl={stats['approx_kl']:>6.4f}  "
                    f"fps={global_step / elapsed:>5.0f} crash_penalty={mean_crash_penalty} crash_speed={mean_crash_speed} number_of_crashes={number_of_crashes}\n\n"
                )
                
                number_of_crashes = 0
                crash_speed_running_sum = 0
                crash_penalty_running_sum = 0

        env.close()
        print("Training complete.")

    # ── checkpointing ────────────────────────────────────────────────

    def save(self, path: str = "ppo_tmrl.pt"):
        torch.save({"policy": self.policy.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)
        print(f"Saved → {path}")

    def load(self, path: str = "ppo_tmrl.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Loaded ← {path}")



# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import tmrl
    import pathlib
    from tmrl.config.config_objects import CONFIG_DICT
    import os
    # import sys
    
    print(CONFIG_DICT)
    
    folder = "runs/large-model-ppo"
    current_run_folder = None
    if(folder is not None):
        current_run_folder = CurrentRunFolder(str(pathlib.Path(folder)))
    else:
        s = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        current_run_folder = CurrentRunFolder(str(pathlib.Path("runs", s)))
    
    
    env = tmrl.get_environment()
    # env = CrashPenaltyWrapper(
    #     tmrl.get_environment(),
    #     base_penalty = 1.0,   # flat penalty on any crash
    #     speed_coef   = 0.04,    # extra penalty per km/h at time of crash
    #     max_penalty  = 5.0,  # cap so one crash cant blow up training
    # )

    trainer, first_obs = PPOTrainer.from_env(
        env,
        rollout_steps = 4096,
        n_epochs      = 10,
        batch_size    = 128,
        lr            = 1e-4,
        entropy_coef  = 0.001,
    )

    chkpts = sorted(os.listdir(current_run_folder.get_folder("chkpts")))
    if(0 < len(chkpts)):
        print(f"Loading old checkpoints \"{str(pathlib.Path(current_run_folder.get_file_name(chkpts[-1], "chkpts")))}\"")
        trainer.load(current_run_folder.get_file_name(chkpts[-1], "chkpts"))
    
    trainer.train(env, first_obs=first_obs, total_steps=2_000_000, current_run_folder=current_run_folder)
    trainer.save(current_run_folder.get_file_name("ppo_final.pt", "chkpts"))
