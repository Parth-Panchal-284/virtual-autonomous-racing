"""
REDQ (Randomized Ensemble Double Q-learning) for TMRL — vision only.

Paper: "Randomized Ensembled Double Q-Learning: Learning Fast Without a Model"
       Chen et al., ICLR 2021.  https://arxiv.org/abs/2101.05982

Key ideas vs SAC
----------------
1. ENSEMBLE  : N independent Q-networks (default N=10) instead of 2.
2. IN-TARGET : At each update, randomly sample M of the N critics (default M=2),
               take the minimum Q value as the Bellman target.
               This gives a much tighter upper bound than classic double-Q.
3. HIGH UTD  : G gradient steps per environment step (default G=20).
               This makes REDQ extremely sample-efficient — important for
               real-time environments like TrackMania where collecting data
               is slow (the game runs at real-time speed).

Architecture (vision only — no lidar)
--------------------------------------
  Stacked grayscale frames  (C x H x W)
           |
       CNN encoder  (shared between actor and ALL critics)
           |
        512-d latent
       /           \\
    Actor          Q1 ... QN   (each critic gets latent + action concatenated)
  (Gaussian        (independent MLP heads)
   policy with
   auto-alpha)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
import time
import tmrl

import threading
# Reuse obs helpers from ppo_tmrl — assumes both files are in the same directory
from ppo_large import split_obs, probe_obs_dims

from datetime import datetime

from util import CurrentRunFolder

# ─────────────────────────────────────────────
# Obs → tensor (image only)
# ─────────────────────────────────────────────
 
def obs_to_img_tensor(obs, device: torch.device) -> torch.Tensor:
    """
    Extract image frames from a tmrl obs tuple and return a (1, C, H, W) tensor.
 
    Handles two layouts tmrl may use:
      A) 8 separate (1, H, W) arrays in the tuple  -> stack into (8, H, W)
      B) one pre-stacked (8, H, W) array in the tuple -> use directly
      C) one (H, W) array per frame                -> stack into (N, H, W)
    """
    img_arrays, _ = split_obs(obs)
    if not img_arrays:
        raise ValueError(
            "No image arrays found in observation. "
            "Check that tmrl is configured to return pixel observations."
        )
 
    frames = []
    for img in img_arrays:
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 2:
            # Plain (H, W) — one grayscale frame
            frames.append(arr)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            # (1, H, W) — single-channel frame, strip the channel dim
            frames.append(arr[0])
        elif arr.ndim == 3:
            # (C, H, W) where C > 1 — already stacked frames, unpack each
            for i in range(arr.shape[0]):
                frames.append(arr[i])
        else:
            raise ValueError(f"Unexpected image array shape: {arr.shape}")
 
    stacked = np.stack(frames, axis=0)                          # (C, H, W)
    return torch.from_numpy(stacked).unsqueeze(0).to(device)   # (1, C, H, W)

# ─────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────

class ReplayBuffer:
    """
    Standard off-policy replay buffer storing (obs_img, action, reward, next_img, done).
    Images are stored as uint8 to save memory (~8x smaller than float32).
    """
 
    def __init__(self, capacity: int, img_shape: tuple, action_dim: int, device: torch.device):
        self.capacity   = capacity
        self.device     = device
        self.ptr        = 0
        self.size       = 0
 
        C, H, W = img_shape
        # uint8 saves ~4x memory vs float32 for pixel buffers
        self.imgs      = np.zeros((capacity, C, H, W), dtype=np.uint8)
        self.next_imgs = np.zeros((capacity, C, H, W), dtype=np.uint8)
        self.actions   = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards   = np.zeros((capacity, 1),          dtype=np.float32)
        self.dones     = np.zeros((capacity, 1),          dtype=np.float32)
 
    def add(self, img, action, reward, next_img, done):
        """img / next_img should be (C,H,W) numpy arrays in [0,255]."""
        i = self.ptr
        self.imgs[i]      = img.astype(np.uint8)
        self.next_imgs[i] = next_img.astype(np.uint8)
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.dones[i]     = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
 
    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.imgs[idx].astype(np.float32)).to(self.device),
            torch.from_numpy(self.actions[idx]).to(self.device),
            torch.from_numpy(self.rewards[idx]).to(self.device),
            torch.from_numpy(self.next_imgs[idx].astype(np.float32)).to(self.device),
            torch.from_numpy(self.dones[idx]).to(self.device),
        )
 
    def __len__(self):
        return self.size


# ─────────────────────────────────────────────
# Shared CNN encoder
# ─────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Shared convolutional encoder used by both the actor and all critics.
    Sharing the encoder dramatically reduces memory vs giving each critic
    its own CNN, while still allowing independent Q-value heads.

    Input : (B, C, H, W)  uint8-cast float, range [0, 255]
    Output: (B, enc_dim)
    """

    def __init__(self, in_channels: int, img_h: int, img_w: int, enc_dim: int = 512):
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
            nn.Linear(flat, enc_dim), nn.ReLU(),
            nn.Linear(enc_dim, enc_dim), nn.ReLU(),
        )
        self.enc_dim = enc_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x / 255.0))


# ─────────────────────────────────────────────
# Actor (SAC-style Gaussian policy)
# ─────────────────────────────────────────────

class Actor(nn.Module):
    """
    Squashed Gaussian actor.  Takes the CNN latent and outputs a
    tanh-squashed action in (-1, 1)^action_dim with a corrected log-prob.
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX =  2

    def __init__(self, enc_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,  hidden), nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden, action_dim)
        self.log_std_layer = nn.Linear(hidden, action_dim)

        # orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)

    def forward(self, latent: torch.Tensor):
        h       = self.net(latent)
        mean    = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()
        dist    = Normal(mean, std)

        x       = dist.rsample()                          # reparameterised sample
        action  = torch.tanh(x)
        log_prob = dist.log_prob(x).sum(-1, keepdim=True) \
                   - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act(self, latent: torch.Tensor):
        action, log_prob = self(latent)
        return action, log_prob


# ─────────────────────────────────────────────
# Critic ensemble
# ─────────────────────────────────────────────

class QNetwork(nn.Module):
    """Single Q-network head: takes (latent, action) -> scalar Q value."""

    def __init__(self, enc_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),               nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, action], dim=-1))


class QEnsemble(nn.Module):
    """
    N independent Q-networks sharing the CNN encoder.

    The encoder is shared to save memory — with N=10 critics, giving each
    its own CNN would multiply memory usage by 10x.  The independent MLP
    heads still provide diverse Q-value estimates for the in-target trick.
    """

    def __init__(self, encoder: CNNEncoder, action_dim: int, N: int = 10, hidden: int = 512):
        super().__init__()
        self.encoder = encoder
        self.critics = nn.ModuleList([
            QNetwork(encoder.enc_dim, action_dim, hidden) for _ in range(N)
        ])
        self.N = N

    def forward(self, img: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns (B, N) tensor of Q values from all critics."""
        latent = self.encoder(img)
        qs = torch.cat([c(latent, action) for c in self.critics], dim=-1)  # (B, N)
        return qs

    def subset_min(self, img: torch.Tensor, action: torch.Tensor, indices: list) -> torch.Tensor:
        """
        Compute Q values for a random subset of critics and return the minimum.
        This is the core REDQ in-target trick.
        Returns (B, 1).
        """
        latent = self.encoder(img)
        qs = torch.stack([self.critics[i](latent, action) for i in indices], dim=-1)  # (B, 1, M)
        return qs.min(dim=-1).values                                                   # (B, 1)


# ─────────────────────────────────────────────
# REDQ Trainer
# ─────────────────────────────────────────────

class REDQTrainer:
    """
    REDQ trainer for tmrl — vision only (no lidar).

    Quickstart
    ----------
    >>> import tmrl
    >>> env = tmrl.get_environment()
    >>> trainer, first_obs = REDQTrainer.from_env(env)
    >>> trainer.train(env, first_obs=first_obs, total_steps=500_000)

    Key hyperparameters
    -------------------
    N   : ensemble size          (more = better estimates, more memory)
    M   : in-target subset size  (always M < N)
    G   : UTD ratio              (gradient steps per env step)
    """

    def __init__(
        self,
        img_channels: int,
        img_h: int,
        img_w: int,
        action_dim: int = 3,
        # REDQ core
        N: int   = 10,           # ensemble size
        M: int   = 2,            # in-target subset size
        G: int   = 20,           # UTD ratio (gradient updates per env step)
        # network
        enc_dim: int    = 512,
        hidden: int     = 512,
        # SAC hyperparams
        lr: float       = 3e-4,
        gamma: float    = 0.99,
        tau: float      = 0.005,  # soft target update rate
        alpha: float    = 0.2,    # initial entropy temperature
        auto_alpha: bool = True,  # learn alpha automatically
        target_entropy: float | None = None,
        # buffer
        buffer_size: int    = 200_000,
        batch_size: int     = 256,
        warmup_steps: int   = 1_000,  # random actions before training starts
        device: str         = "auto",
    ):
        assert M < N, f"In-target subset size M={M} must be < ensemble size N={N}"

        self.N            = N
        self.M            = M
        self.G            = G
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.warmup_steps = warmup_steps
        self.img_shape    = (img_channels, img_h, img_w)
        self.action_dim   = action_dim

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )

        # ── networks ────────────────────────────────────────────────
        # Shared encoder (used by both actor and critic ensemble)
        self.encoder        = CNNEncoder(img_channels, img_h, img_w, enc_dim).to(self.device)
        self.encoder_target = CNNEncoder(img_channels, img_h, img_w, enc_dim).to(self.device)
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        self.actor   = Actor(enc_dim, action_dim, hidden).to(self.device)
        self.critics = QEnsemble(self.encoder, action_dim, N, hidden).to(self.device)

        # Target critic ensemble (separate encoder copy for stable targets)
        self.critics_target = QEnsemble(self.encoder_target, action_dim, N, hidden).to(self.device)
        self.critics_target.load_state_dict(self.critics.state_dict())

        # ── optimisers ──────────────────────────────────────────────
        self.actor_opt   = optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()), lr=lr)
        self.critic_opt  = optim.Adam(self.critics.parameters(), lr=lr)

        # ── auto-alpha (entropy temperature) ────────────────────────
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = target_entropy or -float(action_dim)
            self.log_alpha      = torch.tensor(np.log(alpha), requires_grad=True,
                                               device=self.device, dtype=torch.float32)
            self.alpha_opt      = optim.Adam([self.log_alpha], lr=lr)
            self.alpha          = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # ── replay buffer ────────────────────────────────────────────
        self.buffer = ReplayBuffer(buffer_size, self.img_shape, action_dim, self.device)

        n_params = sum(p.numel() for p in list(self.encoder.parameters())
                       + list(self.actor.parameters())
                       + list(self.critics.parameters()))
        print(f"REDQ  device={self.device}  N={N}  M={M}  G={G}  params={n_params:,}")
        print(f"      CNN : {img_channels}x{img_h}x{img_w}  enc_dim={enc_dim}")

    @classmethod
    def from_env(cls, env, **kwargs):
        """Auto-detect image dimensions from a live tmrl environment."""
        obs, _ = env.reset()
        img_ch, img_h, img_w, _ = probe_obs_dims(obs)
        if img_ch is None:
            raise ValueError("No image observations found. Check tmrl is configured for pixels.")
        action_dim = env.action_space.shape[0]
        print(f"Auto-detected: img={img_ch}x{img_h}x{img_w}  actions={action_dim}")
        trainer = cls(
            img_channels = img_ch,
            img_h        = img_h,
            img_w        = img_w,
            action_dim   = action_dim,
            **kwargs,
        )
        return trainer, obs

    # ── soft target update ───────────────────────────────────────────

    @torch.no_grad()
    def _soft_update(self):
        """Polyak averaging for both encoder and critic targets."""
        for p, pt in zip(self.encoder.parameters(), self.encoder_target.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.critics.critics.parameters(),
                         self.critics_target.critics.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

    # ── one REDQ gradient step ───────────────────────────────────────

    def _update(self):
        imgs, actions, rewards, next_imgs, dones = self.buffer.sample(self.batch_size)

        # ── critic update ────────────────────────────────────────────
        with torch.no_grad():
            # Sample next action from current actor
            next_latent            = self.encoder(next_imgs)
            next_actions, next_lp  = self.actor(next_latent)

            # REDQ in-target: random subset of M critics, take min
            subset_idx   = random.sample(range(self.N), self.M)
            next_q       = self.critics_target.subset_min(next_imgs, next_actions, subset_idx)
            target_q     = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_lp)

        # All N critics update toward the same target
        all_q       = self.critics(imgs, actions)             # (B, N)
        target_q_ex = target_q.expand(-1, self.N)            # (B, N)
        critic_loss = F.mse_loss(all_q, target_q_ex)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── actor update ─────────────────────────────────────────────
        latent             = self.encoder(imgs)
        new_actions, lp    = self.actor(latent)
        # Use mean Q across all N critics for actor gradient
        q_vals             = self.critics(imgs, new_actions).mean(dim=-1, keepdim=True)
        actor_loss         = (self.alpha * lp - q_vals).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── alpha update ─────────────────────────────────────────────
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()

        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss,
            "alpha":       self.alpha,
            "q_mean":      q_vals.mean().item(),
        }
        
    def _update_loop(self, stop_event: threading.Event, stat_queue: list, lock: threading.Lock):
        """
        Background thread: runs gradient updates continuously while the
        main thread steps the environment. This decouples learning from
        data collection so rtgym's real-time clock is never blocked.
        """
        while not stop_event.is_set():
            if len(self.buffer) < self.batch_size:
                time.sleep(0.01)   # wait for buffer to fill
                continue
 
            stats = None
            for _ in range(self.G):
                stats = self._update()
 
            if stats is not None:
                with lock:
                    stat_queue.append(stats)
    
    def train(self, env : tmrl.GenericGymEnv, first_obs=None, total_steps: int = 500_000, log_interval: int = 1_000, current_run_folder: CurrentRunFolder = CurrentRunFolder("./run")):
        obs = first_obs if first_obs is not None else env.reset()[0]
        ep_reward, ep_count = 0.0, 0
        recent_rewards : deque[float]  = deque(maxlen=20)
        stat_queue: list = []          # background thread pushes stats here
        lock            = threading.Lock()
        global_step     = 0
        start_time      = time.time()
 
        print(f"Warming up for {self.warmup_steps} steps with random actions...")
        print(f"Gradient updates run in background thread (G={self.G} per cycle).")
 
        # ── start background update thread after warmup ──────────────
        stop_event   = threading.Event()
        update_thread = None   # started once warmup is done
 
        for _ in range(self.warmup_steps):
            action_np = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            reward = float(reward)
            done = terminated or truncated
            img_np      = obs_to_img_tensor(obs,      self.device).squeeze(0).cpu().numpy()
            next_img_np = obs_to_img_tensor(next_obs, self.device).squeeze(0).cpu().numpy()
            self.buffer.add(img_np, action_np, reward, next_img_np, done)
 
            obs        = next_obs
            ep_reward += reward
            global_step += 1
 
            if done:
                if("termination_reason" in info):
                    print(f"Terminated: {info["termination_reason"]}")
                print(f"Terminated, Score: {ep_reward}")
                recent_rewards.append(ep_reward)
                ep_count  += 1
                ep_reward  = 0.0
                obs, _     = env.reset()
            
 
        update_thread = threading.Thread(
            target=self._update_loop,
            args=(stop_event, stat_queue, lock),
            daemon=True,
        )
        update_thread.start()
        print("Background update thread started.")



        n_crashes = 0
        n_episodes = 0
        n_no_motion = 0
 
        while global_step < total_steps:
            img_t = obs_to_img_tensor(obs, self.device)
            with torch.no_grad():
                latent = self.encoder(img_t)
                action, _ = self.actor.act(latent)
            action_np = action.squeeze(0).cpu().numpy()
 
            # ── env step (never blocked by gradient updates) ─────────
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            reward = float(reward)
            done = terminated or truncated
 
            img_np      = obs_to_img_tensor(obs,      self.device).squeeze(0).cpu().numpy()
            next_img_np = obs_to_img_tensor(next_obs, self.device).squeeze(0).cpu().numpy()
            self.buffer.add(img_np, action_np, reward, next_img_np, done)
 
            obs        = next_obs
            ep_reward += reward
            global_step += 1
 
            if done:
                if("termination_reason" in info):
                    print(f"Terminated: {info["termination_reason"]}")
                    match(info["termination_reason"]):
                        case "no-movement":
                            n_no_motion += 1
                        case "crash":
                            n_crashes += 1
                n_episodes += 1
                print(f"Terminated, Score: {ep_reward}")
                recent_rewards.append(ep_reward)
                ep_count  += 1
                ep_reward  = 0.0
                obs, _     = env.reset()
 
            if global_step % log_interval == 0:
                with lock:
                    recent_stats = list(stat_queue)
                    stat_queue.clear()
 
                checkpoint_file = current_run_folder.get_date_file_name("pt", "chkpts")
                self.save(checkpoint_file)
                
                
                if recent_stats:
                    elapsed  = time.time() - start_time
                    mean_rew = np.mean(recent_rewards) if recent_rewards else float("nan")
                    
                    with open(current_run_folder.get_file_name("data.csv"), "a+") as f:
                        f.write(f"{global_step},{mean_rew},{ep_count},"
                                f"{np.mean([s['critic_loss'] for s in recent_stats])},"
                                f"{np.mean([s['actor_loss']  for s in recent_stats])},"
                                f"{np.mean([s['alpha']       for s in recent_stats])},"
                                f"{np.mean([s['q_mean']          for s in recent_stats])},"
                                f"{global_step / elapsed},\"{checkpoint_file}\","
                                f"{n_episodes},{n_crashes},{n_no_motion}"
                                f"\n")
                    print(
                        f"[{global_step:>7d}] "
                        f"rew={mean_rew:>8.2f}  ep={ep_count:>4d}  "
                        f"critic={np.mean([s['critic_loss'] for s in recent_stats]):>7.4f}  "
                        f"actor={np.mean([s['actor_loss']  for s in recent_stats]):>7.4f}  "
                        f"alpha={np.mean([s['alpha']       for s in recent_stats]):.4f}  "
                        f"Q={np.mean([s['q_mean']          for s in recent_stats]):>7.3f}  "
                        f"fps={global_step / elapsed:>5.0f}  "
                        f"n_eps={n_episodes}  n_crashes={n_crashes}  n_no_motion={n_no_motion}"
                    )
                    
                    n_episodes = 0
                    n_crashes = 0
                    n_no_motion = 0
 
        # ── shut down background thread cleanly ──────────────────────
        stop_event.set()
        if update_thread is not None:
            update_thread.join(timeout=5.0)
 
        env.close()
        print("Training complete.")

    def save(self, path: str = "redq_tmrl.pt"):
        torch.save({
            "encoder":        self.encoder.state_dict(),
            "encoder_target": self.encoder_target.state_dict(),
            "actor":          self.actor.state_dict(),
            "critics":        self.critics.state_dict(),
            "critics_target": self.critics_target.state_dict(),
            "log_alpha":      self.log_alpha if self.auto_alpha else None,
        }, path)
        print(f"Saved → {path}")

    def load(self, path: str = "redq_tmrl.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.encoder_target.load_state_dict(ckpt["encoder_target"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critics.load_state_dict(ckpt["critics"])
        self.critics_target.load_state_dict(ckpt["critics_target"])
        if self.auto_alpha and ckpt["log_alpha"] is not None:
            self.log_alpha.data = ckpt["log_alpha"].data
            self.alpha = self.log_alpha.exp().item()
        print(f"Loaded ← {path}")


if __name__ == "__main__":
    try:
        import tmrl
        from envs.crash_penalty import CrashPenaltyWrapper
        from envs.no_movement_penalty import NoMovementPenalty
        import pathlib
        import os
        
        import warnings
        warnings.simplefilter("ignore")
    
        folder = "runs/redq"
        current_run_folder = None
        if(folder is not None):
            current_run_folder = CurrentRunFolder(str(pathlib.Path(folder)))
        else:
            s = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
            current_run_folder = CurrentRunFolder(str(pathlib.Path("runs", s)))

        env = tmrl.get_environment()
        env = CrashPenaltyWrapper(
            env,
            base_penalty      = 0.0,
            speed_coef = 0.0,
            early_stop= True
        )
        env = NoMovementPenalty(
            env, 2, 40
        )

        trainer, first_obs = REDQTrainer.from_env(
            env,
            N             = 10,       # ensemble size
            M             = 2,        # in-target subset
            G             = 20,       # UTD ratio
            enc_dim       = 512,
            hidden        = 512,
            lr            = 3e-4,
            gamma         = 0.99,
            tau           = 0.005,
            buffer_size   = 200_000,
            batch_size    = 256,
            warmup_steps  = 1_000,
        )

        chkpts = sorted(os.listdir(current_run_folder.get_folder("chkpts")))
        if(0 < len(chkpts)):
            print(f"Loading old checkpoints \"{str(pathlib.Path(current_run_folder.get_file_name(chkpts[-1], "chkpts")))}\"")
            trainer.load(current_run_folder.get_file_name(chkpts[-1], "chkpts"))
    

        trainer.train(env, first_obs=first_obs, total_steps=500_000, current_run_folder=current_run_folder, log_interval=1_000)
        trainer.save(current_run_folder.get_file_name("redq_tmrl_final.pt", "chkpts"))

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure ppo_tmrl.py is in the same directory and tmrl is installed.")