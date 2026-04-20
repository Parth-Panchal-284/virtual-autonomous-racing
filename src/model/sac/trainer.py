from collections import deque
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.sac.actor_critic import SquashedGaussianActor, TwinQCritic
from model.sac.sac import build_encoder
from util import CurrentRunFolder, split_obs


def _parse_obs(obs) -> tuple[np.ndarray | None, np.ndarray | None]:
    img_arrays, vec_arrays = split_obs(obs)

    img_np = None
    if img_arrays:
        frames = []
        for img in img_arrays:
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 2:
                frames.append(arr)
            elif arr.ndim == 3 and arr.shape[0] == 1:
                frames.append(arr[0])
            elif arr.ndim == 3:
                for i in range(arr.shape[0]):
                    frames.append(arr[i])
            else:
                raise ValueError(f"Unexpected image shape in observation: {arr.shape}")
        img_np = np.stack(frames, axis=0)

    vec_np = None
    if vec_arrays:
        vec_np = np.concatenate([v.flatten().astype(np.float32) for v in vec_arrays], axis=0)

    return img_np, vec_np


def _obs_to_tensors(obs, device: torch.device):
    img_np, vec_np = _parse_obs(obs)
    img_t = torch.from_numpy(img_np).unsqueeze(0).to(device) if img_np is not None else None
    vec_t = torch.from_numpy(vec_np).unsqueeze(0).to(device) if vec_np is not None else None
    return img_t, vec_t


def _infer_obs_spec(obs):
    img_np, vec_np = _parse_obs(obs)
    img_channels = img_h = img_w = None
    if img_np is not None:
        img_channels, img_h, img_w = img_np.shape
    vec_dim = int(vec_np.shape[0]) if vec_np is not None else None
    return img_channels, img_h, img_w, vec_dim


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        action_dim: int,
        device: torch.device,
        img_shape: tuple[int, int, int] | None,
        vec_dim: int | None,
    ):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.has_img = img_shape is not None
        self.has_vec = vec_dim is not None and vec_dim > 0

        if self.has_img:
            c, h, w = img_shape  # type: ignore
            self.imgs = np.zeros((capacity, c, h, w), dtype=np.uint8)
            self.next_imgs = np.zeros((capacity, c, h, w), dtype=np.uint8)

        if self.has_vec:
            self.vecs = np.zeros((capacity, vec_dim), dtype=np.float32)  # type: ignore
            self.next_vecs = np.zeros((capacity, vec_dim), dtype=np.float32)  # type: ignore

        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs_img, obs_vec, action, reward, next_img, next_vec, done):
        i = self.ptr

        if self.has_img:
            self.imgs[i] = obs_img.astype(np.uint8)
            self.next_imgs[i] = next_img.astype(np.uint8)

        if self.has_vec:
            self.vecs[i] = obs_vec.astype(np.float32)
            self.next_vecs[i] = next_vec.astype(np.float32)

        self.actions[i] = action.astype(np.float32)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)

        img_t = None
        next_img_t = None
        vec_t = None
        next_vec_t = None

        if self.has_img:
            img_t = torch.from_numpy(self.imgs[idx].astype(np.float32)).to(self.device)
            next_img_t = torch.from_numpy(self.next_imgs[idx].astype(np.float32)).to(self.device)

        if self.has_vec:
            vec_t = torch.from_numpy(self.vecs[idx]).to(self.device)
            next_vec_t = torch.from_numpy(self.next_vecs[idx]).to(self.device)

        actions_t = torch.from_numpy(self.actions[idx]).to(self.device)
        rewards_t = torch.from_numpy(self.rewards[idx]).to(self.device)
        dones_t = torch.from_numpy(self.dones[idx]).to(self.device)

        return (img_t, vec_t), actions_t, rewards_t, (next_img_t, next_vec_t), dones_t

    def __len__(self):
        return self.size


class SACTrainer:
    def __init__(
        self,
        img_channels: int | None,
        img_h: int | None,
        img_w: int | None,
        vec_dim: int | None,
        action_dim: int,
        enc_dim: int = 512,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: float | None = None,
        batch_size: int = 256,
        buffer_size: int = 200_000,
        warmup_steps: int = 2_000,
        updates_per_step: int = 1,
        max_grad_norm_actor: float = 1.0,
        max_grad_norm_critic: float = 1.0,
        device: str = "auto",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.max_grad_norm_actor = max_grad_norm_actor
        self.max_grad_norm_critic = max_grad_norm_critic
        self.action_dim = action_dim

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        actor_encoder, enc_dim = build_encoder(
            img_channels=img_channels,
            img_h=img_h,
            img_w=img_w,
            vec_dim=vec_dim,
            enc_dim=enc_dim,
        )
        critic_encoder, _ = build_encoder(
            img_channels=img_channels,
            img_h=img_h,
            img_w=img_w,
            vec_dim=vec_dim,
            enc_dim=enc_dim,
        )
        critic_target_encoder, _ = build_encoder(
            img_channels=img_channels,
            img_h=img_h,
            img_w=img_w,
            vec_dim=vec_dim,
            enc_dim=enc_dim,
        )

        self.actor = SquashedGaussianActor(actor_encoder, enc_dim, action_dim).to(self.device)
        self.critic = TwinQCritic(critic_encoder, enc_dim, action_dim).to(self.device)
        self.critic_target = TwinQCritic(critic_target_encoder, enc_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
            self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, requires_grad=True, device=self.device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.target_entropy = target_entropy
            self.log_alpha = None
            self.alpha_opt = None
            self.alpha = init_alpha

        img_shape = None
        if img_channels is not None and img_h is not None and img_w is not None:
            img_shape = (img_channels, img_h, img_w)

        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            action_dim=action_dim,
            device=self.device,
            img_shape=img_shape,
            vec_dim=vec_dim,
        )

        param_count = sum(p.numel() for p in list(self.actor.parameters()) + list(self.critic.parameters()))
        print(f"SAC  device={self.device}  params={param_count:,}  alpha={self.alpha:.4f}")
        print(
            f"     obs_img={img_shape} vec_dim={vec_dim} action_dim={action_dim} "
            f"batch={batch_size} buffer={buffer_size}"
        )

    @classmethod
    def from_env(cls, env, **kwargs):
        obs, _ = env.reset()
        img_channels, img_h, img_w, vec_dim = _infer_obs_spec(obs)
        action_dim = int(env.action_space.shape[0])
        print(
            f"Auto-detected: img={img_channels}x{img_h}x{img_w} vec={vec_dim} actions={action_dim}"
        )
        trainer = cls(
            img_channels=img_channels,
            img_h=img_h,
            img_w=img_w,
            vec_dim=vec_dim,
            action_dim=action_dim,
            **kwargs,
        )
        return trainer, obs

    @torch.no_grad()
    def _soft_update(self):
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def _update(self):
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            next_q1, next_q2 = self.critic_target(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        cur_q1, cur_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(cur_q1, target_q) + F.mse_loss(cur_q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm_critic)
        self.critic_opt.step()

        new_actions, log_prob = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_prob - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm_actor)
        self.actor_opt.step()

        alpha_loss_value = 0.0
        if self.auto_alpha and self.log_alpha is not None and self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss_value = alpha_loss.item()

        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_value,
            "alpha": self.alpha,
            "q_mean": q_pi.mean().item(),
        }

    def train(
        self,
        env,
        first_obs=None,
        total_steps: int = 500_000,
        log_interval: int = 1_000,
        current_run_folder: CurrentRunFolder = CurrentRunFolder("./run"),
    ):
        obs = first_obs if first_obs is not None else env.reset()[0]
        ep_reward = 0.0
        ep_count = 0
        recent_rewards: deque[float] = deque(maxlen=20)

        global_step = 0
        start_time = time.time()

        n_crashes = 0
        n_episodes = 0
        n_no_motion = 0

        print(f"Warmup with random actions: {self.warmup_steps} steps")

        while global_step < total_steps:
            if global_step < self.warmup_steps:
                action_np = env.action_space.sample()
            else:
                obs_t = _obs_to_tensors(obs, self.device)
                with torch.no_grad():
                    action = self.actor.act(obs_t, deterministic=False)
                action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = bool(terminated or truncated)

            obs_img, obs_vec = _parse_obs(obs)
            next_img, next_vec = _parse_obs(next_obs)

            self.buffer.add(
                obs_img=obs_img,
                obs_vec=obs_vec,
                action=action_np,
                reward=float(reward),
                next_img=next_img,
                next_vec=next_vec,
                done=done,
            )

            if "termination_reason" in info:
                if info["termination_reason"] == "crash":
                    n_crashes += 1
                if info["termination_reason"] == "no-movement":
                    n_no_motion += 1

            obs = next_obs
            ep_reward += float(reward)
            global_step += 1

            update_stats = []
            if (
                global_step >= self.warmup_steps
                and len(self.buffer) >= self.batch_size
            ):
                for _ in range(self.updates_per_step):
                    update_stats.append(self._update())

            if done:
                recent_rewards.append(ep_reward)
                ep_count += 1
                n_episodes += 1
                ep_reward = 0.0
                obs, _ = env.reset()

            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                mean_rew = float(np.mean(recent_rewards)) if recent_rewards else float("nan")

                checkpoint_file = current_run_folder.get_date_file_name("pt", "chkpts")
                self.save(checkpoint_file)

                if update_stats:
                    mean_critic = float(np.mean([s["critic_loss"] for s in update_stats]))
                    mean_actor = float(np.mean([s["actor_loss"] for s in update_stats]))
                    mean_alpha = float(np.mean([s["alpha"] for s in update_stats]))
                    mean_q = float(np.mean([s["q_mean"] for s in update_stats]))
                else:
                    mean_critic = float("nan")
                    mean_actor = float("nan")
                    mean_alpha = float(self.alpha)
                    mean_q = float("nan")

                with open(current_run_folder.get_file_name("data.csv"), "a+") as f:
                    f.write(
                        f"{global_step},{mean_rew},{ep_count},"
                        f"{mean_critic},{mean_actor},{mean_alpha},{mean_q},"
                        f"{global_step / elapsed},\"{checkpoint_file}\","
                        f"{n_episodes},{n_crashes},{n_no_motion}"
                        f"\n"
                    )

                print(
                    f"[{global_step:>7d}] rew={mean_rew:>8.2f} ep={ep_count:>4d} "
                    f"critic={mean_critic:>8.4f} actor={mean_actor:>8.4f} "
                    f"alpha={mean_alpha:>7.4f} Q={mean_q:>8.3f} "
                    f"fps={global_step / elapsed:>5.0f} "
                    f"n_eps={n_episodes} n_crashes={n_crashes} n_no_motion={n_no_motion}"
                )

                n_crashes = 0
                n_episodes = 0
                n_no_motion = 0

        env.close()
        print("Training complete.")

    def save(self, path: str = "sac_tmrl.pt"):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu() if self.log_alpha is not None else None,
                "alpha": self.alpha,
            },
            path,
        )
        print(f"Saved -> {path}")

    def load(self, path: str = "sac_tmrl.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])

        if self.auto_alpha and self.log_alpha is not None and ckpt.get("log_alpha") is not None:
            self.log_alpha.data = ckpt["log_alpha"].to(self.device)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = float(ckpt.get("alpha", self.alpha))

        print(f"Loaded <- {path}")
