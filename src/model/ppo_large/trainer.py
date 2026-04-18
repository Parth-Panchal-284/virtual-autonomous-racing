from model.ppo_large.actor_critic import ActorCritic
from model.ppo_large.ppo import CNNStream, DualStreamEncoder, VecStream
from util import probe_obs_dims, split_obs, CurrentRunFolder
import torch.optim as optim
from collections import deque
import time
import tmrl
import torch
import numpy as np
import torch.nn as nn


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

class PPOTrainer:

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
    def from_env(cls, env: tmrl.GenericGymEnv, **kwargs):
        """Auto-detect observation dimensions from a live tmrl environment."""
        obs, _ = env.reset()
        img_ch, img_h, img_w, vec_dim = probe_obs_dims(obs)
        action_dim = env.action_space.shape[0] # type: ignore
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
        

class PPORunner:
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
        device: str = "auto"):

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

    # ── training loop ────────────────────────────────────────────────

    def run(self, env, first_obs=None, total_steps: int = 2_000_000, log_interval: int = 5, current_run_folder: CurrentRunFolder = CurrentRunFolder("./run"), run_file: str = "run.csv"):
        obs = first_obs if first_obs is not None else env.reset()[0]
        ep_reward, ep_count = 0.0, 0
        recent_rewards = deque(maxlen=20)
        global_step, rollout_idx = 0, 0
        start_time = time.time()

        while global_step < total_steps:
            # ── collect rollout ──────────────────────────────────────
            self.policy.eval()
            img_t, vec_t = obs_to_tensors(obs, self.device)
            with torch.no_grad():
                action, log_prob, value = self.policy.act((img_t, vec_t))

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            obs        = next_obs
            done = terminated or truncated
            ep_reward += reward
            global_step += 1

            if done:
                recent_rewards.append(ep_reward)
                ep_count  += 1
                ep_reward  = 0.0
                obs, _     = env.reset()

            if global_step >= total_steps:
                break

            if global_step % log_interval == 0:
                elapsed  = time.time() - start_time
                mean_rew = np.mean(recent_rewards) if recent_rewards else float("nan")
                
                with open(current_run_folder.get_file_name("run.csv")) as f:
                    f.write(f"{global_step},{mean_rew},{ep_count},,{global_step / elapsed}")
                
                print(
                    f"[{global_step:>8d}] "
                    f"rew={mean_rew:>8.2f}  ep={ep_count:>4d}  "
                    f"fps={global_step / elapsed:>5.0f}"
                )

        env.close()
        print("Training complete.")

    @classmethod
    def from_env(cls, env):
        """Auto-detect observation dimensions from a live tmrl environment."""
        obs, _ = env.reset()
        img_ch, img_h, img_w, vec_dim = probe_obs_dims(obs)
        action_dim = env.action_space.shape[0]
        print(f"Auto-detected: img={'%d×%d×%d' % (img_ch, img_h, img_w) if img_ch else None}  "
              f"vec={vec_dim}  actions={action_dim}")
        runner = cls(
            img_channels = img_ch,
            img_h        = img_h or 64,
            img_w        = img_w or 64,
            vec_dim      = vec_dim,
            action_dim   = action_dim,
        )
        return runner, obs   # return first obs so env isn't reset twice

    def load(self, path: str = "ppo_tmrl.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        # self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Loaded ← {path}")