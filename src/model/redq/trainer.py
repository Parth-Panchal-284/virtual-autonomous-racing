import torch
from model.redq.actor import SquashedGaussianMLPActor, QFunction, REDQActorCritic
import logging
import numpy as np

class REDQSACAgent():
    observation_space: type
    action_space: type
    device: torch.device | None = None  # device where the model will live (None for auto)
    model_cls: type = REDQActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    lr_entropy: float = 1e-3  # entropy autotuning
    learn_entropy_coef: bool = True
    target_entropy: float | None = None  # if None, the target entropy is set automatically
    n: int = 10  # number of REDQ parallel Q networks
    m: int = 2  # number of REDQ randomly sampled target networks
    q_updates_per_policy_update: int = 1  # in REDQ, this is the "UTD ratio" (20), this interplays with lr_actor

    # model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logging.debug(f" device REDQ-SAC: {device}")
        self.model = model.to(device)
        # self.model_target = no_grad(deepcopy(self.model))
        self.pi_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer_list = [torch.optim.Adam(q.parameters(), lr=self.lr_critic) for q in self.model.qs]
        self.criterion = torch.nn.MSELoss()
        self.loss_pi = torch.zeros((1,), device=device)

        self.i_update = 0  # for UTD ratio

        if self.target_entropy is None:  # automatic entropy coefficient
            self.target_entropy = -np.prod(action_space.shape)  # .astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    # def get_actor(self):
    #     return self.model_nograd.actor

    def train(self, batch):

        self.i_update += 1
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)

        o, a, r, o2, d, _ = batch

        if update_policy:
            pi, logp_pi = self.model.actor(o)
        # FIXME? log_prob = log_prob.reshape(-1, 1)

        loss_alpha = None
        if self.learn_entropy_coef and update_policy:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)

            sample_idxs = np.random.choice(self.n, self.m, replace=False)

            q_prediction_next_list = [self.model_target.qs[i](o2, a2) for i in sample_idxs]
            q_prediction_next_cat = torch.stack(q_prediction_next_list, -1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            backup = r.unsqueeze(dim=-1) + self.gamma * (1 - d.unsqueeze(dim=-1)) * (min_q - alpha_t * logp_a2.unsqueeze(dim=-1))

        q_prediction_list = [q(o, a) for q in self.model.qs]
        q_prediction_cat = torch.stack(q_prediction_list, -1)
        backup = backup.expand((-1, self.n)) if backup.shape[1] == 1 else backup

        loss_q = self.criterion(q_prediction_cat, backup)  # * self.n  # averaged for homogeneity with SAC

        for q in self.q_optimizer_list:
            q.zero_grad()
        loss_q.backward()

        if update_policy:
            for q in self.model.qs:
                q.requires_grad_(False)

            qs_pi = [q(o, pi) for q in self.model.qs]
            qs_pi_cat = torch.stack(qs_pi, -1)
            ave_q = torch.mean(qs_pi_cat, dim=1, keepdim=True)
            loss_pi = (alpha_t * logp_pi.unsqueeze(dim=-1) - ave_q).mean()
            self.pi_optimizer.zero_grad()
            loss_pi.backward()

            for q in self.model.qs:
                q.requires_grad_(True)

        for q_optimizer in self.q_optimizer_list:
            q_optimizer.step()

        if update_policy:
            self.pi_optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        if update_policy:
            self.loss_pi = loss_pi.detach()
        ret_dict = dict(
            loss_actor=self.loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict

    # @classmethod
    # def from_env(cls, env, **kwargs):
    #     """Auto-detect image dimensions from a live tmrl environment."""
    #     obs, _ = env.reset()
    #     img_ch, img_h, img_w, _ = probe_obs_dims(obs)
    #     if img_ch is None:
    #         raise ValueError("No image observations found. Check tmrl is configured for pixels.")
    #     action_dim = env.action_space.shape[0]
    #     print(f"Auto-detected: img={img_ch}x{img_h}x{img_w}  actions={action_dim}")
    #     trainer = cls(
    #         img_channels = img_ch,
    #         img_h        = img_h,
    #         img_w        = img_w,
    #         action_dim   = action_dim,
    #         **kwargs,
    #     )
    #     return trainer, obs

    # ── soft target update ───────────────────────────────────────────

    # @torch.no_grad()
    # def _soft_update(self):
    #     """Polyak averaging for both encoder and critic targets."""
    #     for p, pt in zip(self.encoder.parameters(), self.encoder_target.parameters()):
    #         pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
    #     for p, pt in zip(self.critics.critics.parameters(),
    #                      self.critics_target.critics.parameters()):
    #         pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

    # ── one REDQ gradient step ───────────────────────────────────────

    # def _update(self):
    #     imgs, actions, rewards, next_imgs, dones = self.buffer.sample(self.batch_size)

    #     # ── critic update ────────────────────────────────────────────
    #     with torch.no_grad():
    #         # Sample next action from current actor
    #         next_latent            = self.encoder(next_imgs)
    #         next_actions, next_lp  = self.actor(next_latent)

    #         # REDQ in-target: random subset of M critics, take min
    #         subset_idx   = random.sample(range(self.N), self.M)
    #         next_q       = self.critics_target.subset_min(next_imgs, next_actions, subset_idx)
    #         target_q     = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_lp)

    #     # All N critics update toward the same target
    #     all_q       = self.critics(imgs, actions)             # (B, N)
    #     target_q_ex = target_q.expand(-1, self.N)            # (B, N)
    #     critic_loss = F.mse_loss(all_q, target_q_ex)

    #     self.critic_opt.zero_grad()
    #     critic_loss.backward()
    #     self.critic_opt.step()

    #     # ── actor update ─────────────────────────────────────────────
    #     latent = self.encoder(imgs).detach() 
    #     new_actions, lp = self.actor(latent)
    #     # Use mean Q across all N critics for actor gradient
    #     q_vals             = self.critics(imgs, new_actions).mean(dim=-1, keepdim=True)
    #     actor_loss         = (self.alpha * lp - q_vals).mean()

    #     self.actor_opt.zero_grad()
    #     actor_loss.backward()
    #     self.actor_opt.step()

    #     # ── alpha update ─────────────────────────────────────────────
    #     alpha_loss = 0.0
    #     if self.auto_alpha:
    #         alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
    #         self.alpha_opt.zero_grad()
    #         alpha_loss.backward()
    #         self.alpha_opt.step()
            
    #         with torch.no_grad():
    #             self.log_alpha.clamp_(min=np.log(self.min_alpha))
    #         self.alpha = self.log_alpha.exp().item()
    #         alpha_loss = alpha_loss.item()

    #     self._soft_update()

    #     return {
    #         "critic_loss": critic_loss.item(),
    #         "actor_loss":  actor_loss.item(),
    #         "alpha_loss":  alpha_loss,
    #         "alpha":       self.alpha,
    #         "q_mean":      q_vals.mean().item(),
    #     }
        
    # def _update_loop(self, stop_event: threading.Event, stat_queue: list, lock: threading.Lock):
    #     """
    #     Background thread: runs gradient updates continuously while the
    #     main thread steps the environment. This decouples learning from
    #     data collection so rtgym's real-time clock is never blocked.
    #     """
    #     while not stop_event.is_set():
    #         if len(self.buffer) < self.batch_size:
    #             time.sleep(0.01)   # wait for buffer to fill
    #             continue
 
    #         stats = None
    #         for _ in range(self.G):
    #             stats = self._update()
 
    #         if stats is not None:
    #             with lock:
    #                 stat_queue.append(stats)
    
    # def train(self, env : tmrl.GenericGymEnv, first_obs=None, total_steps: int = 500_000, log_interval: int = 1_000, current_run_folder: CurrentRunFolder = CurrentRunFolder("./run")):
    #     obs = env.reset()[0]
    #     ep_reward, ep_count = 0.0, 0
    #     recent_rewards : deque[float]  = deque(maxlen=20)
    #     stat_queue: list = []          # background thread pushes stats here
    #     lock            = threading.Lock()
    #     global_step     = 0
    #     start_time      = time.time()
 
    #     print(f"Warming up for {self.warmup_steps} steps with random actions...")
    #     print(f"Gradient updates run in background thread (G={self.G} per cycle).")
 
    #     # ── start background update thread after warmup ──────────────
    #     stop_event   = threading.Event()
    #     update_thread = None   # started once warmup is done
 
    #     for _ in range(self.warmup_steps):
    #         # img_t = obs_to_img_tensor(obs, self.device)
    #         # with torch.no_grad():
    #         #     latent = self.encoder(img_t)
    #         #     action, _ = self.actor.act(latent)
    #         # action_np = np.clip(action.squeeze(0).cpu().numpy() + env.action_space.sample(),-1,1)
            
    #         action_np = np.array([
    #                 np.random.uniform(-1.0, 1.0),                               # Gas
    #                 np.random.uniform(-1.0, 1.0),                             
    #                 np.random.uniform(-1.0, 1.0)       # Steer
    #             ])
    #         env_action = np.zeros_like(action_np)
            
    #         # Gas: [-1, 1] -> [0, 1]
    #         env_action[0] = (action_np[0] + 1.0) / 2.0  
            
    #         # Brake: [-1, 1] -> [-1, 0] (Assuming your action_dim > 1)
    #         if self.action_dim > 1:
    #             env_action[1] = (action_np[1] - 1.0) / 2.0  
                
    #         # Steer: [-1, 1] -> [-1, 1] (Assuming your action_dim > 2)
    #         if self.action_dim > 2:
    #             env_action[2] = action_np[2]
    #         next_obs, reward, terminated, truncated, info = env.step(action_np)
    #         reward = float(reward)
    #         done = terminated or truncated
    #         img_np      = obs_to_img_tensor(obs,      self.device).squeeze(0).cpu().numpy()
    #         next_img_np = obs_to_img_tensor(next_obs, self.device).squeeze(0).cpu().numpy()
    #         self.buffer.add(img_np, action_np, reward, next_img_np, done)
 
    #         obs        = next_obs
    #         ep_reward += reward
    #         global_step += 1
 
    #         if done:
    #             if("termination_reason" in info):
    #                 print(f"Terminated: {info["termination_reason"]}")
    #             print(f"Terminated, Score: {ep_reward}")
    #             recent_rewards.append(ep_reward)
    #             ep_count  += 1
    #             ep_reward  = 0.0
    #             obs, _     = env.reset()
            
 
    #     update_thread = threading.Thread(
    #         target=self._update_loop,
    #         args=(stop_event, stat_queue, lock),
    #         daemon=True,
    #     )
    #     update_thread.start()
    #     print("Background update thread started.")



    #     n_crashes = 0
    #     n_episodes = 0
    #     n_no_motion = 0
 
    #     while global_step < total_steps:
    #         img_t = obs_to_img_tensor(obs, self.device)
    #         with torch.no_grad():
    #             latent = self.encoder(img_t)
    #             action, _ = self.actor.act(latent)
    #         action_np = action.squeeze(0).cpu().numpy()
            
    #         env_action = np.zeros_like(action_np)
            
    #         # Gas: [-1, 1] -> [0, 1]
    #         env_action[0] = (action_np[0] + 1.0) / 2.0  
            
    #         # Brake: [-1, 1] -> [-1, 0] (Assuming your action_dim > 1)
    #         if self.action_dim > 1:
    #             env_action[1] = (action_np[1] - 1.0) / 2.0  
                
    #         # Steer: [-1, 1] -> [-1, 1] (Assuming your action_dim > 2)
    #         if self.action_dim > 2:
    #             env_action[2] = action_np[2]
 
    #         # ── env step (never blocked by gradient updates) ─────────
    #         next_obs, reward, terminated, truncated, info = env.step(env_action)
    #         reward = float(reward)
    #         done = terminated or truncated
 
    #         img_np      = obs_to_img_tensor(obs,      self.device).squeeze(0).cpu().numpy()
    #         next_img_np = obs_to_img_tensor(next_obs, self.device).squeeze(0).cpu().numpy()
    #         self.buffer.add(img_np, action_np, reward, next_img_np, done)
 
    #         obs        = next_obs
    #         ep_reward += reward
    #         global_step += 1
 
    #         if done:
    #             if("termination_reason" in info):
    #                 print(f"Terminated: {info["termination_reason"]}")
    #                 match(info["termination_reason"]):
    #                     case "no-movement":
    #                         n_no_motion += 1
    #                     case "crash":
    #                         n_crashes += 1
    #             n_episodes += 1
    #             print(f"Terminated, Score: {ep_reward}")
    #             recent_rewards.append(ep_reward)
    #             ep_count  += 1
    #             ep_reward  = 0.0
    #             obs, _     = env.reset()
 
    #         if global_step % log_interval == 0:
    #             with lock:
    #                 recent_stats = list(stat_queue)
    #                 stat_queue.clear()
 
    #             checkpoint_file = current_run_folder.get_date_file_name("pt", "chkpts")
    #             self.save(checkpoint_file)
                
                
    #             if recent_stats:
    #                 elapsed  = time.time() - start_time
    #                 mean_rew = np.mean(recent_rewards) if recent_rewards else float("nan")
                    
    #                 with open(current_run_folder.get_file_name("data.csv"), "a+") as f:
    #                     f.write(f"{global_step},{mean_rew},{ep_count},"
    #                             f"{np.mean([s['critic_loss'] for s in recent_stats])},"
    #                             f"{np.mean([s['actor_loss']  for s in recent_stats])},"
    #                             f"{np.mean([s['alpha']       for s in recent_stats])},"
    #                             f"{np.mean([s['q_mean']          for s in recent_stats])},"
    #                             f"{global_step / elapsed},\"{checkpoint_file}\","
    #                             f"{n_episodes},{n_crashes},{n_no_motion}"
    #                             f"\n")
    #                 print(
    #                     f"[{global_step:>7d}] "
    #                     f"rew={mean_rew:>8.2f}  ep={ep_count:>4d}  "
    #                     f"critic={np.mean([s['critic_loss'] for s in recent_stats]):>7.4f}  "
    #                     f"actor={np.mean([s['actor_loss']  for s in recent_stats]):>7.4f}  "
    #                     f"alpha={np.mean([s['alpha']       for s in recent_stats]):.4f}  "
    #                     f"Q={np.mean([s['q_mean']          for s in recent_stats]):>7.3f}  "
    #                     f"fps={global_step / elapsed:>5.0f}  "
    #                     f"n_eps={n_episodes}  n_crashes={n_crashes}  n_no_motion={n_no_motion}"
    #                 )
                    
    #                 n_episodes = 0
    #                 n_crashes = 0
    #                 n_no_motion = 0
 
    #     # ── shut down background thread cleanly ──────────────────────
    #     stop_event.set()
    #     if update_thread is not None:
    #         update_thread.join(timeout=5.0)
 
    #     env.close()
    #     print("Training complete.")

    # def save(self, path: str = "redq_tmrl.pt"):
    #     torch.save({
    #         "encoder":        self.encoder.state_dict(),
    #         "encoder_target": self.encoder_target.state_dict(),
    #         "actor":          self.actor.state_dict(),
    #         "critics":        self.critics.state_dict(),
    #         "critics_target": self.critics_target.state_dict(),
    #         "log_alpha":      self.log_alpha if self.auto_alpha else None,
    #     }, path)
    #     print(f"Saved → {path}")

    # def load(self, path: str = "redq_tmrl.pt"):
    #     ckpt = torch.load(path, map_location=self.device)
    #     self.encoder.load_state_dict(ckpt["encoder"])
    #     self.encoder_target.load_state_dict(ckpt["encoder_target"])
    #     self.actor.load_state_dict(ckpt["actor"])
    #     self.critics.load_state_dict(ckpt["critics"])
    #     self.critics_target.load_state_dict(ckpt["critics_target"])
    #     if self.auto_alpha and ckpt["log_alpha"] is not None:
    #         self.log_alpha.data = ckpt["log_alpha"].data
    #         self.alpha = self.log_alpha.exp().item()
    #     print(f"Loaded ← {path}")