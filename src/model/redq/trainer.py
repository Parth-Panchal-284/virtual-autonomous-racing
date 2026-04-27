import numpy as np
import torch
import torch.optim as optim

from model.redq.actor import SquashedGaussianActor, REDQCritic
from model.redq.model import build_encoder

class REDQTrainer:
    def __init__(
        self,
        img_shape: tuple[int, int, int],
        vec_dim: int,
        action_dim: int,
        enc_dim: int = 512,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        alpha_lr: float = 1e-3,
        init_alpha: float = 0.2,
        auto_alpha: bool = True,
        target_entropy: float | None = None,
        polyak: float = 0.995,
        max_grad_norm_actor: float = 0.5,
        max_grad_norm_critic: float = 0.5,
        N: int = 10,
        M: int = 2,
        device: torch.device | str = "auto"
    ):
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm_actor = max_grad_norm_actor
        self.max_grad_norm_critic = max_grad_norm_critic
        self.action_dim = action_dim

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        actor_encoder = build_encoder(
            img_shape=img_shape,
            vec_dim=vec_dim,
            enc_dim=enc_dim,
        )
        def _build_encoder():
            return build_encoder(
                img_shape=img_shape,
                vec_dim=vec_dim,
                enc_dim=enc_dim,
            )
            
        self.N = N
        self.M = M
        self.polyak = polyak

        self.actor = SquashedGaussianActor(actor_encoder, enc_dim, action_dim).to(self.device)
        
        self.critic = REDQCritic(_build_encoder, enc_dim, action_dim, N).to(self.device)
        self.critic_target = REDQCritic(_build_encoder, enc_dim, action_dim, N).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimzer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizers = [ optim.Adam(model.parameters(), lr=critic_lr) for model in self.critic.qs]
        self.critic_criterion = torch.nn.MSELoss()
        self.loss_pi = torch.zeros((1,), device=self.device)
    
        self.auto_alpha = auto_alpha
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().item()
            
        self.q_updates_per_policy_update = 10

        param_count = sum(p.numel() for p in list(self.actor.parameters()) + list(self.critic.parameters()))
        print(f"SAC  device={self.device}  params={param_count:,}  alpha={self.alpha:.4f}")
        print(
            f"     obs_img={img_shape} vec_dim={vec_dim} action_dim={action_dim} "
            # f"batch={batch_size} buffer={buffer_size}"
        )
        print(self.critic)
        print(self.actor)
        
        self.i_update = 0
        self.p_update = 0

    def _train(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        
        self.i_update += 1
        update_policy = (self.i_update % self.q_updates_per_policy_update == 0)
        
        if update_policy:
            # print("Updating Policy")
            self.p_update += 1
            pi, logp_pi = self.actor.sample(obs)
        loss_alpha = None
        if(self.auto_alpha and update_policy):
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha = alpha_t
        else:
            alpha_t = torch.tensor(float(self.alpha)).to(self.device)

        if loss_alpha is not None:
            self.alpha_opt.zero_grad()
            loss_alpha.backward()
            self.alpha_opt.step()
            
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_obs)
            
            sample_idxs = np.random.choice(self.N, self.M, replace=False)
            
            q_prediction_next_list = [self.critic_target.qs[i].forward(next_obs, next_actions) for i in sample_idxs]
            q_prediction_next_cat = torch.cat(q_prediction_next_list, dim=-1)
            min_q, _ = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            backup = rewards + self.gamma * (1.0 - dones) * (min_q - alpha_t * next_log_prob)

        q_prediction_list = self.critic.forward(obs, actions)
        q_prediction_cat = torch.cat(q_prediction_list, -1)
        backup = backup.expand(-1, self.N)
        
        loss_q = self.critic_criterion(q_prediction_cat, backup)
        
        for q in self.critic_optimizers:
            q.zero_grad()
        loss_q.backward()
        
        if update_policy:
            for q in self.critic.qs:
                q.requires_grad_(False)

            qs_pi = [ q.forward(obs, pi) for q in self.critic.qs] # type: ignore
            qs_pi_cat = torch.cat(qs_pi, -1)
            ave_q = torch.mean(qs_pi_cat, dim=1, keepdim=True)
            loss_pi = (alpha_t * logp_pi - ave_q).mean() # type: ignore
            self.actor_optimzer.zero_grad()
            loss_pi.backward()
            
            for q in self.critic.qs:
                q.requires_grad_(True)
            
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm_critic)
        for q_optimizer in self.critic_optimizers:
            q_optimizer.step()
            
        if update_policy:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm_actor)
            self.actor_optimzer.step()
        
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        if update_policy:
            self.loss_pi = loss_pi.detach()
            
        ret_dict = dict(
            loss_actor=self.loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
            mean_q=q_prediction_cat.mean().item(),
        )
        
        if self.auto_alpha:
            if(loss_alpha is not None):
                ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            else:
                ret_dict["loss_entropy_coef"] = -1
            ret_dict["entropy_coef"] = alpha_t.item()
        
        return ret_dict

    def act(self, obs):
        with torch.no_grad():
            obs = obs[0].unsqueeze(0), obs[1].unsqueeze(0)
            action, _ = self.actor.sample(obs)
            return action

    def save(self, path: str = "sac_tmrl.pt"):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_optimzer.state_dict(),
                "critic_opt": [optimizer.state_dict() for optimizer in self.critic_optimizers],
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
        self.actor_optimzer.load_state_dict(ckpt["actor_opt"])
        for optimizer, state_dict in zip(self.critic_optimizers, ckpt["critic_opt"]):
            optimizer.load_state_dict(state_dict)
            
        if self.auto_alpha and self.log_alpha is not None and ckpt.get("log_alpha") is not None:
            self.log_alpha.data = ckpt["log_alpha"].to(self.device)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = float(ckpt.get("alpha", self.alpha))

        print(f"Loaded <- {path}")
        
        
        