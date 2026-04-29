import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import tmrl

from envs.crash_penalty import CrashPenaltyWrapper
from envs.no_movement_penalty import NoMovementPenaltyWrapper
from envs.survival_bonus import SurvivalBonusWrapper
from model.droq.trainer import DroQTrainer
from util import CurrentRunFolder


if __name__ == "__main__":
    env = tmrl.get_environment()
    env = CrashPenaltyWrapper(env, penalty=-5.0)
    env = NoMovementPenaltyWrapper(env, speed_index=0, speed_threshold=0.03, duration=100, penalty=-1.0)
    env = SurvivalBonusWrapper(env, bonus_per_step=0.02)

    current_run_folder = CurrentRunFolder("runs/DROQ_enc256")

    trainer, first_obs = DroQTrainer.from_env(
        env,
        enc_dim=256,
        gamma=0.99,
        tau=0.002,
        actor_lr=3e-5,
        critic_lr=3e-5,
        alpha_lr=5e-5,
        init_alpha=0.15,
        auto_alpha=True,
        batch_size=512,
        buffer_size=500_000,
        warmup_steps=15_000,
        updates_per_step=20,
        dropout_p=0.01,
        layer_norm=True,
        max_grad_norm_actor=1.0,
        max_grad_norm_critic=0.5,
    )

    chkpt_dir = current_run_folder.get_folder("chkpts")
    chkpts = sorted(os.listdir(chkpt_dir))

    if len(chkpts) > 0:
        latest = current_run_folder.get_file_name(chkpts[-1], "chkpts")
        print(f'Trying to load checkpoint "{latest}"')
        try:
            trainer.load(latest)
            print("Checkpoint loaded successfully.")
        except RuntimeError as e:
            print("Checkpoint is incompatible with the current model architecture.")
            print("Starting a fresh run instead.")
            print(f"Load error: {e}")

    trainer.train(
        env,
        first_obs=first_obs,
        total_steps=500_000,
        log_interval=1_000,
        current_run_folder=current_run_folder,
    )
    trainer.save(current_run_folder.get_file_name("droq_final.pt", "chkpts"))