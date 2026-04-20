from datetime import datetime
import os
import pathlib

import tmrl

from envs.crash_penalty import CrashPenaltyWrapper
from envs.no_movement_penalty import NoMovementPenalty
from model.sac.trainer import SACTrainer
from util import CurrentRunFolder


if __name__ == "__main__":
    from tmrl.config.config_objects import CONFIG_DICT

    print(CONFIG_DICT)

    # Use a separate folder for this architecture/config
    folder = "runs/SAC_enc256"
    if folder is not None:
        current_run_folder = CurrentRunFolder(str(pathlib.Path(folder)))
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        current_run_folder = CurrentRunFolder(str(pathlib.Path("runs", timestamp)))

    env = tmrl.get_environment()
    env = CrashPenaltyWrapper(
        env,
        base_penalty=-1.0,
        speed_coef=-0.005,
        early_stop=True,
    )
    env = NoMovementPenalty(env, 2, 40)

    trainer, first_obs = SACTrainer.from_env(
        env,
        enc_dim=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=5e-5,
        critic_lr=5e-5,
        alpha_lr=1e-4,
        init_alpha=0.2,
        auto_alpha=True,
        batch_size=512,
        buffer_size=500_000,
        warmup_steps=10_000,
        updates_per_step=1,
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
    trainer.save(current_run_folder.get_file_name("sac_final.pt", "chkpts"))