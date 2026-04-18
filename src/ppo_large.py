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

from datetime import datetime
from util import CurrentRunFolder
from model.ppo_large.trainer import PPOTrainer
import tmrl


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
    #     env,
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
    
    print(trainer.policy.encoder)
    print(trainer.policy.critic)

    chkpts = sorted(os.listdir(current_run_folder.get_folder("chkpts")))
    if(0 < len(chkpts)):
        print(f"Loading old checkpoints \"{str(pathlib.Path(current_run_folder.get_file_name(chkpts[-1], "chkpts")))}\"")
        trainer.load(current_run_folder.get_file_name(chkpts[-1], "chkpts"))
    
    trainer.train(env, first_obs=first_obs, total_steps=2_000_000, current_run_folder=current_run_folder)
    trainer.save(current_run_folder.get_file_name("ppo_final.pt", "chkpts"))
