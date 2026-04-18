from model.ppo.trainer import PPOTrainer
from util import CurrentRunFolder
import os
import tmrl
import pathlib
from tmrl.config.config_objects import CONFIG_DICT

if __name__ == "__main__":
    print(CONFIG_DICT)
    
    s = "ppo"
    current_run_folder = CurrentRunFolder(str(pathlib.Path("runs", s)))
    
    env = tmrl.get_environment()

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