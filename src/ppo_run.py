"""
Proximal Policy Optimization (PPO) for TMRL (TrackMania Reinforcement Learning)

Implements clipped PPO with:
  - Actor-Critic network with shared CNN encoder (for image obs) or MLP
  - Generalized Advantage Estimation (GAE)
  - Multiple epochs of minibatch updates per rollout
  - Entropy bonus for exploration
  - Value function clipping

Compatible with tmrl's custom gym-like interface.
"""
import os
from datetime import datetime
from util import CurrentRunFolder
from model.ppo.trainer import PPORunner

if __name__ == "__main__":
    import tmrl
    import pathlib
    from tmrl.config.config_objects import CONFIG_DICT
    print(CONFIG_DICT)
    
    
    folder = "runs/ppo"
    current_run_folder = None
    if(folder is not None):
        current_run_folder = CurrentRunFolder(str(pathlib.Path(folder)))
    else:
        s = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        current_run_folder = CurrentRunFolder(str(pathlib.Path("runs", s)))
    
    env = tmrl.get_environment()

    runner, first_obs = PPORunner.from_env(env)
    

    chkpts = sorted(os.listdir(current_run_folder.get_folder("chkpts")))
    if(0 < len(chkpts)):
        print(f"Loading old checkpoints \"{str(pathlib.Path(current_run_folder.get_file_name(chkpts[-1], "chkpts")))}\"")
        runner.load(current_run_folder.get_file_name(chkpts[-1], "chkpts"))
        
    print(runner.policy.encoder)
    print(runner.policy.critic)
    
    runner.run(env)
    