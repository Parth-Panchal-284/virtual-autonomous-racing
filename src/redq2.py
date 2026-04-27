
import torch
import numpy as np
from model.replay import ReplayBuffer
from model.redq.trainer import REDQTrainer
from util import CurrentRunFolder
import pathlib

import threading
import time

from tqdm import tqdm

G = 10
BATCH_SIZE = 256
EPOCHS = 10000
ENVIRONMENT_RUN_LOG_INTERVAL = 1000
TRAIN_LOG_INTERVAL = 1000
# WARMUP_STEPS = 256
WARMUP_STEPS = 10000
REPLAY_BUFFER_SIZE = 100000

def observation_to_tensors(observation) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.as_tensor(observation[3]), torch.concat(
        [
            torch.as_tensor(observation[0]),
            torch.as_tensor(observation[1]),
            torch.as_tensor(observation[2]),
            # torch.as_tensor(observation[4]),
            # torch.as_tensor(observation[5]),
        ]
    )

def train_loop(stop_event: threading.Event, agent: REDQTrainer, replay_buffer: ReplayBuffer, lock: threading.Lock, current_run_folder: CurrentRunFolder):
    """
    Background thread: runs gradient updates continuously while the
    main thread steps the environment. This decouples learning from
    data collection so rtgym's real-time clock is never blocked.
    """
                    
    i = 0
    epoch = 0
    run_info = {
        "epoch": 0,
        "loss_actor": 0,
        "loss_critic": 0,
        "mean_q": 0,
        "loss_entropy_coef": 0,
        "entropy_coef": 0,
        "n_model_updates": 0,
        "n_policy_updates": 0,
    }
    last_i_update = 0
    last_p_update = 0
    while not stop_event.is_set():
        if len(replay_buffer) < BATCH_SIZE:
            time.sleep(0.01)   # wait for buffer to fill
            continue

        # stats = None
        for _ in range(G):
            sample = replay_buffer.sample(BATCH_SIZE)
            stats = agent._train(sample)
            
            run_info["loss_actor"] += stats["loss_actor"]
            run_info["loss_critic"] += stats["loss_critic"]
            run_info["mean_q"] += stats["mean_q"]
            run_info["entropy_coef"] += stats["entropy_coef"]
            
            if(i % TRAIN_LOG_INTERVAL == 0):
                stats["step"] = i
                run_info["epoch"] += 1
                run_info["n_model_updates"] = agent.i_update - last_i_update
                last_i_update = agent.i_update
                run_info["n_policy_updates"] = agent.p_update - last_p_update
                last_p_update = agent.p_update
                
                
                run_info["loss_actor"] /= run_info["n_model_updates"]
                run_info["loss_critic"] /= run_info["n_model_updates"]
                run_info["mean_q"] /= run_info["n_model_updates"]
                run_info["entropy_coef"] /= run_info["n_model_updates"]
                
                csv = current_run_folder.get_file_name("training-data.csv")
                exists = pathlib.Path(csv).exists()
                with open(csv, "a+") as f:
                    if(not exists):
                        f.write(",".join(stats.keys()) + "\n")
                    f.write(",".join([str(v) for v in stats.values()]) + "\n")
                print(" ".join([ f"{k}={v}" for k,v in stats.items()]))
            i += 1

def run(agent : REDQTrainer, env, replay_buffer : ReplayBuffer, current_run_folder: CurrentRunFolder):

    print("Warming up buffer.")
    last_observation : tuple[torch.Tensor, torch.Tensor] = None # type: ignore
    for i in tqdm(range(WARMUP_STEPS)):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        observation = observation_to_tensors(observation)
        if(last_observation is not None):
            replay_buffer.add(
                np.asarray(last_observation[0]),
                np.asarray(last_observation[1]),
                action,
                reward, 
                np.asarray(observation[0]),
                np.asarray(observation[1]),
                done
            )
        last_observation = observation
        if(done):
            env.reset()
    
    print("Finished warming up buffer.")
    lock            = threading.Lock()
    stop_event   = threading.Event()

    update_thread = threading.Thread(
        target=train_loop,
        args=(stop_event, agent, replay_buffer, lock, current_run_folder),
        daemon=True,
    )
    update_thread.start()
    print("Background update thread started.")
    
    run_info = {
        "epoch": 0,
        "episodes": 0,
        "total_episodes": 0,
        "mean_rewards_per_episode": 0.0,
        "max_rewards": 0.0,
        "step": 0,
        "n_model_updates": 0
    }
    
    last_n_model_updates = 0
    current_reward = 0.0
    total_rewards = 0.0
    
    
    for epoch in range(EPOCHS):
        for i in tqdm(range(ENVIRONMENT_RUN_LOG_INTERVAL)):
            last_observation = last_observation[0].to(agent.device), last_observation[1].to(agent.device)
            action = agent.act(last_observation)
            action = action.squeeze(0).cpu().detach()
            action = torch.clamp(action, min=-1, max=1).numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation = observation_to_tensors(observation)
            if(last_observation is not None):
                replay_buffer.add(
                    last_observation[0].cpu().numpy(),
                    last_observation[1].cpu().numpy(),
                    action,
                    reward, 
                    np.asarray(observation[0]),
                    np.asarray(observation[1]),
                    done
                )
            last_observation = observation
            current_reward += reward
            if(done):
                run_info["episodes"] += 1
                total_rewards += current_reward
                run_info["max_rewards"] = max(run_info["max_rewards"], current_reward)
                current_reward = 0.0
                env.reset()
            
        run_info["epoch"] = epoch + 1
        run_info["total_episodes"] += run_info["episodes"]
        run_info["mean_rewards_per_episode"] = total_rewards / run_info["episodes"] if run_info["episodes"] > 0 else 0
        run_info["n_model_updates"] = agent.i_update - last_n_model_updates
        last_n_model_updates = agent.i_update
        chkpt = current_run_folder.get_date_file_name("pt", "chkpts")
        csv = current_run_folder.get_file_name("run.csv")
        exists = pathlib.Path(csv).exists()
        with open(csv, "a+") as f:
            if(not exists):
                f.write(",".join(run_info.keys()) + "\n")
            f.write(",".join([str(v) for v in run_info.values()]) + f",\"{chkpt}\"" + "\n")
        print(" ".join([ f"{k}={v}" for k,v in run_info.items()]))
        agent.save(chkpt)
        run_info["max_rewards"] = 0.0
        run_info["episodes"] = 0
        total_rewards = 0
    


if __name__ == "__main__":
    try:
        import warnings
        warnings.simplefilter("ignore")
        import tmrl
        env = tmrl.get_environment()
        
        env.reset()
        x = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(x)
        print([ o.shape for o in observation])
        observation = observation_to_tensors(observation)
        
        img_shape = observation[0].shape
        vec_dim = observation[1].shape[0]
        action_dim = env.action_space.shape[0]# type: ignore
        agent = REDQTrainer(
            img_shape=img_shape, # type: ignore
            vec_dim=vec_dim,
            action_dim=action_dim,
            
        )
        
        replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, action_dim, torch.device("cuda"), img_shape, vec_dim) # type: ignore
        current_run_folder = CurrentRunFolder("runs/redq5")
        
        import os
        
        chkpts = sorted(os.listdir(current_run_folder.get_folder("chkpts")))
        if(0 < len(chkpts)):
            print(f"Loading old checkpoints \"{str(pathlib.Path(current_run_folder.get_file_name(chkpts[-1], "chkpts")))}\"")
            agent.load(current_run_folder.get_file_name(chkpts[-1], "chkpts"))
        
        run(agent, env, replay_buffer, current_run_folder)

        env.close()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure ppo_tmrl.py is in the same directory and tmrl is installed.")