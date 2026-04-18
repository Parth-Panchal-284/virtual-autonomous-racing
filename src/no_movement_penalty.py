import numpy as np

# ─────────────────────────────────────────────
# Crash penalty wrapper
# ─────────────────────────────────────────────
 
class NoMovementPenalty:
 
    def __init__(self, env, min_speed: float = 5.0, max_frames: int = 20):
        self.env          = env
        # Forward standard gym attributes
        self.action_space      = env.action_space
        self.observation_space = env.observation_space
        
        self.min_speed = min_speed
        self.max_frames = max_frames
        
        self.frames_since_above_min_speed = 0
 
    def _get_speed(self, obs) -> float:
        """Extract speed (km/h) from obs. Assumes speed is the first scalar."""
        if isinstance(obs, (tuple, list)):
            speed_arr = np.array(obs[0], dtype=np.float32).flatten()
            return float(speed_arr[0])
        return float(np.array(obs, dtype=np.float32).flatten()[0])
 
    def reset(self, **kwargs):
        self.frames_since_above_min_speed = 0
        return self.env.reset(**kwargs)
    
 
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)


        speed     = self._get_speed(obs)
        print(speed)
        if(self.min_speed < speed):
            self.frames_since_above_min_speed = 0
        if(self.max_frames < self.frames_since_above_min_speed):
            # print("NO MOVEMENT")
            info["termination_reason"] = "no-movement"
            terminated = truncated = True
        
        self.frames_since_above_min_speed += 1
        return obs, reward, terminated, truncated, info
 
    def close(self):
        return self.env.close()