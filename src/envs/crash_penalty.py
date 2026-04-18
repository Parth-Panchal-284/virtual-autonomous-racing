import numpy as np

# ─────────────────────────────────────────────
# Crash penalty wrapper
# ─────────────────────────────────────────────
 
class CrashPenaltyWrapper:
    """
    Wraps a tmrl environment and adds a speed-scaled penalty on crash.
 
    How it works
    ------------
    tmrl signals a crash via `terminated=True`.  At that moment we read
    the current speed from the observation and subtract:
 
        penalty = base_penalty + speed_coef * speed_kmh
 
    where speed_kmh is extracted from the first scalar in the obs tuple
    (tmrl's default layout puts speed first, in km/h).
 
    Parameters
    ----------
    env         : the raw tmrl environment
    base_penalty: flat penalty applied on any crash              (default 50)
    speed_coef  : extra penalty per km/h of speed at crash time  (default  1)
    max_penalty : ceiling so a single crash can't dominate       (default 200)
 
    Example penalty curve
    ---------------------
      0 km/h crash  →  -50
     50 km/h crash  → -100
    150 km/h crash  → -200  (capped)
    """
 
    def __init__(self, env, base_penalty: float = 50.0,
                 speed_coef: float = 1.0, max_penalty: float = 200.0, early_stop: bool = False):
        self.env          = env
        self.base_penalty = base_penalty
        self.speed_coef   = speed_coef
        self.max_penalty  = max_penalty
        # Forward standard gym attributes
        self.action_space      = env.action_space
        self.observation_space = env.observation_space
        
        self.last_speed = 0
        self.early_stop = early_stop
        self.frames_since_last_crash = 10000
 
    def _get_speed(self, obs) -> float:
        """Extract speed (km/h) from obs. Assumes speed is the first scalar."""
        if isinstance(obs, (tuple, list)):
            speed_arr = np.array(obs[0], dtype=np.float32).flatten()
            return float(speed_arr[0])
        return float(np.array(obs, dtype=np.float32).flatten()[0])
 
    def reset(self, **kwargs):
        self.frames_since_last_crash = 10000
        self.last_speed = 0
        return self.env.reset(**kwargs)
    
    def determine_crash(self, speed):
        difference = (speed - self.last_speed) * 15
        return difference < -25 and 3 < speed and 30 < self.frames_since_last_crash
        
 
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["crashed"] = False        
        speed     = self._get_speed(obs)
        if(self.determine_crash(speed)):
            self.frames_since_last_crash = 0
            penalty   = self.base_penalty + self.speed_coef * abs(speed)
            penalty   = min(penalty, self.max_penalty)
            print(f"\n\nCRASHED: {speed} mph, Penalty: {penalty}, Rewqard: {reward}")
            reward   -= penalty
            info["crash_penalty"] = penalty
            info["crash_speed"]   = speed
            info["crashed"] = True            
            if(self.early_stop):
                info["termination_reason"] = "crash"
                terminated = truncated = True
 
        self.last_speed = speed
        self.frames_since_last_crash += 1
        return obs, reward, terminated, truncated, info
 
    def close(self):
        return self.env.close()