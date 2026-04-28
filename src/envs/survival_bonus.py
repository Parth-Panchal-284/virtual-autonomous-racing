class SurvivalBonusWrapper:
    """
    Adds a fixed positive reward bonus each environment step.

    This encourages policies that stay alive longer by making every
    non-terminated step slightly more valuable.
    """

    def __init__(self, env, bonus_per_step: float = 0.1):
        self.env = env
        self.bonus_per_step = bonus_per_step

        # Forward standard gym attributes
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward) + self.bonus_per_step
        info["survival_bonus"] = self.bonus_per_step
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()
