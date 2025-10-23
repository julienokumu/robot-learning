import mujoco
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import time

class UR5eEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path("mujoco_menagerie/universal_robots_ur5e/ur5e.xml")
        self.data = mujoco.MjData(self.model)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.target = np.array([0.5, 0, 0.5])
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}
    
    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = -np.linalg.norm(self._get_end_effector_pos() - self.target)
        done = False
        truncated = False

        return obs, reward, done, truncated, {}
    
    def _get_obs(self):
        return np.concatenate(self.data.qpos[:6], self.data.qvel[:6])
    
    def _get_end_effector_pos(self):
        return self.data.body("wrist_3_link").xpos
    
env = UR5eEnv()
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="/content/rl_tensorboard/",
    learning_rate=0.0001,
    batch_size=128,
    gamma=0.95
)
model.learn(total_timesteps=100_000)
model.save("UR5ePPO_policy_tuned.zip")