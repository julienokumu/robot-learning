import mujoco
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from mujoco import viewer
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
        return np.concatenate([self.data.qpos[:6], self.data.qvel[:6]])
    
    def _get_end_effector_pos(self):
        return self.data.body("wrist_3_link").xpos
    
env = UR5eEnv(render_mode="human")
model = PPO.load("UR5e PPO/UR5ePPO_policy.zip")
print("policy loaded successfuly")
obs, _ = env.reset()
with viewer.launch_passive(env.model, env.data) as v:
    for _ in range(100000):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        v.sync()
        time.sleep(0.01)
        if done or truncated:
            obs, _ = env.reset()