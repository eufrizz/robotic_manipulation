import gymnasium as gym
from gymnasium.envs.registration import register

register(id='UfactoryCubePickup-v0', entry_point='gym_lite6.env:UfactoryLite6Env')

env = gym.make(
    "UfactoryCubePickup-v0",
    obs_type="pixels_pose",
    max_episode_steps=300,
)
