import gym_lite6.pickup_task
import imageio
import gymnasium as gym
import numpy as np
import mediapy as media
import torch
import gym_lite6.scripted_policy, gym_lite6.env
import mujoco
import cv2

import h5py
from pathlib import Path
import os

import tracemalloc
import pdb

tracemalloc.start()

def create_h5py_dataset(data_dict, group):
  for key, val in data_dict.items():
    if isinstance(val, dict):
      grp = group.create_group(key)
      create_h5py_dataset(data_dict[key], grp)
    elif isinstance(val, list):
      group.create_dataset(key, data=val)

def record_episodes(env, policy, dataset_dir, n=1, len=300):

  if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)

  successful_trajectories = 0

  while successful_trajectories < n:
    snapshot = tracemalloc.take_snapshot()
    for index, stat in enumerate(snapshot.statistics('lineno')[:5], 1):
      frame = stat.traceback[0]
      filename = os.sep.join(frame.filename.split(os.sep)[-2:])
      print(f"#{index} {filename}:{frame.lineno}: {stat.size/(1024**2):.1f}MiB")
    # pdb.set_trace()
    # print(snapshot)
    episode_idx = successful_trajectories
    print(f"Episode {episode_idx}")
    observation, info = env.reset()
    policy.reset()

    data = {"action": {"qpos": [], "gripper": []}, "observation": {"state": {"qpos": [], "gripper": []}, "pixels": []}, "reward": []}
    for step in range(len):
      action = policy(env.unwrapped.model, env.unwrapped.data, observation, info)
      observation, reward, terminated, truncated, info = env.step(action)

      data["action"]["qpos"].append(action["qpos"])
      data["action"]["gripper"].append(action["gripper"])
      data["observation"]["state"]["qpos"].append(observation["state"]["qpos"])
      data["observation"]["state"]["gripper"].append(observation["state"]["gripper"])
      data["observation"]["pixels"].append(observation["pixels"])
      data["reward"].append(reward)
    
    if policy.stage == 4:
      path = dataset_dir + f"/ep_{episode_idx}"
      with h5py.File(path + ".hdf5", "w") as f:
        create_h5py_dataset(data, f)
      # out = cv2.VideoWriter(path + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), env.metadata["render_fps"], (w, h))
      media.write_video(path + ".mp4", data["observation"]["pixels"], fps=env.metadata["render_fps"])
      successful_trajectories += 1
      print(f"Success, saved {path}")
    else:
      print("Failed, retrying")

    # media.show_video(data["observation"]["pixels"])


task = gym_lite6.pickup_task.GraspAndLiftTask('gripper_left_finger', 'gripper_right_finger', 'box', 'floor')

env = gym.make(
    "UfactoryCubePickup-v0",
    task=task,
    obs_type="pixels_state",
    max_episode_steps=400,
)

policy = gym_lite6.scripted_policy.ScriptedPickupPolicy(env, 'end_effector', 'box', 'gripper_left_finger', 'gripper_right_finger', max_vel=0.2)
record_episodes(env, policy, "dataset/pickup_script", n=50)
