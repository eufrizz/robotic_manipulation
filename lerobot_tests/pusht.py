import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
import torch
import numpy as np
device = torch.device("cuda")

policy = DiffusionPolicy.from_pretrained("ckpts/pusht_diffusion2025-01-09_22-37-03").to(device)

from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.utils.utils import init_hydra_config
cfg = init_hydra_config(lerobot.__path__[0] + "/configs/env/pusht.yaml")

import gymnasium as gym
import mediapy as media
import gym_pusht
env = gym.make_vec('gym_pusht/PushT-v0', **dict(cfg.env.get("gym", {})))
# obs, info = env.reset()

import time
import torch.profiler as profiler

frames = []
obs, info = env.reset()
prep_obs = preprocess_observation(obs)
prep_obs = {key: prep_obs[key].to(device, non_blocking=True) for key in prep_obs}
frames.append(obs['pixels'][0])

terminated =  truncated = False
step = 0
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    # profile_memory=True,
) as prof:
    while step < 32 and (not truncated or terminated):
        prof.step()
        print(f"Step {step}")
        t0 = time.time()
        prep_obs = preprocess_observation(obs)
        prep_obs = {key: prep_obs[key].to(device, non_blocking=True) for key in prep_obs}
        t1 = time.time()
        print(f"Preprocessing: {t1-t0}s")
        with torch.inference_mode():
            action = policy.select_action(prep_obs )
        t2 = time.time()
        print(f"Inference: {t2-t1}s")

        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        t3 = time.time()
        print(f"Env step: {t3-t2}s")
        frames.append(obs['pixels'][0])
        step += 1
        prof.step()

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
prof.export_chrome_trace("log/pusht_trace.json")