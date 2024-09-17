# %%
import gymnasium as gym
import numpy as np
import mediapy as media
import torch
import torchvision
# torch.multiprocessing.set_start_method('spawn')
import gym_lite6.env, gym_lite6.pickup_task, gym_lite6.policies.mlp
# %env MUJOCO_GL=egl # Had to export this before starting jupyter server
# import mujoco
import time



# %%
if __name__ == "__main__":

  from datasets import load_from_disk
  from torch.utils.data import DataLoader
  from tqdm import tqdm
  from torch.utils.tensorboard import SummaryWriter
  import datetime
  from pathlib import Path
  import argparse


  # %%

  parser = argparse.ArgumentParser(
                    prog='Train Lite6 BC-MLP-MSE',
                    description='Train BC-MLP-MSE on Ufactory Lite6')
  parser.add_argument('--checkpoint')
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--n_epochs', default=20, type=int)

  args = parser.parse_args()

  # %%
  task = gym_lite6.pickup_task.GraspAndLiftTask('gripper_left_finger', 'gripper_right_finger', 'box', 'floor')
  env = gym.make(
      "UfactoryCubePickup-v0",
      task=task,
      obs_type="pixels_state",
      max_episode_steps=350,
      visualization_width=320,
      visualization_height=240,
  )
  observation, info = env.reset()
  # media.show_image(env.render(), width=400, height=400)


  # %%
  from lerobot.common.datasets.utils import hf_transform_to_torch
  dataset_path = "datasets/50_single_2024-09-16_15-07-50.hf"
  dataset = load_from_disk(dataset_path)
  if "from" not in dataset.column_names:
    first_frames=dataset.filter(lambda example: example['frame_index'] == 0)
    from_idxs = torch.tensor(first_frames['index'])
    to_idxs = torch.tensor(first_frames['index'][1:] + [len(dataset)])
    episode_data_index={"from": from_idxs, "to": to_idxs}
      
  dataset.set_transform(hf_transform_to_torch)
  # dataset.set_transform(lambda x: interface.lerobot_preprocess(hf_transform_to_torch(x)))
  # dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)


  # %%
  from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
  lerobot_dataset = LeRobotDataset.from_preloaded(root=Path(dataset_path),
          split="train",
          delta_timestamps={"action.qpos": [0, 0.1], "action.gripper": [0, 0.1]},
          # additional preloaded attributes
          hf_dataset=dataset,
          episode_data_index=episode_data_index,
          info = {
            "codebase_version": CODEBASE_VERSION,
            "fps": env.metadata["render_fps"]
          })


  # %%
  
  
  params = {}
  if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    start_epoch = checkpoint["epoch"] + 1
    step = checkpoint["step"]
    params = checkpoint["params"]
    if not "hidden_layer_dims" in params:
      params["hidden_layer_dims"] = [64, 64, 64]
    if not "dropout" in params:
      params["dropout"] = False
  else:
    print("train from scratch")
    start_epoch = 0
    step = 0
    params = {}
    params["normalize_qpos"] = True
    params["dropout"] = False
    params["hidden_layer_dims"] = [64, 64, 64]

  # Override these params
  params["lr"] = 1e-3
  params["device"] = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

  policy = gym_lite6.policies.mlp.MLPPolicy(params["hidden_layer_dims"], dropout=params["dropout"]).to(params["device"])
  loss=torch.tensor(0)
  optimizer = torch.optim.Adam(policy.parameters(), lr=params["lr"])

  if args.checkpoint is not None:
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=params["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss = checkpoint["loss"]
    print(f"Loaded checkpoint at epoch {start_epoch}")
  
  loss_fn = torch.nn.MSELoss()
  dataloader = DataLoader(
          lerobot_dataset,
          num_workers=4,
          batch_size=128,
          shuffle=True,
          # sampler=sampler,
          pin_memory=params["device"].type != "cpu",
          drop_last=False,
      )

  # TODO: Maybe check that these are the same as what is loaded from checkpoint?
  jnt_range_low = env.unwrapped.model.jnt_range[:6, 0]
  jnt_range_high = env.unwrapped.model.jnt_range[:6, 1]
  bounds_centre = torch.tensor((jnt_range_low + jnt_range_high) / 2, dtype=torch.float32)
  bounds_range = torch.tensor(jnt_range_high - jnt_range_low, dtype=torch.float32)
  params["joint_bounds"] = {"centre": bounds_centre, "range": bounds_range}

  interface = gym_lite6.policies.mlp.Interface(params)
  

  curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  hidden_layer_dims = '_'.join([str(x.out_features) for x in policy.actor[:-1] if 'out_features' in x.__dict__])
  OUTPUT_FOLDER=f'ckpts/lite6_grasp_h{hidden_layer_dims}_{curr_time}'
  Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

  if args.eval:
    print("Evaluating...")
    policy.eval()
    print(f"Epoch: {start_epoch}, steps: {step}, loss: {loss.item()}")
    avg_reward, frames = interface.evaluate_policy(env, policy, 5)
    media.write_video(OUTPUT_FOLDER + f"/epoch_{start_epoch}.mp4", frames, fps=env.metadata["render_fps"])
  
  else:
    writer = SummaryWriter(log_dir=f"runs/lite6_grasp/{curr_time}")

    end_epoch = start_epoch+args.n_epochs
    for epoch in range(start_epoch, end_epoch+1):
      policy.train()
      end = time.time()
      for batch in tqdm(dataloader):
        data_load_time = time.time()

        batch = interface.batched_preprocess(batch)

        # Send data tensors from CPU to GPU
        state = batch["preprocessed.observation.state.qpos"].to(params["device"], non_blocking=True)
        image_side = batch["observation.pixels.side"].to(params["device"], non_blocking=True)
        image_gripper = batch["observation.pixels.gripper"].to(params["device"], non_blocking=True)

        # Because we sample the action ahead in time [0, 0.1], it has an extra dimension, and we select the last dim
        a_hat = batch["preprocessed.action.state.qpos"][:, -1, :].to(params["device"], non_blocking=True)
        # print([(x, batch[x]) for x in batch if "pixels" not in x])

        gpu_load_time = time.time()

        a_pred = policy.predict(state, image_side, image_gripper)

        pred_time = time.time()

        loss = loss_fn(a_pred, a_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_time = time.time()

        writer.add_scalar("Loss/train", loss.item(), step)
        writer.add_scalar("Time/data_load", data_load_time - end, step)
        writer.add_scalar("Time/gpu_transfer", gpu_load_time - data_load_time, step)
        writer.add_scalar("Time/pred_time", pred_time - gpu_load_time, step)
        writer.add_scalar("Time/train_time", train_time - pred_time, step)
        writer.add_scalar("Time/step_time", time.time() - end, step)

        step += 1
        end = time.time()
      
      if epoch in [1, 2, 4, 8, 16, 32, 64]:
        # Evaluate
        policy.eval()
        print(f"Epoch: {epoch}/{end_epoch}, steps: {step}, loss: {loss.item()}")
        qpos0 = np.array([0, 0.541, 1.49 , 2.961, 0.596, 0.203])
        box_pos0 = np.array([0.2, 0, 0.0])
        box_quat0 = None
        avg_reward, frames = interface.evaluate_policy(env, policy, 5, qpos0, box_pos0, box_quat0)
        media.write_video(OUTPUT_FOLDER + f"/epoch_{epoch}.mp4", frames, fps=env.metadata["render_fps"])
        print("avg reward: ", avg_reward)
        writer.add_scalar("Reward/val", avg_reward, step)
        # _, frames = evaluate_policy(policy, env, 1, visualise=True)
        writer.add_images("Image", np.stack([frames[x].transpose(2, 0, 1) for x in range(0, len(frames), 50)], axis=0), step)
      
        writer.add_scalar("Time/eval_time", time.time() - end, step)


      if epoch % 10 == 0 or epoch == end_epoch:
      # if epoch in [1, 2, 4, 8, 10, 16, 20, 32, 64]:
        torch.save({
                'epoch': epoch,
                'step': step,
                'params': params,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, OUTPUT_FOLDER + f'/epoch_{epoch}.pt')
      
    writer.flush()

    writer.close()

