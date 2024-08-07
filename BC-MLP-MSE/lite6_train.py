# %%
import gymnasium as gym
import numpy as np
import mediapy as media
import torch
import torchvision
# torch.multiprocessing.set_start_method('spawn')
import gym_lite6.env, gym_lite6.pickup_task
# %env MUJOCO_GL=egl # Had to export this before starting jupyter server
# import mujoco
import time


# %%

class MLPPolicy(torch.nn.Module):
  def __init__(self, hidden_layer_dims, state_dims=9):
    """
    state_dims: 6 for arm, 3 for gripper
    """
    super().__init__()

    # self.img_feature_extractor = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet18', )
    
    self.img_feature_extractor_side = self._create_img_feature_extractor()
    self.img_feature_extractor_gripper = self._create_img_feature_extractor()
    # Resnet output is 1x512, 2 bits for gripper
    self.actor = self._create_actor(512*2 + state_dims, hidden_layer_dims, state_dims)

    self.sigmoid = torch.nn.Sigmoid()
  
  def _create_actor(self, input_size, hidden_layer_dims, output_size):
    actor = []
    actor.append(torch.nn.Linear(input_size, hidden_layer_dims[0]))
    actor.append(torch.nn.ReLU())
    for i in range(len(hidden_layer_dims) - 1):
      actor.append(torch.nn.Linear(hidden_layer_dims[i], hidden_layer_dims[i+1]))
      actor.append(torch.nn.ReLU())
    actor.append(torch.nn.Linear(hidden_layer_dims[-1], output_size))
    return torch.nn.Sequential(*actor)

  def _create_img_feature_extractor(self, frozen=False):
    """
    ResNet18 backbone with last fc layer chopped off
    Weights frozen
    Ouput shape [1, 512, 1, 1]
    """
    resnet = torchvision.models.resnet18(weights='DEFAULT')
    modules = list(resnet.children())[:-1]
    backbone = torch.nn.Sequential(*modules)
    backbone.requires_grad_(not frozen)
    return backbone

  def forward(self, state, side_image, gripper_image):
    img_features_side = torch.squeeze(self.img_feature_extractor_side(side_image), dim=[2, 3])
    img_features_gripper = torch.squeeze(self.img_feature_extractor_gripper(gripper_image), dim=[2, 3])
    input = torch.hstack((state, img_features_side, img_features_gripper))
    out = self.actor(input)
    # Gripper sigmoid
    out[:, 6:8] = self.sigmoid(out[:, 6:8])
    return out

  
  def predict(self, state, side_image, gripper_image, episode_start=None, deterministic=None):
    return self.forward(state, side_image, gripper_image)



# %%
from lerobot.common.policies.normalize import Normalize, Unnormalize

class Trainer:
  def __init__(self, params) -> None:
    # self.env = env # This breaks caching of preprocess_data

    self.params = params
    assert(len(self.params["joint_bounds"]["centre"]) > 1)
    assert(len(self.params["joint_bounds"]["range"]) == len(self.params["joint_bounds"]["centre"]))

  def normalize_qpos(self, qpos):
    """
    Scale from joint bounds to (-1, 1)
    """
    return (qpos - self.params["joint_bounds"]["centre"]) / self.params["joint_bounds"]["range"] * 2

  def unnormalize_qpos(self, qpos):
    """
    Scale from (-1, 1) to joint bounds
    """
    return (qpos / 2) * self.params["joint_bounds"]["range"] + self.params["joint_bounds"]["centre"]
  
  # TODO: clamp output to joint bounds
  # def clamp_qpos(self, qpos):

  #   torch.clamp(qpos, min=)

  
  def embed_gripper(self, gripper):
    """
    Convert from (-1, 1) to one hot encoded
    One hot needs them as 1d
    """
    return torch.nn.functional.one_hot(gripper + 1, num_classes=3)

  def decode_gripper(self, gripper):
    """
    Convert from one hot encoded to column vector in range (-1, 1)
    """
    return (torch.argmax(gripper, dim=1) - 1).unsqueeze(1).to(int)

  def batched_preprocess(self, batch):
    """
    Take a batch of data and put it in a suitable tensor format for the model
    Batches here are as a list
    batch: action.qpos : b * t * 6, where b is batch size and t is number of time samples
    action.gripper: b * t
    """
    
    observation_gripper = self.embed_gripper(batch["observation.state.gripper"]).to(torch.float32)
    action_gripper = self.embed_gripper(batch["action.gripper"]).to(torch.float32)
    
    if self.params["normalize_qpos"] is not False:
      batch["observation.state.qpos"] = self.normalize_qpos(batch["observation.state.qpos"])
      batch["action.qpos"] = self.normalize_qpos(batch["action.qpos"])
    
    batch["preprocessed.action.state.qpos"] = torch.cat((batch["action.qpos"], action_gripper), dim=-1)
    batch["preprocessed.observation.state.qpos"] = torch.cat((batch["observation.state.qpos"], observation_gripper), dim=-1)

    return batch
   
  
  def evaluate_policy(self, env, policy, n):
    avg_reward = 0
    for i in range(n):
      numpy_observation, info = env.reset()

      # Prepare to collect every rewards and all the frames of the episode,
      # from initial state to final state.
      rewards = []
      frames = []
      action = {}

      # Render frame of the initial state
      frames.append(env.render())

      step = 0
      done = False
      while not done and len(frames) < 300:
        # Prepare observation for the policy running in Pytorch
        # Get qpos in range (-1, 1), gripper is already in range (-1, 1)
        qpos = torch.from_numpy(numpy_observation["state"]["qpos"]).unsqueeze(0)
        gripper = self.embed_gripper(torch.tensor(numpy_observation["state"]["gripper"])).unsqueeze(0)
        if self.params["normalize_qpos"]:
          qpos = self.normalize_qpos(qpos)
        state = torch.hstack((qpos, gripper))
        image_side = torch.from_numpy(numpy_observation["pixels"]["side"]).permute(2, 0, 1).unsqueeze(0) / 255
        image_gripper = torch.from_numpy(numpy_observation["pixels"]["gripper"]).permute(2, 0, 1).unsqueeze(0) / 255
        
        # Convert to float32 with image from channel first in [0,255]
        # to channel last in [0,1]
        state = state.to(torch.float32)

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        image_side = image_side.to(device, non_blocking=True)
        image_gripper = image_gripper.to(device, non_blocking=True)

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
          raw_action = policy.predict(state, image_side, image_gripper).to("cpu")
        
        action["qpos"] = raw_action[:, :6]
        if self.params["normalize_qpos"]:
          action["qpos"] = self.unnormalize_qpos(action["qpos"])
        
        action["qpos"] = action["qpos"].flatten().numpy()
        action["gripper"] = self.decode_gripper(raw_action[:, 6:8]).item()

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(action)
        # Keep track of all the rewards and frames
        rewards.append(reward)
        frames.append(env.render())

        # The rollout is considered done when the success state is reach (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1
      
      avg_reward += rewards[-1]/n
    
      return avg_reward, frames



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

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")


  # %%
  task = gym_lite6.pickup_task.PickupTask('gripper_left_finger', 'gripper_right_finger', 'box', 'floor')
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

  dataset = load_from_disk("datasets/pickup/scripted_trajectories_50_2024-08-02_12-49-56.hf")
  if "from" not in dataset.column_names:
    first_frames=dataset.filter(lambda example: example['frame_index'] == 0)
    from_idxs = torch.tensor(first_frames['index'])
    to_idxs = torch.tensor(first_frames['index'][1:] + [len(dataset)])
    episode_data_index={"from": from_idxs, "to": to_idxs}
      
  dataset.set_transform(hf_transform_to_torch)
  # dataset.set_transform(lambda x: trainer.lerobot_preprocess(hf_transform_to_torch(x)))
  # dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)


  # %%
  from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
  lerobot_dataset = LeRobotDataset.from_preloaded(root=Path("datasets/scripted_trajectories_50_2024-08-02_12-49-56.hf"),
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
  dataloader = DataLoader(
          lerobot_dataset,
          num_workers=4,
          batch_size=128,
          shuffle=True,
          # sampler=sampler,
          pin_memory=device.type != "cpu",
          drop_last=False,
      )
  

  policy = MLPPolicy([64, 64, 64]).to(device)
  optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
  loss_fn = torch.nn.MSELoss()

  if args.checkpoint is None:
    print("train from scratch")
    start_epoch = 0
    step = 0
    params = {}
    params["normalize_qpos"] = False
    loss=torch.tensor(0)

  else:
    checkpoint = torch.load(args.checkpoint)
    start_epoch = checkpoint["epoch"] + 1
    step = checkpoint["step"]
    params = checkpoint["params"]
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss = checkpoint["loss"]
    print(f"Loaded checkpoint at epoch {start_epoch}")

  # TODO: Maybe check that these are the same as what is loaded from checkpoint?
  jnt_range_low = env.unwrapped.model.jnt_range[:6, 0]
  jnt_range_high = env.unwrapped.model.jnt_range[:6, 1]
  bounds_centre = torch.tensor((jnt_range_low + jnt_range_high) / 2, dtype=torch.float32)
  bounds_range = torch.tensor(jnt_range_high - jnt_range_low, dtype=torch.float32)
  params["joint_bounds"] = {"centre": bounds_centre, "range": bounds_range}

  trainer = Trainer(params)

  

  curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  hidden_layer_dims = '_'.join([str(x.out_features) for x in policy.actor[:-1] if 'out_features' in x.__dict__])
  OUTPUT_FOLDER=f'ckpts/lite6_pick_place_h{hidden_layer_dims}_{curr_time}'
  Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

  if args.eval:
    print("Evaluating...")
    policy.eval()
    print(f"Epoch: {start_epoch}, steps: {step}, loss: {loss.item()}")
    avg_reward, frames = trainer.evaluate_policy(env, policy, 5)
    media.write_video(OUTPUT_FOLDER + f"/epoch_{start_epoch}.mp4", frames, fps=env.metadata["render_fps"])
  
  else:
    writer = SummaryWriter(log_dir=f"runs/lite6_pick_place/{curr_time}")

    end_epoch = start_epoch+args.n_epochs
    for epoch in range(start_epoch, end_epoch+1):
      policy.train()
      end = time.time()
      for batch in tqdm(dataloader):
        data_load_time = time.time()

        batch = trainer.batched_preprocess(batch)

        # Send data tensors from CPU to GPU
        state = batch["preprocessed.observation.state.qpos"].to(device, non_blocking=True)
        image_side = batch["observation.pixels.side"].to(device, non_blocking=True)
        image_gripper = batch["observation.pixels.gripper"].to(device, non_blocking=True)

        # Because we sample the action ahead in time [0, 0.1], it has an extra dimension, and we select the last dim
        a_hat = batch["preprocessed.action.state.qpos"][:, -1, :].to(device, non_blocking=True)
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
      
      if epoch % 2 == 0 or epoch == end_epoch-1:
        # Evaluate
        policy.eval()
        print(f"Epoch: {epoch}/{end_epoch}, steps: {step}, loss: {loss.item()}")
        avg_reward, frames = trainer.evaluate_policy(env, policy, 5)
        media.write_video(OUTPUT_FOLDER + f"/epoch_{epoch}.mp4", frames, fps=env.metadata["render_fps"])
        print("avg reward: ", avg_reward)
        writer.add_scalar("Reward/val", avg_reward, step)
        # _, frames = evaluate_policy(policy, env, 1, visualise=True)
        writer.add_images("Image", np.stack([frames[x].transpose(2, 0, 1) for x in range(0, len(frames), 50)], axis=0), step)
      
        writer.add_scalar("Time/eval_time", time.time() - end, step)


      if epoch % 10 == 0 or epoch == end_epoch-1:
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

