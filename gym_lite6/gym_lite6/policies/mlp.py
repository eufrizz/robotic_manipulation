import torch
import torchvision

class MLPPolicy(torch.nn.Module):
  def __init__(self, hidden_layer_dims, input_state_dims=9, output_dims=9, dropout=False):
    """
    state_dims: 6 for arm, 3 for gripper
    """
    super().__init__()

    # self.img_feature_extractor = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet18', )
    
    self.img_feature_extractor_side = self._create_img_feature_extractor()
    self.img_feature_extractor_gripper = self._create_img_feature_extractor()
    # Resnet output is 1x512, 2 bits for gripper
    self.actor = self._create_actor(512*2 + input_state_dims, hidden_layer_dims, output_dims, dropout)
    self.sigmoid = torch.nn.Sigmoid()
  
  def _create_actor(self, input_size, hidden_layer_dims, output_size, dropout):
    actor = []
    actor.append(torch.nn.Linear(input_size, hidden_layer_dims[0]))
    actor.append(torch.nn.ReLU())
    for i in range(len(hidden_layer_dims) - 1):
      actor.append(torch.nn.Linear(hidden_layer_dims[i], hidden_layer_dims[i+1]))
      actor.append(torch.nn.ReLU())
      if dropout:
        actor.append(torch.nn.Dropout(p=0.3))
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
    out[:, -2:] = self.sigmoid(out[:, -2:])
    return out

  
  def predict(self, state, side_image, gripper_image, episode_start=None, deterministic=None):
    return self.forward(state, side_image, gripper_image)





# Not so much for training but more for using the model
class Interface:
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
    
    if "use_obs_vel" in self.params and self.params["use_obs_vel"]:
      batch["preprocessed.observation.state.qpos"] = torch.cat((batch["observation.state.qpos"], batch["observation.state.qvel"], observation_gripper), dim=-1)
    else:
      batch["preprocessed.observation.state.qpos"] = torch.cat((batch["observation.state.qpos"], observation_gripper), dim=-1)

    batch["preprocessed.action.state.qpos"] = torch.cat((batch["action.qpos"], action_gripper), dim=-1)

    return batch
   
  
  def evaluate_policy(self, env, policy, n, qpos0=None, box_pos0=None, box_quat0=None):
    avg_reward = 0
    for i in range(n):
      numpy_observation, info = env.reset(qpos=qpos0, box_pos=box_pos0, box_quat=box_quat0)

      # Prepare to collect every rewards and all the frames of the episode,
      # from initial state to final state.
      rewards = []
      frames = []
      action = {}

      # Render frame of the initial state
      frames.append(env.render())

      step = 0
      done = False
      while not done:
        # Prepare observation for the policy running in Pytorch
        # Get qpos in range (-1, 1), gripper is already in range (-1, 1)
        qpos = torch.from_numpy(numpy_observation["state"]["qpos"]).unsqueeze(0)
        qvel = torch.from_numpy(numpy_observation["state"]["qvel"]).unsqueeze(0)
        gripper = self.embed_gripper(torch.tensor(numpy_observation["state"]["gripper"])).unsqueeze(0)
        if self.params["normalize_qpos"]:
          qpos = self.normalize_qpos(qpos)
        state = torch.hstack((qpos, qvel, gripper))
        image_side = torch.from_numpy(numpy_observation["pixels"]["side"]).permute(2, 0, 1).unsqueeze(0) / 255
        image_gripper = torch.from_numpy(numpy_observation["pixels"]["gripper"]).permute(2, 0, 1).unsqueeze(0) / 255
        
        # Convert to float32 with image from channel first in [0,255]
        # to channel last in [0,1]
        state = state.to(torch.float32)

        # Send data tensors from CPU to GPU
        state = state.to(self.params["device"], non_blocking=True)
        image_side = image_side.to(self.params["device"], non_blocking=True)
        image_gripper = image_gripper.to(self.params["device"], non_blocking=True)

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

        # Normally we would exit on terminated = True, but we want our evaluation to continue
        # so it can collect the full reward
        done = truncated | done
        step += 1
      
      avg_reward += sum(rewards)/len(rewards)/n
    
    return avg_reward, frames

