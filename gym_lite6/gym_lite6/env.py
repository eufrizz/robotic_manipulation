import gymnasium as gym
from gymnasium import utils, spaces
import mujoco
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent.parent.resolve() / "models"  # note: absolute path

class UfactoryLite6Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    """
    Action space: [pose (7),        # position and quaternion orientation of end effector
                   gripper (1)      # gripper state (0: close, 1: open], discrete]
    Observation space: 
    """
    def __init__(
        self,
        # task,
        # xml_file: str = str(MODEL_DIR/"lite6_viz.xml"),
        xml_file: str = str(MODEL_DIR/"cube_pickup.xml"),
        obs_type="pixels_pose",
        render_mode="rgb_array",
        observation_width: int =640,
        observation_height: int=480,
        visualization_width: int=640,
        visualization_height: int=480,
    ):
        # xml_file = MODEL_DIR/"lite6_viz.xml"
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        # self.renderer = mujoco.Renderer(self.model)

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 1.2
        self.camera.elevation = -15
        self.camera.azimuth = -130
        self.camera.lookat = (0, 0, 0.3)

        self.bounds = [self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]]

        # self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height


        if self.obs_type == "state":
            # Pos and vel? Originally unimplemented
            self.observation_space = spaces.Dict(
              {
                "pose" : spaces.Box(
                  low=-100.0,
                  high=100.0,
                  shape=(self.model.nu,),
                  dtype=np.float64,
                ),
                "gripper" : spaces.Discrete( # This could be continuous but we don't get gripper position feedback
                  low=0,
                  high=1,
                  dtype=int,
                )
              }
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "image": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_pose":
            self.observation_space = spaces.Dict(
                {
                  "image": spaces.Box(
                      low=0,
                      high=255,
                      shape=(self.observation_height, self.observation_width, 3),
                      dtype=np.uint8,
                  ),
                  "state": spaces.Dict(
                      {
                        "pose" : spaces.Box(
                          low=-100.0,
                          high=100.0,
                          shape=(self.model.nu,),
                          dtype=np.float64,
                        ),
                        "gripper" : spaces.Discrete(2) # This could be continuous but we don't get gripper position feedback
                      }
                  ),
                }
            )
          # elif self.obs_type == "pixels_state":
          #   self.observation_space = spaces.Dict(
          #       {
          #           "images": spaces.Dict(
          #               {
          #                   "top": spaces.Box(
          #                       low=0,
          #                       high=255,
          #                       shape=(self.observation_height, self.observation_width, 3),
          #                       dtype=np.uint8,
          #                   )
          #               }
          #           ),
          #           "state": spaces.Dict(
          #               {
          #                 spaces.Box(
          #               low=-100.0,
          #               high=100.0,
          #               shape=(self.model.nv + 1,),
          #               dtype=np.float64,
          #           ),
          #       }
          #   )
        else:
          raise KeyError(f"Invalid observation type {self.obs_type}")

        # Pos and quaternion
        self.action_space = spaces.Dict(
            {
              "pos": spaces.Box(low=np.array([-1, -1, -1, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1, 1, 1]), shape=(7,), dtype=np.float32),
              "gripper": spaces.Discrete(2)
            }
        )

    def render(self):
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    # def _render(self, visualize=False):
    #     assert self.render_mode == "rgb_array"
    #     width, height = (
    #         (self.visualization_width, self.visualization_height)
    #         if visualize
    #         else (self.observation_width, self.observation_height)
    #     )
    #     # if mode in ["visualize", "human"]:
    #     #     height, width = self.visualize_height, self.visualize_width
    #     # elif mode == "rgb_array":
    #     #     height, width = self.observation_height, self.observation_width
    #     # else:
    #     #     raise ValueError(mode)
    #     # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
    #     image = self._env.physics.render(height=height, width=width, camera_id="top")
        
    #     return image

    # def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        # time_limit = float("inf")

        # if "cube" in task_name:
        #     xml_path = MODEL_DIR / "cube_pickup.xml"
        #     physics = mujoco.Physics.from_xml_path(str(xml_path))
        #     # task = CubePickupTask()
        # else:
        #     raise NotImplementedError(task_name)

        # env = control.Environment(
        #     physics, task, time_limit, control_timestep=DT, n_sub_steps=None, flat_observation=False
        # )
        # return env

    # def _format_raw_obs(self, raw_obs):
    #     if self.obs_type == "state":
    #         raise NotImplementedError()
    #     elif self.obs_type == "pixels":
    #         obs = {"top": raw_obs["images"]["top"].copy()}
    #     elif self.obs_type == "pixels_agent_pos":
    #         obs = {
    #             "pixels": {"top": raw_obs["images"]["top"].copy()},
    #             "agent_pos": raw_obs["qpos"],
    #         }
    #     return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        observation = self._get_observation()
        info = {}

        return observation, info

        # # TODO(rcadene): how to seed the env?
        # if seed is not None:
        #     self._env.task.random.seed(seed)
        #     self._env.task._random = np.random.RandomState(seed)

        # # TODO(rcadene): do not use global variable for this
        # if "transfer_cube" in self.task:
        #     BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        # elif "insertion" in self.task:
        #     BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        # else:
        #     raise ValueError(self.task)

        # raw_obs = self._env.reset()

        # observation = self._format_raw_obs(raw_obs.observation)

        # info = {"is_success": False}
        # return observation, info

    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        mujoco.mj_step(self.model, self.data)
        observation = self._get_observation()
        # _, reward, _, raw_obs = self._env.step(action)
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # TODO(rcadene): add an enum
        # terminated = is_success = reward == 4

        # info = {"is_success": is_success}

        # observation = self._format_raw_obs(raw_obs)

        # truncated = False
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        pos = self.data.site('attachment_site').xpos
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site('attachment_site').xmat)
        observation =  {"state": {"pos": np.stack(pos, quat), "gripper": 0}, "image": self.render()}
        return observation
        

    def close(self):
        pass