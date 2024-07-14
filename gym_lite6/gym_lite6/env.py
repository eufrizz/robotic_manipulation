import gymnasium as gym
from gymnasium import utils, spaces
import mujoco
from mujoco import minimize
import numpy as np
from pathlib import Path
import copy

MODEL_DIR = Path(__file__).parent.parent.parent.resolve() / "models"  # note: absolute path

def compare(var1, var2):
    for attr in dir(var1):
        print(f"{attr}:", getattr(var1, attr))
        print(f"{attr}:", getattr(var2, attr))
        print(var1 == var2)
        print()


class UfactoryLite6Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    """
    Action space: [qpos (6),        # joint angles
                   gripper (1)      # gripper state (-1: close, 0: off, 1: open], discrete]
    Observation space: 
    """
    def __init__(
        self,
        task,
        # xml_file: str = str(MODEL_DIR/"lite6_viz.xml"),
        xml_file: str = str(MODEL_DIR/"cube_pickup.xml"),
        obs_type="pixels_pose",
        render_mode="rgb_array",
        visualization_width: int=244,
        visualization_height: int=244,
    ):
        # xml_file = MODEL_DIR/"lite6_viz.xml"
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        # Separate data to do IK with that doesn't intefere with the sim
        self.ik_data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=visualization_height, width=visualization_width)
        self.dof = 6
        

        self.voption = mujoco.MjvOption()
        # self.voption.frame = mujoco.mjtFrame.mjFRAME_SITE

        mujoco.mj_forward(self.model, self.data)

        self.bounds = [self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1]]

        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        self.external_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.external_camera)
        self.external_camera.distance = 0.6
        self.external_camera.elevation = -30
        self.external_camera.azimuth = -130
        self.external_camera.lookat = (0.3, 0.3, 0.2)
        
        # TODO: Move cameras to XML
        self.top_camera = mujoco.MjvCamera()
        # mujoco.mjv_defaultFreeCamera(self.model, self.top_camera)
        self.top_camera.distance = 0.6
        self.top_camera.elevation = -90
        self.top_camera.azimuth = 0
        self.top_camera.lookat = (0.3, 0, 0.2)

        self.side_camera = mujoco.MjvCamera()
        # mujoco.mjv_defaultFreeCamera(self.model, self.side_camera)
        self.side_camera.distance = 1
        self.side_camera.elevation = 0
        self.side_camera.azimuth = 180
        self.side_camera.lookat = (0, 0, 0.1)

        self.ego_camera = mujoco.MjvCamera()
        # mujoco.mjv_defaultFreeCamera(self.model, self.ego_camera)
        self.ego_camera.distance = 0.2
        self.ego_camera.elevation = -80
        self.ego_camera.azimuth = 0


        if self.obs_type == "state":
            # Pos and vel? Originally unimplemented
            self.observation_space = spaces.Dict(
              {
                "qpos": spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64),
                "gripper": spaces.Discrete(3, start=-1) # release, off, grip
              }
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "image": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.visualization_height, self.visualization_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_pose":
            self.observation_space = spaces.Dict(
                {
                  "pixels": spaces.Box(
                      low=0,
                      high=255,
                      shape=(self.visualization_height, self.visualization_width, 3),
                      dtype=np.uint8,
                  ),
                  "state": spaces.Dict(
                      {
                        "pose" : spaces.Box(
                          low=-100.0,
                          high=100.0,
                          shape=(7,),
                          dtype=np.float64,
                        ),
                        "gripper" : spaces.Discrete(3, start=-1) # This could be continuous but we don't get gripper position feedback
                      }
                  ),
                }
            )
        elif self.obs_type == "pixels_state":
          self.observation_space = spaces.Dict(
              {
                "pixels": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.visualization_height, self.visualization_width, 3),
                    dtype=np.uint8,
                ),
                  "state": spaces.Dict(
                    {
                      "qpos": spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64),
                      "gripper": spaces.Discrete(3, start=-1) # release, off, grip
                    }
                  ),
              }
          )
        else:
          raise KeyError(f"Invalid observation type {self.obs_type}")

        # Pos and quaternion
        self.action_space = spaces.Dict(
            {
              # "pos": spaces.Box(low=np.array([-1, -1, -1, 0, 0, 0, 0]), high=np.array([1, 1, 1, 1, 1, 1, 1]), shape=(7,), dtype=np.float32),
              # "pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
              "qpos": spaces.Box(low=self.model.jnt_range[:6, 0], high=self.model.jnt_range[:6, 1], dtype=np.float64),
              "gripper": spaces.Discrete(3, start=-1) # release, off, grip
            }
        )

        self.object_space = spaces.Box(low=np.array([0.1, -0.4, 0, 0, 0, 0, 0]), high=np.array([0.4, 0.4, 0, 1, 1, 1, 1]), dtype=np.float32)

    def gripper_action_to_force(self, action):
        """
        Map (-1, 1) to (min_force, max_force)
        """
        force =  {
          -1: self.model.actuator('gripper').ctrlrange[0],
           0: 0,
           1: self.model.actuator('gripper').ctrlrange[1],
        }[action]
        return force

    def force_to_gripper_action(self, force):
        """
        Map (-force limit, force limit) to discrete (-1, 1)
        """
        if force < 1e-3:
            return -1
        elif force > 1e-3:
            return 1
        else:
            return 0
        
    def normalize_qpos(self, qpos):
        """
        map from joint bounds to (-1, 1)
        """
        if len(qpos.shape) == 1:
            qpos = np.atleast_2d(qpos)
        assert(qpos.shape[1] == 6), qpos
        bounds_centre = (self.model.jnt_range[:6, 0] + self.model.jnt_range[:6, 1]) / 2
        bounds_range = (self.model.jnt_range[:6, 1] - self.model.jnt_range[:6, 0])
        return (qpos - bounds_centre) * 2.0 / bounds_range
    
    def unnormalize_qpos(self, qpos):
        """
        map from (-1, 1) to joint bounds
        """
        if len(qpos.shape) == 1:
            qpos = np.atleast_2d(qpos)
        assert(qpos.shape[1] == 6), qpos
        bounds_centre = (self.model.jnt_range[:6, 0] + self.model.jnt_range[:6, 1]) / 2
        bounds_range = (self.model.jnt_range[:6, 1] - self.model.jnt_range[:6, 0])
        return (qpos - bounds_centre) /2.0 * bounds_range
    
    def map_bounds(self, vals, in_range=None, out_range=None):
        """
        Todo: move to utils, write test
        """
        if in_range is None and out_range is None:
            raise ValueError("One of in_range or out_range must be specified")
        if in_range is None:
            in_range = self.model.jnt_range[:6]
        elif out_range is None:
            out_range = self.model.jnt_range[:6]
        
        # if len(in_range.shape) == 1:
        #     in_range = np.tile(in_range, 6)
        # if len(out_range.shape) == 1:
        #     out_range = np.tile(out_range, 6)
        
        assert(vals.shape[0] == in_range.shape[0] == out_range.shape[0])
        assert(len(vals.shape) == 1)
        assert(2 == in_range.shape[1] == out_range.shape[1])
        in_range_centre = (in_range[:, 1] + in_range[:, 0]) / 2
        in_range_bounds= in_range[:, 1] - in_range[:, 0]

        out_range_centre = (out_range[:, 1] + out_range[:, 0]) / 2
        out_range_bounds= out_range[:, 1] - out_range[:, 0]

        return (vals - in_range_centre) / in_range_bounds * out_range_bounds + out_range_centre

    def render(self, show_sites=False):
        self.ego_camera.lookat = self.data.site('end_effector').xpos

        # self.renderer.update_scene(self.data, self.external_camera, self.voption)
        self.renderer.update_scene(self.data, self.side_camera, self.voption)

        return self.renderer.render()

    # def _render(self, visualize=False):
    #     assert self.render_mode == "rgb_array"
    #     width, height = (
    #         (self.visualization_width, self.visualization_height)
    #         if visualize
    #         else (self.visualization_width, self.visualization_height)
    #     )
    #     # if mode in ["visualize", "human"]:
    #     #     height, width = self.visualize_height, self.visualize_width
    #     # elif mode == "rgb_array":
    #     #     height, width = self.visualization_height, self.visualization_width
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


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:6] = self.observation_space["state"]["qpos"].sample()
        # gripper 6,7
        box_pose = self.object_space.sample()
        # Drop from height to avoid intersection with ground
        self.data.qpos[8:11] = box_pose[:3] + np.array([0, 0, 0.01])
        # Quaternion pose - normalise
        z_rot = np.random.rand()
        quat = np.array([1-z_rot, 0, 0, z_rot])
        self.data.qpos[11:] = quat / np.linalg.norm(quat)
        mujoco.mj_forward(self.model, self.data)
        
        # Ensure robot is not self-intersecting
        # TODO: don't hardcode geoms
        while any(np.isin(self.data.contact.geom.flatten(), np.arange(1, 7))):
            self.data.qpos[:6] = self.observation_space["state"]["qpos"].sample()
            mujoco.mj_forward(self.model, self.data)


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
        # assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?
        self.data.ctrl[:6] = action["qpos"]
        self.data.ctrl[6] = self.gripper_action_to_force(action["gripper"])
        
        timesteps_per_frame = int(1 / self.metadata["render_fps"] / self.model.opt.timestep)
        for i in range(timesteps_per_frame):
            mujoco.mj_step(self.model, self.data)
        observation = self._get_observation()
        # _, reward, _, raw_obs = self._env.step(action)
        reward = self.task.get_reward(self.model, self.data)
        terminated = False
        truncated = False
        info = {}

        # TODO(rcadene): add an enum
        terminated = is_success = reward == self.task.max_reward

        info = {"is_success": is_success}

        # truncated = False
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        if self.obs_type == "pixels_state":
          qpos = self.data.qpos[:6]
          gripper = self.force_to_gripper_action(self.data.actuator('gripper').ctrl)
          observation =  {"state": {"qpos": qpos, "gripper": gripper}, "pixels": self.render()}

        else:
          raise NotImplementedError()
          observation =  {"state": {"pose": np.hstack((pos, quat)), "gripper": 0}, "pixels": self.render()}
        return observation
        

    def solve_ik(self, pos, quat):
        """
        Solve for an end effector pose, return joint angles
        """

        x0 = self.data.qpos[:6]

        ik_target = lambda x: self.ik(x, pos=pos, quat=quat, radius=0.5,
                                reg_target=x0, reg=0.1)
        x, _ = minimize.least_squares(x0, ik_target, self.bounds,
                                    jacobian=self.ik_jac,
                                    verbose=0)

        return x

    def ik(self, x, pos, quat, radius=0.04, reg=1e-3, reg_target=None, ref_frame='end_effector'):
        """Residual for inverse kinematics.

        Args:
            x: numpy column vector of joint angles.
            pos: target position for the end effector.
            quat: target orientation for the end effector.
            radius: scaling of the 3D cross.

        Returns:
            The residual of the Inverse Kinematics task.
        """

        # Move the mocap body to the target
        id = self.model.body('target').mocapid
        self.ik_data.mocap_pos[id] =  pos
        self.ik_data.mocap_quat[id] = quat

        res = []
        # For batched operation, each column can be a different x
        for i in range(x.shape[1]):
            # Forward kinematics for given state
            self.ik_data.qpos[:6] = x[:, i]
            mujoco.mj_kinematics(self.model, self.ik_data)

            # Position residual
            res_pos = self.ik_data.site(ref_frame).xpos - self.ik_data.site('target').xpos
            # print(self.ik_data.site(ref_frame).xpos)
            
            # Get the ref frame orientation (convert from rotation matrix to quaternion)
            ref_quat = np.empty(4)
            mujoco.mju_mat2Quat(ref_quat, self.ik_data.site(ref_frame).xmat)

            # Target quat, exploit the fact that the site is aligned with the body.
            target_quat = self.ik_data.body('target').xquat

            # Orientation residual: quaternion difference.
            res_quat = np.empty(3)
            mujoco.mju_subQuat(res_quat, target_quat, ref_quat)
            res_quat *= radius

            res_reg = reg * (x[:, i] - reg_target)
            
            res_i = np.hstack((res_pos, res_quat, res_reg))
            res.append(np.atleast_2d(res_i).T)
        
        return np.hstack(res)
    
    def ik_jac(self, x, res=None, radius=0.04, reg=1e-3, ref_frame='end_effector'):
        """Analytic Jacobian of inverse kinematics residual

        Args:
            x: joint angles.
            res: least_squares() passes the value of the residual at x which is sometimes useful, but we don't need it here.
            radius: scaling of the 3D cross.

        Returns:
            The Jacobian of the Inverse Kinematics task.
            (3 + 3 + nv)  * nv
        """
        mujoco.mj_kinematics(self.model, self.ik_data)
        mujoco.mj_comPos(self.model, self.ik_data) #calculate CoM position

        # Get end-effector site Jacobian.
        jac_pos = np.empty((3, self.model.nv))
        jac_quat = np.empty((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.ik_data, jac_pos, jac_quat, self.ik_data.site(ref_frame).id)
        jac_pos = jac_pos[:, :self.dof]
        jac_quat = jac_quat[:, :self.dof]

        # Get the ref frame orientation (convert from rotation matrix to quaternion)
        ref_quat = np.empty(4)
        mujoco.mju_mat2Quat(ref_quat, self.ik_data.site(ref_frame).xmat)
        
        # Get Deffector, the 3x3 Jacobian for the orientation difference
        target_quat = self.ik_data.body('target').xquat
        Deffector = np.empty((3, 3))
        mujoco.mjd_subQuat(target_quat, ref_quat, None, Deffector)

        # Rotate into target frame, multiply by subQuat Jacobian, scale by radius.
        target_mat = self.ik_data.site('target').xmat.reshape(3, 3)
        mat =  Deffector.T @ target_mat.T
        jac_quat = radius * mat @ jac_quat

        # Regularization Jacobian
        jac_reg = reg * np.eye(self.dof)

        return np.vstack((jac_pos, jac_quat, jac_reg))