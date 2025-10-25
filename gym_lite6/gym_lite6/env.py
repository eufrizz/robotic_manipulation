import gymnasium as gym
from gymnasium import utils, spaces
import mujoco
from mujoco import minimize
import numpy as np
from pathlib import Path
from copy import deepcopy

# MENAGERIE_DIR = Path(__file__).parent.parent.parent.resolve() / "mujoco_menagerie"  # note: absolute path
MODEL_DIR = Path(__file__).parent.parent.resolve() / "models"  # note: absolute path

class UfactoryLite6Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    """
    Action space: [qpos (6),        # joint angles
                   gripper (1)      # gripper state (-1: close, 0: off, 1: open], discrete]
    Observation space: 
    """
    def __init__(
        self,
        task,
        model_xml: str = str(MODEL_DIR/"lite6_gripper_wide.xml"),
        obj_xml: str = str(MODEL_DIR/"cube_pickup_large.xml"),
        scene_xml: str = str(MODEL_DIR/"scene.xml"),
        obs_type="pixels_state",
        action_type="qpos",
        render_mode="rgb_array",
        visualization_width: int=640,
        visualization_height: int=480,
        render_fps=30,
        joint_noise_magnitude=0
    ):
        self.metadata["render_fps"] = render_fps
        super().__init__()

        self.joint_noise_magnitude=joint_noise_magnitude

        self.model = self.load_xmls([scene_xml, model_xml, obj_xml])
        self.data = mujoco.MjData(self.model)
        # Separate data to do IK with that doesn't intefere with the sim
        self.ik_data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=visualization_height, width=visualization_width)
        # Start in position control by default
        self.disable_actuator_group(2)
        # These are the indexes of our joints, from the XML
        self.joint_qpos = [0, 1, 2, 3, 4, 5]
        self.dof = len(self.joint_qpos)
        
        self.voption = mujoco.MjvOption()
        # self.voption.frame = mujoco.mjtFrame.mjFRAME_SITE

        mujoco.mj_forward(self.model, self.data)

        self.bounds = [self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1]]

        if hasattr(task, '__call__'):
            self.task = task()
        else:
            self.task = task
        self.task_description = self.task.task_description
        self.obs_type = obs_type
        self.action_type = action_type
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

        # Now defined in XML
        # self.side_camera = mujoco.MjvCamera()
        # # mujoco.mjv_defaultFreeCamera(self.model, self.side_camera)
        # self.side_camera.distance = 1
        # self.side_camera.elevation = 0
        # self.side_camera.azimuth = 180
        # self.side_camera.lookat = (0, 0, 0.1)

        if self.obs_type == "state":
            raise NotImplementedError()
            # Pos and vel? Originally unimplemented
            self.observation_space = spaces.Dict(
              {
                "qpos": spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float64),
                "gripper": spaces.Discrete(3, start=-1) # release, off, grip
              }
            )
        elif self.obs_type == "pixels":
            raise NotImplementedError()
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
            raise NotImplementedError()
            self.observation_space = spaces.Dict(
                {
                "pixels": spaces.Dict(
                    {
                    "side": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.visualization_height, self.visualization_width, 3),
                        dtype=np.uint8,
                    ),
                    "gripper": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.visualization_height, self.visualization_width, 3),
                        dtype=np.uint8,
                    ),
                    }
                ),
                  "ee_pose": spaces.Dict(
                    {
                        "pos": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64),
                        "quat": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float64),
                        "vel": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64),
                        "ang_vel": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float64),
                    }
                ),
              }
            )
        elif self.obs_type == "pixels_state" or self.obs_type == "pixels_state_lerobot":
          self.observation_space = spaces.Dict(
            {
              "pixels": spaces.Dict(
                {
                "side": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.visualization_height, self.visualization_width, 3),
                    dtype=np.uint8,
                ),
                "gripper": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.visualization_height, self.visualization_width, 3),
                    dtype=np.uint8,
                ),
                }
              ),
              "state": spaces.Dict(
                  {
                      "qpos": spaces.Box(low=self.model.jnt_range[self.joint_qpos, 0], high=self.model.jnt_range[self.joint_qpos, 1], shape=(6,), dtype=np.float64),
                      "qvel": spaces.Box(low=-100, high=100, shape=(len(self.joint_qpos),), dtype=np.float64),
                      "gripper": spaces.Discrete(3, start=-1) # release, off, grip
                  }
              ),
              "ee_pose": spaces.Dict(
                  {
                      "pos": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64),
                      "quat": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float64),
                      "vel": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64),
                      "ang_vel": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float64),
                  }
              ),
            }
          )
          if self.obs_type == "pixels_state_lerobot":
              self.observation_space["agent_pos"] = spaces.Box(low=np.concatenate((self.model.jnt_range[self.joint_qpos, 0], [-1])), high=np.concatenate((self.model.jnt_range[self.joint_qpos, 1],[1])), shape=(7,), dtype=np.float64)
              print(f"{self.observation_space.keys()=}")
        else:
          raise KeyError(f"Invalid observation type {self.obs_type}")

        if self.action_type == "qpos":
            self.action_space = spaces.Dict(
                {
                "qpos": spaces.Box(low=self.model.jnt_range[self.joint_qpos, 0], high=self.model.jnt_range[self.joint_qpos, 1], dtype=np.float64),
                "gripper": spaces.Discrete(3, start=-1) # release, off, grip
                }
            )
            self.disable_actuator_group(2)
        elif self.action_type == "qvel":
            self.action_space = spaces.Dict(
                {
                "qvel": spaces.Box(low=-100, high=100, dtype=np.float64), # Arbitrarily high bounds
                "gripper": spaces.Discrete(3, start=-1) # release, off, grip
                }
            )
            self.disable_actuator_group(1)
        elif self.action_type == "qpos_gripper":
            self.action_space = spaces.Box(low=self.model.jnt_range[self.joint_qpos, 0]+[-1], high=self.model.jnt_range[self.joint_qpos, 1]+[1], dtype=np.float64)
            self.disable_actuator_group(2)
        # elif self.action_type == "ee_vel":
        #     self.action_space = spaces.Dict(
        #         {
        #         "ee_vel": spaces.Box(low=-100, high=100, dtype=np.float64), # Arbitrarily high bounds
        #         "ee_ang_vel": spaces.Box(low=-100, high=100, dtype=np.float64), # Arbitrarily high bounds
        #         "gripper": spaces.Discrete(3, start=-1) # release, off, grip
        #         }
        #     )
        #     self.disable_actuator_group(1)
        else:
          raise KeyError(f"Invalid action type {self.action_type}")

        box_min_height = self.model.geom('box').size[2] + 1e-3
        self.object_space = spaces.Box(low=np.array([0.1, -0.3, box_min_height, 0, 0, 0, 0]), high=np.array([0.4, 0.3, box_min_height, 1, 1, 1, 1]), dtype=np.float32)

    def load_xmls(self, xmls):
        """
        Using the first XML file as the base, add the other XML files to it
        https://mujoco.readthedocs.io/en/latest/python.html#model-editing
        Args:
        - xmls: a list of paths to mujoco XML files
        Returns:
        - Compiled model
        """
        parent_spec = mujoco.MjSpec.from_file(xmls[0])
        frame = parent_spec.worldbody.add_frame()
        for xml in xmls[1:]:
            spec = mujoco.MjSpec.from_file(xml)
            # Usually body 0 is world frame, body 1 is the next thing
            frame.attach(spec)
        return parent_spec.compile()


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
        TODO: Should this be (0, 1)?
        """
        if force < -1e-3:
            return -1
        elif force > 1e-3:
            return 1
        else:
            return 0
    
    def map_bounds(self, vals, in_range=None, out_range=None):
        """
        Todo: move to utils, write test
        """
        if in_range is None and out_range is None:
            raise ValueError("One of in_range or out_range must be specified")
        if in_range is None:
            in_range = self.model.jnt_range[self.joint_qpos]
        elif out_range is None:
            out_range = self.model.jnt_range[self.joint_qpos]
        
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

    def render(self, camera=0):

        # self.renderer.update_scene(self.data, self.external_camera, self.voption)
        self.renderer.update_scene(self.data, camera, self.voption)

        return self.renderer.render()


    def reset(self, seed=None, options=None, qpos=None, box_pos=None, box_quat=None):
        """
        If state is specified, it directly sets the mujoco qpos. Otherwise it is reset to a random state
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        if box_pos is None:
            box_pos = self.object_space.sample()[:3]
        # Ensure box is above the floor
        elif box_pos[2] < self.object_space.sample()[2]:
            box_pos[2] = self.object_space.sample()[2]
        if box_quat is None:
            z_rot = np.random.rand()
            # Quaternion orientation - normalise
            box_quat = np.array([1-z_rot, 0, 0, z_rot])
            box_quat = box_quat / np.linalg.norm(box_quat)
        
        self.data.qpos[8:11] = box_pos
        self.data.qpos[11:] = box_quat
        

        if qpos is not None:
            assert len(qpos) == self.dof, f"Got {len(qpos)=} instead of {self.dof=}"
            self.data.qpos[self.joint_qpos] = qpos
            mujoco.mj_forward(self.model, self.data)

        else:
            # Divide by two to get angles that are less extreme
            self.data.qpos[self.joint_qpos] = self.observation_space["state"]["qpos"].sample()/2
            mujoco.mj_forward(self.model, self.data)
        
            # Ensure robot is not self-intersecting
            # TODO: don't hardcode geoms. 0 is floor, 1-17 are the robot arm, 18 - 22 are gripper fingers (which may be touching)
            while any(np.isin(self.data.contact.geom.flatten(), np.arange(1, 18))):
                self.data.qpos[self.joint_qpos] = self.observation_space["state"]["qpos"].sample()/2
                mujoco.mj_forward(self.model, self.data)

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        # assert action.ndim == 1
        if isinstance(action, dict) and self.action_type not in action:
          raise NotImplementedError(f"Action does not correspond to selected action type {self.action_type}")
    
        if self.action_type == "qpos_gripper":
            # split qpos and gripper up
            action = {
                "qpos_gripper": action[:-1],
                "gripper": round(np.clip(action[-1], -1, 1))
            }
        
        timesteps_per_frame = int(1 / self.metadata["render_fps"] / self.model.opt.timestep)
        for i in range(timesteps_per_frame):
            self.data.ctrl[self.joint_actuators] = action[self.action_type]
            self.data.ctrl[self.model.actuator('gripper').id] = self.gripper_action_to_force(action["gripper"])

            # --- Add noise here ---
            if self.joint_noise_magnitude > 0:
                noise = self.np_random.normal(size=self.dof) * self.joint_noise_magnitude
                # self.data.qfrc_applied[self.joint_actuators] += noise
                self.data.ctrl[self.joint_actuators] += noise

            mujoco.mj_step(self.model, self.data)
        observation = self._get_observation()
        reward = self.task.get_reward(self.model, self.data)
        terminated = False
        truncated = False
        info = {}

        terminated = is_success = reward == self.task.max_reward

        info = {"is_success": is_success}
        # truncated = False
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self, ref_frame='end_effector'):
        if self.obs_type == "pixels_state" or self.obs_type == "pixels_state_lerobot":
            qpos = deepcopy(self.data.qpos[:self.dof])
            qvel = deepcopy(self.data.qvel[:self.dof])
            gripper = self.force_to_gripper_action(deepcopy(self.data.actuator('gripper').ctrl))

            pos = self.data.site(ref_frame).xpos
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, self.data.site(ref_frame).xmat)

            # Velocity in end effector frame
            ee_rot_vel = np.empty(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.model.site(ref_frame).id, ee_rot_vel, 1)
            ang_vel = deepcopy(ee_rot_vel[:3])
            vel = deepcopy(ee_rot_vel[3:])
            observation = {
                            "pixels": {
                                "side": self.render(camera="side_cam"),
                                "gripper": self.render(camera="gripper_cam")
                            },
                            "state": {
                                "qpos": qpos, "qvel": qvel, "gripper": gripper
                            },
                            "ee_pose": {
                                "pos": pos, "quat": quat, "vel": vel, "ang_vel": ang_vel
                            },
                        }
            if self.obs_type == "pixels_state_lerobot":
                observation["agent_pos"] = np.concatenate((qpos, np.array([gripper])))
        else:
            raise NotImplementedError()
            observation =  {"state": {"pose": np.hstack((pos, quat)), "gripper": 0}, "pixels": self.render()}
        return observation
        

    def solve_ik(self, pos, quat, init=None, radius=0.5, reg=0.01):
        """
        Solve for an end effector pose, return joint angles
        """

        if init is None:
            x0 = self.data.qpos[self.joint_qpos]
        else:
            x0 = init

        ik_target = lambda x: self.ik(x, pos=pos, quat=quat, radius=radius,
                                reg_target=x0, reg=reg)
        ik_jac_target = lambda x, res: self.ik_jac(x, radius=radius, reg=reg)

        x, _ = minimize.least_squares(x0, ik_target, self.bounds,
                                    jacobian=ik_jac_target,
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
            self.ik_data.qpos[self.joint_qpos] = x[:, i]
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
    
    def solve_dq(self, pos, quat, ref_frame='end_effector'):
        """
        Get dq, the error between the current pose and the desired, in state space, using damped ik.
        """
        jac = np.zeros((6, self.model.nv))
        twist = np.empty(6)
        error = np.zeros(6)
        curr_quat = np.empty(4)
        curr_quat_conj = np.empty(4)
        quat_err = np.empty(4)
        dq = np.zeros(6)
        site_id = self.model.site(ref_frame).id

        error[:3] = pos - self.data.site(site_id).xpos

        # Quat error
        mujoco.mju_mat2Quat(curr_quat, self.data.site(site_id).xmat)
        mujoco.mju_negQuat(curr_quat_conj, curr_quat)
        mujoco.mju_mulQuat(quat_err, quat, curr_quat_conj)
        mujoco.mju_quat2Vel(error[3:], quat_err, 1.0)

        # Get end-effector site Jacobian.
        mujoco.mj_jacSite(self.model, self.data, jac[:3, :], jac[3:, :], site_id)
        jac_arm = jac[:, self.joint_qpos]

        diag = 1e-4 * np.eye(6) # damping
        # Solve system of equations: J @ dq = error.
        dq = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + diag, error)
        return dq

    def solve_ik_vel(self, vel, ang_vel, ref_frame='world', local=False, damping=1e-4):
        """
        Given desired velocity and angular velocity of a site, solve for the joint velocities given the current state
        Velocities are specified in the frame ref_frame
        Solve v = J(theta)*qvel, using damped IK as above
        """
        # Transform to world frame
        if local:
            vec = np.hstack((ang_vel, vel))
            res = np.empty(6)
            mujoco.mju_transformSpatial(res, vec, 0, np.zeros(3), self.data.site(ref_frame).xpos, self.data.site(ref_frame).xmat)
            v = np.hstack((res[3:], res[:3]))
        else:
            v = np.hstack((vel, ang_vel))

        jac = np.zeros((6, self.model.nv))
        site_id = self.model.site(ref_frame).id
        # Get end-effector site Jacobian.
        mujoco.mj_jacSite(self.model, self.data, jac[:3, :], jac[3:, :], site_id)
        jac_arm = jac[:, self.joint_qpos]

        diag = damping * np.eye(6) # damping
        # Solve system of equations: J @ dq = error.
        qvel = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + diag, v)
        return qvel

    def forward_kinematics(self, qpos, ref_frame='end_effector'):
        """
        Given joint angles, return pos and quat of the reference frame
        """

        self.ik_data.qpos[self.joint_qpos] = qpos

        mujoco.mj_forward(self.model, self.ik_data)

        pos = self.ik_data.site(ref_frame).xpos
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.ik_data.site(ref_frame).xmat)

        return pos, quat
    
    def forward_vel_kinematics(self, qpos, qvel, ref_frame='end_effector', local=False):
        """
        Given joint angles and velocities, return pos, quat, velocity and angular velocity of the named site
        Vel and ang_vel can either be in world frame or local frame
        """

        self.ik_data.qpos[self.joint_qpos] = deepcopy(qpos)
        self.ik_data.qvel[self.joint_qpos] = deepcopy(qvel)

        mujoco.mj_forward(self.model, self.ik_data)

        # Jacobian method for velocities in world frame, v = J*qvel
        # jac = np.zeros((6, self.model.nv))
        # site_id = self.model.site(ref_frame).id
        # mujoco.mj_jacSite(self.model, self.data, jac[:3, :], jac[3:, :], site_id)
        # jac_arm = jac[:, self.joint_qpos]

        # vel = jac_arm @ qvel
        # ang_vel = vel[3:]
        # vel = vel[:3]

        # void mj_objectVelocity(const mjModel* m, const mjData* d, int objtype, int objid, mjtNum res[6], int flg_local);
        ee_rot_vel = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.ik_data, mujoco.mjtObj.mjOBJ_SITE, self.model.site(ref_frame).id, ee_rot_vel, 1 if local else 0)
        ang_vel = ee_rot_vel[:3]
        vel = ee_rot_vel[3:]

        pos = self.ik_data.site(ref_frame).xpos
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.ik_data.site(ref_frame).xmat)

        return pos, quat, vel, ang_vel

    def disable_actuator_group(self, group_id):
        # Set the bitfield
        self.model.opt.disableactuator = 2**group_id
        # 0 is our gripper, here we just want the robot joints
        self.joint_actuators = [x for x in range(self.model.nu) if self.model.actuator(x).group != group_id and self.model.actuator(x).group != 0]

    def get_body_pose(self, body):
        """
        Get the position and orientation of a body e.g. the box
        """
        return self.data.body(body).xpos, self.data.body(body).xquat

    def is_state_valid(self, state):
        """OMPL State Validity Checker using MuJoCo for collisions."""
        # Convert OMPL state to numpy array
        qpos = np.array(state[:self.dof])
        
        # Set the robot's configuration in the MuJoCo simulation
        self.ik_data.qpos[:self.dof] = qpos
        
        # Perform a forward kinematics step to update geometry and check for contacts
        mujoco.mj_forward(self.model, self.ik_data)
        
        # A state is valid if there are no collisions
        return not any(np.isin(self.ik_data.contact.geom.flatten(), np.arange(1, 18)))
        # return self.ik_data.ncon == 0