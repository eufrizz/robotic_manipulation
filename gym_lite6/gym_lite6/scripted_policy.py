import numpy as np
import mujoco
from gym_lite6 import utils
from copy import deepcopy

class ScriptedPolicyBase(object):
  def __init__(self, env, ref_name, object_name, l_gripper_name, r_gripper_name, max_vel=0.15) -> None:
    
    self.env = env
    self.ref_name = ref_name
    self.l_gripper_name = l_gripper_name
    self.r_gripper_name = r_gripper_name
    self.object_name = object_name
    self.max_vel = max_vel

    self.reset()

  def __call__(self, model, data, observation, info):
    raise NotImplementedError
  
  def reset(self):
    self.trajectory_params = {}
    self.stage = 0
    self.done = False
    self.prev_action = None
    
  def get_waypoint(self, time, trajectory_params):
    T_curr = utils.screw_interp(trajectory_params["T_start"], trajectory_params["T_end"], time - trajectory_params["start_time"], trajectory_params["end_time"] - trajectory_params["start_time"])
    return utils.tf_matrix_to_pose_quat(T_curr)

class GraspPolicy(ScriptedPolicyBase):
  def __init__(self, env, ref_name, object_name, l_gripper_name, r_gripper_name, max_vel=0.15) -> None:
    super().__init__(env, ref_name, object_name, l_gripper_name, r_gripper_name, max_vel)
  
  def __call__(self, model, data, observation, info):
    # Above the object
    action = {}

    ref_quat = np.empty(4)
    mujoco.mju_mat2Quat(ref_quat, data.site(self.ref_name).xmat)
    ref_pos = deepcopy(data.site(self.ref_name).xpos)

    # Trajectory to object
    if self.stage == 0:
      if self.stage not in self.trajectory_params:
        #TODO: deal with object on its side
        # goal_pos[2] += model.geom(self.object_name).size[2]

        # Align x with x or y
        # x_axis = np.array([1, 0, 0])
        # y_axis = np.array([0, 1, 0])
        # min_rotation = 3.14
        # for axis in [[1, 0, 0], [0, 1, 0]]:

        #   vec = np.empty(3)
        #   mujoco.mju_rotVecQuat(vec, axis, data.body(self.object_name).xquat)
        #   angle = np.dot(vec, )
        #   # Determine the direction of rotation (clockwise or counter-clockwise)
        #   cross_product = np.cross(v1_normalized, v2_normalized)
        #   min_rotation = 
        

        goal_pos = deepcopy(data.geom(self.object_name).xpos)
        goal_pos[2] += model.geom(self.object_name).size[2] * 2 + 0.005
        goal_quat = np.array([0, 1, 0, 0])

        T_start = utils.get_tf_matrix(ref_pos, ref_quat)
        T_end = utils.get_tf_matrix(goal_pos, goal_quat)

        # Straight line distance to keep in maximum straight line velocity
        dist = np.linalg.norm(goal_pos - ref_pos)
        # Go for anywhere between 0.5 to 0.966 of max vel
        vel_scale = np.random.rand() * 0.4 + 0.6
        end_time = dist/(vel_scale * self.max_vel)

        self.trajectory_params = {0: {"start_time": data.time, "end_time": data.time + end_time, "T_start": T_start, "T_end": T_end, "goal_pos": goal_pos, "goal_quat": goal_quat}}
        print(self.trajectory_params)
        self.stage = 0
        self.done = False

      # Open gripper in last half second
      if data.time > self.trajectory_params[self.stage]["end_time"] - 0.5:
        action["gripper"] = -1
      else:
        action["gripper"] = 0

      pos_xy_err = np.linalg.norm(self.trajectory_params[self.stage]["goal_pos"][:2] - ref_pos[:2])
      pos_z_err = np.linalg.norm(self.trajectory_params[self.stage]["goal_pos"][2] - ref_pos[2])
      res_quat = np.empty(3)
      mujoco.mju_subQuat(res_quat, self.trajectory_params[self.stage]["goal_quat"], ref_quat)
      quat_err = np.linalg.norm(res_quat)

      # TODO: need gravity compensation for tighter tolerances here, including Z
      pos_reached = pos_xy_err < 3e-3 and pos_z_err < 5e-3
      quat_reached = quat_err < 5e-3
      # print(pos_err, quat_err)

      if pos_reached and quat_reached:
        self.stage += 1
        print(f"Transitioning to stage {self.stage}")
      elif data.time > self.trajectory_params[self.stage]["end_time"]:
        qpos = self.env.unwrapped.solve_ik(self.trajectory_params[self.stage]["goal_pos"], self.trajectory_params[self.stage]["goal_quat"])
        action["qpos"] = qpos
      else:
        pos, quat = self.get_waypoint(data.time, self.trajectory_params[self.stage])
        qpos = self.env.unwrapped.solve_ik(pos, quat)
        action["qpos"] = qpos
      
    if self.stage == 1:
      # Lower down around block
      # data.site(self.ref_name)

      action["gripper"] = -1

      if self.stage not in self.trajectory_params:
        goal_pos = deepcopy(self.trajectory_params[0]["goal_pos"])
        # Grip height
        goal_pos[2] = max(model.geom(self.object_name).size[2] * 2 - 0.02, 0.001)
        goal_quat = np.array([0, 1, 0, 0])

        T_start = utils.get_tf_matrix(ref_pos, ref_quat)
        T_end = utils.get_tf_matrix(goal_pos, goal_quat)

        end_time = 1

        self.trajectory_params[self.stage] = {"start_time": data.time, "end_time": data.time + end_time, "T_start": T_start, "T_end": T_end, "goal_pos": goal_pos, "goal_quat": goal_quat}

      pos_err = np.linalg.norm(self.trajectory_params[self.stage]["goal_pos"] - ref_pos)
      res_quat = np.empty(3)
      mujoco.mju_subQuat(res_quat, self.trajectory_params[self.stage]["goal_quat"], ref_quat)
      quat_err = np.linalg.norm(res_quat)

      pos_reached = pos_err < 3e-3
      quat_reached = quat_err < 5e-3

      if pos_reached:
        self.stage += 1
        print(f"Transitioning to stage {self.stage}")
      elif data.time > self.trajectory_params[self.stage]["end_time"]:
        qpos = self.env.unwrapped.solve_ik(self.trajectory_params[self.stage]["goal_pos"], self.trajectory_params[self.stage]["goal_quat"])
        action["qpos"] = qpos
      else:
        pos, quat = self.get_waypoint(data.time, self.trajectory_params[self.stage])
        qpos = self.env.unwrapped.solve_ik(pos, quat)
        action["qpos"] = qpos
    
    # Grip
    if self.stage == 2:

      action["gripper"] = 1

      if self.stage not in self.trajectory_params:
        self.trajectory_params[self.stage] = {}
      
      l_gripper_touching_box = False
      r_gripper_touching_box = False

      for geom in data.contact.geom:
        if all(np.isin(geom, [model.geom(self.object_name).id, model.geom(self.l_gripper_name).id])):
          l_gripper_touching_box = True
        if all(np.isin(geom, [model.geom(self.object_name).id,  model.geom(self.r_gripper_name).id])):
          r_gripper_touching_box = True

      if l_gripper_touching_box and r_gripper_touching_box and "contact_time" not in self.trajectory_params[self.stage]:
        self.trajectory_params[self.stage]["contact_time"] = data.time
      
      if "contact_time" not in self.trajectory_params[self.stage]:
        # qpos = self.env.unwrapped.solve_ik(pos, quat)
        action["qpos"] = self.prev_action["qpos"]
      # If we've been gripping for a bit of time
      elif data.time > self.trajectory_params[self.stage]["contact_time"] + 0.2:
        self.stage += 1
        print(f"Transitioning to stage {self.stage}")
      else:
        action["qpos"] = self.prev_action["qpos"]
    
    if self.stage == 3:
      action = self.prev_action
      self.done = True
    
    self.prev_action = action
    return action

class LiftPolicy(ScriptedPolicyBase):
  def __init__(self, env, ref_name, object_name, l_gripper_name, r_gripper_name, max_vel=0.15, lift_height=0.25) -> None:
    super().__init__(env, ref_name, object_name, l_gripper_name, r_gripper_name, max_vel)
    self.lift_height = lift_height
  
  def __call__(self, model, data, observation, info):
    action = {}
    ref_quat = np.empty(4)
    mujoco.mju_mat2Quat(ref_quat, data.site(self.ref_name).xmat)
    ref_pos = deepcopy(data.site(self.ref_name).xpos)
    
    # Lift
    if self.stage == 0:
      # Initialise
      if self.stage not in self.trajectory_params:
        goal_pos = ref_pos + np.array([0, 0, self.lift_height])
        goal_quat = np.array([0, 1, 0, 0])

        T_start = utils.get_tf_matrix(ref_pos, ref_quat)
        T_end = utils.get_tf_matrix(goal_pos, goal_quat)

        end_time = 2

        self.trajectory_params[self.stage] = {"start_time": data.time, "end_time": data.time + end_time, "T_start": T_start, "T_end": T_end, "goal_pos": goal_pos, "goal_quat": goal_quat}
        
        # self.trajectory_params[self.stage] = {}
        # self.trajectory_params[self.stage]["goal_pos"] = self.trajectory_params[0]["goal_pos"]
        # self.trajectory_params[self.stage]["goal_pos"][2] = 0.25
        # self.trajectory_params[self.stage]["goal_quat"] = self.trajectory_params[0]["goal_quat"]
      
      action["gripper"] = 1
      
      pos_err = np.linalg.norm(self.trajectory_params[self.stage]["goal_pos"] - ref_pos)
      res_quat = np.empty(3)
      mujoco.mju_subQuat(res_quat, self.trajectory_params[self.stage]["goal_quat"], ref_quat)
      quat_err = np.linalg.norm(res_quat)

      pos_reached = pos_err < 5e-3
      quat_reached = quat_err < 5e-3
      # print(pos_err, quat_err)

      if pos_reached and quat_reached:
        self.stage += 1
        print(f"Transitioning to stage {self.stage}")
      elif data.time > self.trajectory_params[self.stage]["end_time"]:
        qpos = self.env.unwrapped.solve_ik(self.trajectory_params[self.stage]["goal_pos"], self.trajectory_params[self.stage]["goal_quat"])
        action["qpos"] = qpos
      else:
        pos, quat = self.get_waypoint(data.time, self.trajectory_params[self.stage])
        qpos = self.env.unwrapped.solve_ik(pos, quat)
        action["qpos"] = qpos
    
    # Finished
    if self.stage == 1:
      action = self.prev_action
      self.done = True
    
    self.prev_action = action
    return action

class GraspAndLiftPolicy(ScriptedPolicyBase):
  def __init__(self, env, ref_name, object_name, l_gripper_name, r_gripper_name, max_vel=0.15, lift_height=0.25) -> None:
    self.policies = [GraspPolicy(env, ref_name, object_name, l_gripper_name, r_gripper_name, max_vel), LiftPolicy(env, ref_name, object_name, l_gripper_name, r_gripper_name, max_vel, lift_height)]
    self.policy_idx = 0
    super().__init__(env, ref_name, object_name, l_gripper_name, r_gripper_name, max_vel)
  
  def __call__(self, model, data, observation, info):
    if self.policies[self.policy_idx].done and self.policy_idx < len(self.policies)-1:
      self.policy_idx += 1
    out = self.policies[self.policy_idx](model, data, observation, info)
    self.stage = self.policies[self.policy_idx].stage
    self.done = self.policies[-1].done
    return out
  
  def reset(self):
    self.policy_idx = 0
    self.stage = 0
    for policy in self.policies:
      policy.reset()
