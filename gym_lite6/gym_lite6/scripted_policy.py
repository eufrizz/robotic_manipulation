import numpy as np
import mujoco
from gym_lite6 import utils

class ScriptedPickupPolicy(object):
  """
  Assume static object
  """
  def __init__(self, env, ref_name, object_name, l_gripper_name, r_gripper_name, max_vel=0.15) -> None:
    
    self.env = env
    self.ref_name = ref_name
    self.l_gripper_name = l_gripper_name
    self.r_gripper_name = r_gripper_name
    self.object_name = object_name
    self.max_vel = max_vel

    self.trajectory_params = None
    self.stage = 0
    self.done = False

    self.waypoint_idx = 0
    self.prev_action = None


  def __call__(self, model, data, observation, info):
    """
    Predict
    """
    action = {}

    ref_quat = np.empty(4)
    mujoco.mju_mat2Quat(ref_quat, data.site(self.ref_name).xmat)
    ref_pos = data.site(self.ref_name).xpos

    if self.trajectory_params is None:
        # Above the object
        goal_pos = data.geom(self.object_name).xpos
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
        

        goal_pos = data.geom(self.object_name).xpos + model.geom(self.object_name).size / 2
        goal_pos[2] += model.geom(self.object_name).size[2] + 0.006
        goal_quat = np.array([0, 1, 0, 0])

        T_start = utils.get_tf_matrix(ref_pos, ref_quat)
        T_end = utils.get_tf_matrix(goal_pos, goal_quat)

        # Straight line distance to keep in maximum straight line velocity
        dist = np.linalg.norm(goal_pos - ref_pos)
        # Go for anywhere between 0.3 to 0.966 of max vel
        vel_scale = np.random.rand() * 2/3 + 0.3
        end_time = dist/(vel_scale * self.max_vel)

        self.trajectory_params = {0: {"start_time": data.time, "end_time": data.time + end_time, "T_start": T_start, "T_end": T_end, "goal_pos": goal_pos, "goal_quat": goal_quat}}
        print(self.trajectory_params)
        self.stage = 0
        self.done = False

    # Trajectory generation
    if self.stage == 0:

      # Open gripper in last half second
      if data.time > self.trajectory_params[self.stage]["end_time"] - 0.5:
        action["gripper"] = -1
      else:
        action["gripper"] = 0

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
      
    if self.stage == 1:
      # Lower down around block
      # data.site(self.ref_name)

      action["gripper"] = -1

      if self.stage not in self.trajectory_params:
        goal_pos = self.trajectory_params[0]["goal_pos"]
        # Grip height
        goal_pos[2] = model.geom(self.object_name).size[self.stage] * 0.2
        goal_quat = np.array([0, 1, 0, 0])

        T_start = utils.get_tf_matrix(ref_pos, ref_quat)
        T_end = utils.get_tf_matrix(goal_pos, goal_quat)

        end_time = 1

        self.trajectory_params[self.stage] = {"start_time": data.time, "end_time": data.time + end_time, "T_start": T_start, "T_end": T_end, "goal_pos": goal_pos, "goal_quat": goal_quat}

      pos_err = np.linalg.norm(self.trajectory_params[self.stage]["goal_pos"] - ref_pos)
      res_quat = np.empty(3)
      mujoco.mju_subQuat(res_quat, self.trajectory_params[self.stage]["goal_quat"], ref_quat)
      quat_err = np.linalg.norm(res_quat)

      pos_reached = pos_err < 5e-3
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
        
    
    # Lift
    if self.stage == 3:
      if self.stage not in self.trajectory_params:
        goal_pos = self.trajectory_params[0]["goal_pos"] + np.array([0, 0, 0.25])
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
    if self.stage == 4:
      action = self.prev_action
      self.done = True


    self.prev_action = action
    return action

  
  def reset(self):
    self.trajectory_params = None
    
  def get_waypoint(self, time, trajectory_params):
    T_curr = utils.screw_interp(trajectory_params["T_start"], trajectory_params["T_end"], time - trajectory_params["start_time"], trajectory_params["end_time"] - trajectory_params["start_time"])
    return utils.tf_matrix_to_pose_quat(T_curr)

  