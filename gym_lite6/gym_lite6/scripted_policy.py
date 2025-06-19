import numpy as np
import mujoco
from gym_lite6 import utils
from copy import deepcopy



# upgraded_planner.py
import numpy as np
import mujoco

# OMPL Imports
from ompl import base as ob
from ompl import geometric as og

class OMPLPlanner:
    """
    A Path Planner that uses OMPL to find collision-free paths.
    """
    def __init__(self, env):
        self.env = env
        # 1. Create the OMPL State Space
        # This defines the robot's configuration space (e.g., 6 joints)
        self.space = ob.RealVectorStateSpace(self.env.unwrapped.dof)

        # 2. Set the bounds for the state space from the model
        bounds = ob.RealVectorBounds(self.env.unwrapped.dof)
        for i in range(self.env.unwrapped.dof):
            bounds.setLow(i, env.unwrapped.bounds[0][i])
            bounds.setHigh(i, env.unwrapped.bounds[1][i])
        self.space.setBounds(bounds)

        # 3. Create SpaceInformation
        # This combines the state space with the validity checker
        self.si = ob.SpaceInformation(self.space)

        # 4. Set the State Validity Checker
        # This is the crucial link between OMPL and MuJoCo
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
    
    def is_state_valid(self, state):
        np_state = np.array([state[i] for i in range(self.env.unwrapped.dof)])
        return self.env.is_state_valid(np_state)

    def plan(self, start_qpos, goal_pos, goal_quat, timeout=5.0):
        """
        Plans a path from a start configuration to a goal pose using RRT-Connect.
        """
        print("Planning with OMPL...")
        # A. Solve for the goal configuration using our IK solver
        goal_qpos = self.env.solve_ik(goal_pos, goal_quat, init=start_qpos)
        if goal_qpos is None:
            print("OMPL planning failed: IK could not find a solution for the goal.")
            return None

        # B. Define the OMPL problem
        pdef = ob.ProblemDefinition(self.si)
        
        # Set start state
        start_state = ob.State(self.space)
        for i in range(self.env.unwrapped.dof):
            start_state[i] = start_qpos[i]

        # Set goal state
        goal_state = ob.State(self.space)
        for i in range(self.env.unwrapped.dof):
            goal_state[i] = goal_qpos[i]
            
        pdef.setStartAndGoalStates(start_state, goal_state)

        # C. Instantiate the OMPL planner (RRT-Connect is a good default)
        planner = og.RRTConnect(self.si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        # D. Solve the problem
        solved = planner.solve(timeout) # Solve for a maximum of `timeout` seconds

        if solved:
            print("OMPL path found!")
            # Get the planned path
            path = pdef.getSolutionPath()
            path.interpolate(int(path.length() * 20)) # Densify the path
            
            # Convert path from OMPL states to a list of numpy arrays
            qpos_path = [np.array([path.getState(i)[j] for j in range(self.env.unwrapped.dof)]) for i in range(path.getStateCount())]
            return qpos_path
        else:
            print("OMPL planning failed: No path could be found.")
            return None


class ScriptedPolicyBase(object):
  def __init__(self, env, ref_name, object_name, l_gripper_name, r_gripper_name, max_vel=0.15) -> None:
    self.env = env
    self.ref_name = ref_name
    self.l_gripper_name = l_gripper_name
    self.r_gripper_name = r_gripper_name
    self.object_name = object_name
    self.max_vel = max_vel # max_vel can be used by the planner to determine path duration/speed

    # The planner is instantiated here!
    self.planner = OMPLPlanner(self.env)
    self.reset()

  def __call__(self, model, data, observation, info):
    raise NotImplementedError

  def reset(self):
    self.trajectory_params = {}
    self.active_path = None
    self.path_step = 0
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
    action = {}

    # Stage 0: Move to a pre-grasp position above the object
    if self.stage == 0:
      # If we don't have a path yet, plan one.
      if self.active_path is None:
        goal_pos = deepcopy(data.geom(self.object_name).xpos)
        goal_pos[2] += 0.10  # Pre-grasp height
        goal_quat = np.array([0, 1, 0, 0]) # Top-down grasp orientation

        # Get current configuration to plan from
        start_qpos = deepcopy(data.qpos[:self.env.unwrapped.dof])

        # Ask the planner for a path
        self.active_path = self.planner.plan(start_qpos, goal_pos, goal_quat)
        self.path_step = 0

      # If a path exists, execute it.
      if self.active_path:
        action['qpos'] = self.active_path[self.path_step]
        self.path_step += 1

        # Open gripper in the second half of the path
        if self.path_step > len(self.active_path) / 2:
            action["gripper"] = -1
        else:
            action["gripper"] = 0

        # Check if we've reached the end of the path for this stage
        if self.path_step >= len(self.active_path):
          self.stage += 1
          self.active_path = None # Clear the path to trigger planning for the next stage
          print(f"Transitioning to stage {self.stage}")
      else:
        # Planning failed, what to do? Maybe stay put.
        print("Stage 0: Planning failed, holding position.")
        action['qpos'] = None
        # action['qpos'] = deepcopy(data.qpos[:self.env.unwrapped.dof])

    # Stage 1: Lower down to the grasp position
    elif self.stage == 1:
      action["gripper"] = -1 # Keep gripper open
      if self.active_path is None:
        # The goal is now closer to the object
        goal_pos = deepcopy(data.geom(self.object_name).xpos)
        goal_pos[2] = model.geom(self.object_name).size[2] * 2 - 0.02
        goal_quat = np.array([0, 1, 0, 0])

        start_qpos = deepcopy(data.qpos[:self.env.unwrapped.dof])
        self.active_path = self.planner.plan(start_qpos, goal_pos, goal_quat)
        self.path_step = 0

      if self.active_path:
        action['qpos'] = self.active_path[self.path_step]
        self.path_step += 1
        if self.path_step >= len(self.active_path):
          self.stage += 1
          self.active_path = None
          print(f"Transitioning to stage {self.stage}")
      else:
        print("Stage 1: Planning failed, holding position.")
        action['qpos'] = deepcopy(data.qpos[:self.env.unwrapped.dof])

    # Stage 2: Close the gripper
    elif self.stage == 2:
      action["gripper"] = 1
      action['qpos'] = self.prev_action['qpos'] # Hold joint positions

      # Simple time-based transition for gripping
      if not hasattr(self, 'grip_start_time'):
          self.grip_start_time = data.time
      if data.time - self.grip_start_time > 0.5: # Grip for 0.5 seconds
          self.stage += 1
          delattr(self, 'grip_start_time')
          print(f"Transitioning to stage {self.stage}")

    # Stage 3: Grasp is complete
    elif self.stage == 3:
      action = self.prev_action
      self.done = True

    self.prev_action = deepcopy(action)
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

      pos_reached = pos_err < 7e-3
      quat_reached = quat_err < 9e-3
      # print(pos_err, quat_err)

      if pos_reached and quat_reached:
        self.stage += 1
        print(f"Transitioning to stage {self.stage}")
      elif data.time > self.trajectory_params[self.stage]["end_time"]:
        action["pos"] = self.trajectory_params[self.stage]["goal_pos"]
        action["quat"] = self.trajectory_params[self.stage]["goal_quat"]
        action["qpos"] = self.env.unwrapped.solve_ik(action["pos"], action["quat"])
      else:
        action["pos"], action["quat"] = self.get_waypoint(data.time, self.trajectory_params[self.stage])
        action["qpos"] = self.env.unwrapped.solve_ik(action["pos"], action["quat"])
    
    # Finished
    if self.stage == 1:
      action = self.prev_action
      self.done = True
    
    self.prev_action = deepcopy(action)
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


