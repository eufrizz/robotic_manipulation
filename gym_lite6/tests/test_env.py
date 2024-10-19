import gymnasium as gym
import numpy as np
from gym_lite6 import pickup_task
import pytest
import mujoco
from copy import deepcopy


class TestLite6Env:
    @pytest.fixture(params=[{"action_type":"qpos"}, {"action_type":"qvel"}])
    def env(self, request):
        task = pickup_task.GraspAndLiftTask('gripper_left_finger', 'gripper_right_finger', 'box', 'floor')
        env = gym.make(
                "UfactoryCubePickup-v0",
                task=task,
                obs_type="pixels_state",
                action_type=request.param["action_type"],
                max_episode_steps=300,
            )
        # assert False, "Making" + request.param["action_type"]
        return env

    def test_reset(self, env):
        observation, info = env.reset()
        print("done")

    def test_reset_to_state(self, env):
        qpos0 = np.array([0, 0.541, 1.49 , 2.961, 0.596, 0.203])
        box_pos0 = np.array([0.2, 0, 0.0])
        box_quat0 = None
        observation, info = env.reset(qpos=qpos0, box_pos=box_pos0)

        assert np.allclose(observation['state']['qpos'], qpos0)
        assert np.allclose(observation['state']['qvel'], np.zeros(6))
        assert observation['state']['gripper'] == 0
        # assert np.allclose(observation['state']['box'], state[7:])

    def test_force_to_gripper_action(self, env):
        assert env.unwrapped.force_to_gripper_action(-0.1) == -1
        assert env.unwrapped.force_to_gripper_action(0.) == 0
        assert env.unwrapped.force_to_gripper_action(0.1) == 1

    # def test_step(self):
    #     observation, reward, terminated, truncated, info = 

    def test_forward_kinematics(self, env):
        pos, quat = env.unwrapped.forward_kinematics(np.array([1.546, 0.541, 1.49, 2.961, 0.596, 0.203]), ref_frame='end_effector')
        assert np.allclose(np.array([-0.00253517,  0.48178708,  0.34847367]), pos)
        assert np.allclose(np.array([0.45795918, -0.51325966, 0.50316363, 0.5231293]), quat)

        pos, quat = env.unwrapped.forward_kinematics(np.array([0.0, 0, 0, 0, 0, 0]), ref_frame='end_effector')
        assert np.allclose(np.array([8.69984251e-02, -2.11781530e-06, 7.24896309e-02]), pos)
        assert np.allclose(np.array([1.55188080e-06, 1.00000000e+00, 2.57105653e-12, -2.12132119e-06]), quat)

    def test_forward_vel_kinematics(self, env):
        pos, quat, vel, ang_vel = env.unwrapped.forward_vel_kinematics(qpos=np.array([1.546, 0.541, 1.49, 2.961, 0.596, 0.203]), qvel=np.zeros(6), ref_frame='end_effector')
        assert np.allclose(np.array([-0.00253517,  0.48178708,  0.34847367]), pos)
        assert np.allclose(np.array([0.45795918, -0.51325966, 0.50316363, 0.5231293]), quat)
        assert np.allclose(np.zeros(3), vel)
        assert np.allclose(np.zeros(3), ang_vel)

        # Spin the base link for easy calculation
        pos, quat, vel, ang_vel = env.unwrapped.forward_vel_kinematics(qpos=np.array([0.0,0,0,0,0,0]), qvel=np.array([1.0, 0, 0, 0, 0, 0]), ref_frame='end_effector')
        assert np.allclose(np.array([8.69984251e-02, -2.11781530e-06, 7.24896309e-02]), pos)
        assert np.allclose(np.array([1.55188080e-06, 1.00000000e+00, 2.57105653e-12, -2.12132119e-06]), quat)
        
        ee_rot_vel = np.empty(6)
        mujoco.mj_objectVelocity(env.unwrapped.model, env.unwrapped.ik_data, mujoco.mjtObj.mjOBJ_SITE, env.unwrapped.model.site('end_effector').id, ee_rot_vel, 0)
        ang_vel_expected = ee_rot_vel[:3]
        vel_expected = ee_rot_vel[3:]

        assert not np.allclose(np.zeros_like(ang_vel), ang_vel), ang_vel # Make rotations are not all zero, sanity check
        assert np.allclose(ang_vel_expected, ang_vel), ang_vel
        assert np.allclose(vel_expected, vel), vel
    
    def test_solve_ik_vel(self, env):
        if env.unwrapped.action_type == "qvel":
            qpos0 = np.array([0.0, 0, 0, 0, 0, 0])
            observation, info = env.reset(qpos=qpos0)
            ee_pos0 = deepcopy(observation["ee_pose"]["pos"])
            ee_quat0 = deepcopy(observation["ee_pose"]["quat"])

            # World frame vel and ang_vel
            vel = np.array([1, -1, 0.5])
            ang_vel = np.array([0, 1, -1]) # Make sure to choose rotations that are feasible for the arm
            qvel = env.unwrapped.solve_ik_vel(vel, ang_vel, ref_frame='end_effector', local=False)
            pos_expected, quat_expected, vel_expected, ang_vel_expected = env.unwrapped.forward_vel_kinematics(qpos0, qvel, ref_frame='end_effector', local=False)

            # assert np.all(np.sign(observation["ee_pose"]["vel"]) == np.array([1, 1, -1]))
            assert np.allclose(vel, vel_expected, atol=2e-2), qvel
            assert np.allclose(ang_vel, ang_vel_expected, atol=2e-2), qvel


            # End effector frame
            # Make sure these movements are large enough to be seen in the output, because the IK solution is not exact, and the robot dynamics are involved
            vels = [[1, -1, 0.5]]
            ang_vels = [[-1, 1, 1]]
            for vel, ang_vel in zip(vels, ang_vels):
                qvel = env.unwrapped.solve_ik_vel(vel, ang_vel, ref_frame='end_effector', local=True)

                action = {"qvel": qvel, "gripper": 0}
                observation, reward, terminated, truncated, info = env.step(action)
                # Due to the dynamics of how the model responds to velocity control, we just check that it's moving in the right directions, not the exact value
                assert np.all(np.sign(observation["ee_pose"]["vel"]) == np.sign(vel)), f'{observation["ee_pose"]["vel"]}, {vel}'
                assert np.all(np.sign(observation["ee_pose"]["ang_vel"]) == np.sign(ang_vel)), f'{observation["ee_pose"]["ang_vel"]}, {ang_vel}'

        


# def qpos_action(action_type):
#     return Lite6EnvTest(action_type=action_type)