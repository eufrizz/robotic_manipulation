import gymnasium as gym
import numpy as np
from gym_lite6 import pickup_task
import pytest



# @pytest.mark.parametrize("action_type", [
#     "qpos", "qvel"
# ])


class TestLite6Env:
    @pytest.fixture(params=[{"action_type":"qpos"}, {"action_type":"qvel"}])
    def env(self, request):
        task = pickup_task.PickupTask('gripper_left_finger', 'gripper_right_finger', 'box', 'floor')
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
        state = np.array([ 2.00623691e-01,  6.44001646e-01,  1.80970371e+00,  2.85094406e-05,
            1.16738198e+00,  2.00728000e-01,  0.0,  6.73108789e-04,
            2.81449724e-01, -3.75870166e-01,  4.99996502e-03,  6.07372585e-01,
        -2.45167540e-06, -3.27133563e-07,  7.94417109e-01])
        observation, info = env.reset(state=state)

        assert np.allclose(observation['state']['qpos'], state[:6])
        assert np.allclose(observation['state']['qvel'], np.zeros(6))
        assert observation['state']['gripper'] == state[6]
        # assert np.allclose(observation['state']['box'], state[7:])

    def test_force_to_gripper_action(self, env):
        assert env.unwrapped.force_to_gripper_action(-0.1) == -1
        assert env.unwrapped.force_to_gripper_action(0.) == 0
        assert env.unwrapped.force_to_gripper_action(0.1) == 1

    # def test_step(self):
    #     observation, reward, terminated, truncated, info = 


# def qpos_action(action_type):
#     return Lite6EnvTest(action_type=action_type)