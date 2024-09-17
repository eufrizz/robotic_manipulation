import numpy as np
import mediapy as media
import mujoco
import numpy as np
from gym_lite6.pickup_task import GraspAndLiftTask

scene_path = "models/cube_pickup.xml"
model = mujoco.MjModel.from_xml_path(scene_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 1.2
camera.elevation = -15
camera.azimuth = -130
camera.lookat = (0, 0, 0.3)

task = GraspAndLiftTask('gripper_left_finger', 'gripper_right_finger', 'box', 'floor')

camera.distance = 0.5
camera.elevation = -10
# camera.azimuth = -130
camera.lookat = (0.1, 0.1, 0)

voption = mujoco.MjvOption()
voption.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
voption.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

f = []
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera, voption); f.append(renderer.render())
reward = task.get_reward(model, data)
assert reward == 0, f"Reward is {reward}, contacts: {data.contact}"

# Gripper touching ground near cube
data.qpos = np.array([0.57, 1.04, 1.2, -0.00014, 0.00034, 5.1e-05, -0.0086, 0.0083, 0.3, 0.22, 0.005, 0.65, -0.65, 0.28, -0.28])
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera, voption); f.append(renderer.render())
reward = task.get_reward(model, data)
assert reward == 0, f"Reward is {reward}, contacts: {data.contact}"

# Close to box
data.qpos = np.array([0.75, 0.73, 0.96, -0.00021, 0.00023, -1.4e-06, -0.0095, 0.0086, 0.2, 0.2, 0.01, 1, -1.7e-06, 1.8e-06, -0.0023])
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera, voption); f.append(renderer.render())
reward = task.get_reward(model, data)
assert reward == 1, f"Reward is {reward}, contacts: {data.contact}"

# One gripper touching box
data.qpos = np.array([0.75, 0.86, 0.97, -0.00027, 0.0029, -0.00015, -0.0096, 0.0087, 0.2, 0.2, 0.01, 1, -3.5e-05, -6.1e-05, -0.0039])
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera, voption); f.append(renderer.render())
reward = task.get_reward(model, data)
assert reward == 2, f"Reward is {reward}, contacts: {data.contact}"

# Two grippers touching box but also touching ground
data.qpos = np.array([0.82, 0.981, 1.1, -0.00012, -1.6e-05, 1.6e-05, -0.0066, 0.0085, 0.2, 0.21, 0.005, 0.71, -0.71, 0.052, -0.052])
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera, voption); f.append(renderer.render())
reward = task.get_reward(model, data)
assert reward == 0, f"Reward is {reward}, contacts: {data.contact.geom}"

## Seems to be an issue with feeding qpos in and getting contact with both grippers

# Two grippers touching box and box on ground
data.qpos = np.array([0.63, 1.19, 1.6, -0.0004, 0.39, -5.6e-07, -0.0049, 0.0051, 0.3, 0.21, 0.005, 0.63, -0.63, 0.32, -0.32])
# data.ctrl = np.array([0.63, 1.2, 1.7, 0, 0.39, 0, 1.5])
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera, voption); f.append(renderer.render())
reward = task.get_reward(model, data)
media.show_images(f)
assert reward == 3, f"Reward is {reward}, contacts: {data.contact.geom}"

# data.qpos = np.array([0.82, 1, 1.2, -0.00022, 0.00026, -4.4e-06, -0.0043, 0.0062, 0.22, 0.23, 0.013, 0.69, -0.65, 0.21, -0.23])
# mujoco.mj_forward(model, data)
# renderer.update_scene(data, camera, voption); f.append(renderer.render())
# reward = task.get_reward(model, data)
# assert reward == 4, f"Reward is {reward}, contacts: {data.contact}"

# # Lifted
# data.qpos = np.array([0.82, 0.42, 1.2, -0.00068, 0.00086, -1.4e-06, -0.004, 0.006, 0.27, 0.29, 0.23, 0.82, -0.44, 0.17, -0.32])
# mujoco.mj_forward(model, data)
# camera.lookat = (0.1, 0.1, 0.2)
# renderer.update_scene(data, camera, voption); f.append(renderer.render())
# reward = task.get_reward(model, data)
# assert reward == 4, f"Reward is {reward}, contacts: {data.contact}"

