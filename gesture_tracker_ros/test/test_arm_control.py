
from gesture_tracker_ros.arm_control import ArmControl
import mujoco
import numpy as np

def test_constructor():
    ac = ArmControl(sim=True)

def test_calculate_error_damped():
    model_path = "../models/lite6_viz.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    ref_site_id = data.site('attachment_site').id
    target_site_id = data.site('target').id
    mocapid = model.body('target').mocapid

    # There's no way to directly set the pose of a body in mujoco
    # Either it has to be set via mocap id like target, or you set its joint angles via qpos

    #mujoco.mju_quat2Mat(mat, )
    data.mocap_pos[mocapid] = np.array([0.3, -0.2, 0.3])
    data.mocap_quat[mocapid] = np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2, 0])
    data.qpos = np.array([-0.699893017984,  0.508651457421,  1.207171344449, -0.832813594386, -1.056524098634,
       -2.645751159127])
    #mujoco.mj_kinematics(model, data)
    #assert np.allclose(data.site('target').xpos, data.site('attachment_site').xpos, atol=1e-6)
    # sites only have xmat, so convert from xquat

    error = ArmControl.calculate_error(model, data, ref_site_id, target_site_id)
    assert np.allclose(error, 0, atol=1e-6), f"error: {error}"

    dq = ArmControl.calculate_dq_damped(model, data, ref_site_id, error)
    assert np.allclose(dq, 0, atol=1e-6), f"dq: {dq}"
    
    # Next we check the pos error has the right sign
    data.mocap_pos[mocapid] = np.array([0.31, -0.21, 0.31])
    error = ArmControl.calculate_error(model, data, ref_site_id, target_site_id)
    assert np.isclose(error[0], 0.01, atol=1e-6)
    assert np.isclose(error[1], -0.01, atol=1e-6)
    assert np.isclose(error[2], 0.01, atol=1e-6)

    rot_quat = np.empty(4)
    # This is a 10 degree rotation around x
    mujoco.mju_mulQuat(rot_quat, np.array([0.996, 0.087, 0, 0]), data.mocap_quat[mocapid].T)
    rad = 10 * np.pi / 180
    data.mocap_quat[mocapid] = rot_quat
    error = ArmControl.calculate_error(model, data, ref_site_id, target_site_id)
    assert np.isclose(error[3], rad, atol=1e-3), f"{error[3:]}"





