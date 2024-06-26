import mujoco
from xarm.wrapper import XArmAPI
import numpy as np
import time

class ArmControl(object):
    def __init__(self, pos_bounds=None, sim=False, arm_ip="192.168.1.185"):
        model_path = "/media/ssd/eugene/robotic_manipulation/models/lite6_viz.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.qcurr = None
        #target_id = model.body('target').mocapid
        self.target_pos = self.data.body('target').xpos
        self.target_quat = self.data.body('target').xquat
        if not sim:
            self.arm = XArmAPI(arm_ip, is_radian=True)
            self.arm.set_mode(4)
        self.Kp = 20

        self.pos_bounds = pos_bounds
        self.ref_site_id = self.data.site('attachment_site').id
        self.target_site_id = self.data.site('target').id


    def stop(self):
        self.arm.motion_enable(False)

    def start(self):
        self.arm.motion_enable(True)
        time.sleep(1)

    def reset(self):
        self.arm.reset()

    @staticmethod
    def calculate_error(model, data, ref_site_id: int, target_site_id: int):
        """
        Pass in model and data with ref site at current arm position, target site at goal position
        This is probably the best way for testing
        """
        mujoco.mj_kinematics(model, data)
        #mujoco.mj_forward(model, data)
        mujoco.mj_comPos(model, data)
        
        # Pre-assign variables
        error = np.zeros(6)
        #curr_quat = np.empty(4)
        #target_quat = np.empty(4)
        #curr_quat_conj = np.empty(4)
        mat_err = np.empty((3, 3))
        quat_err = np.empty(4)
        dq = np.zeros(model.nv)

        # Pos error
        error[:3] = data.site(target_site_id).xpos - data.site(ref_site_id).xpos
        print(data.site(target_site_id).xpos, data.site(ref_site_id).xpos)
        # Quat error
        #mujoco.mju_mat2Quat(curr_quat, data.site(ref_site_id).xmat)
        #mujoco.mju_mat2Quat(target_quat, data.site(target_site_id).xmat)
        #mujoco.mju_negQuat(curr_quat_conj, curr_quat)
        #mujoco.mju_mulQuat(quat_err, target_quat, curr_quat_conj)
        
        mat_err = data.site(target_site_id).xmat.reshape((3, 3)) @ data.site(ref_site_id).xmat.reshape((3, 3)).T
        mujoco.mju_mat2Quat(quat_err, mat_err.reshape((9, 1)))
        mujoco.mju_quat2Vel(error[3:], quat_err, 1.0)

        return error

    @staticmethod
    def calculate_dq_damped(model, data, ref_site_id, error):
        jac = np.zeros((6, model.nv))
        # Get jacobian from mujoco
        mujoco.mj_jacSite(model, data, jac[:3, :], jac[3:, :], ref_site_id)
        diag = 1e-4 * np.eye(6)
        # Solve system of equations: J @ dq = error.
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
        return dq 

    def step(self):
        """
        Control step, this needs to be looped regularly
        1. Check for errors
        1. Calculate err
        2. Calculate dq
        3. Update vel
        """
        if not self.target_pos:
            return
        if self.arm.has_error:
            print(f"Exiting, error {self.arm.error_code}")
            return self.arm_error_code
        
        if not sim:
            code, state = arm.get_servo_angle()
            self.qcurr = np.array(state[:6])

        self.data.qpos = self.qcurr
        #self.data.body('target') = self.target_pose
        error = self.calculate_error(self.model, self.data, self.ref_site_id, self.target_site_id)
        dq = self.calculate_dq_damped(self.model, self.data, self.ref_site_id, error)
        
        if not sim:
            arm.vc_set_joint_velocity(speeds=list(Kp*dq), duration=0.1)

        return


    def update_target(self, target_pos: np.ndarray, target_quat: np.ndarray):
        """

        """
        assert target_pos.size == 3
        assert target_quat.size == 4
        assert np.isclose(np.linalg.norm(target_quat), 1), "target_quat is unnormalised"

        if self.pos_bounds and \
            (np.any(target_pos < self.pos_bounds[0]) 
            or np.any(target_pos > self.pos_bounds[1])
            ):
            print(f"Target pos {target_pos} out of bounds {self.pos_bounds}, ignoring")
            return

        self.target_pos = target_pos
        self.target_quat = target_quat
        
    def update_pos_bounds(self, pos_bounds):
        """
        Bounds on end effector position
        TODO: check legit
        """
        self.pos_bounds = bounds


