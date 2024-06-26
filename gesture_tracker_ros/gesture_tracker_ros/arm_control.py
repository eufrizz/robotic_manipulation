import mujoco
from xarm.wrapper import XArmAPI
import numpy as np
import time

class ArmControl(object):
    def __init__(self, pos_bounds=None, sim=False, arm_ip="192.168.1.185"):
        model_path = "/media/ssd/eugene/robotic_manipulation/models/lite6_viz.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        # self.renderer = mujoco.Renderer(model, height=360, width=480)
        # self.camera = mujoco.MjvCamera()
        # mujoco.mjv_defaultFreeCamera(model, camera)
        # camera.distance = 1.2
        # camera.elevation = -15
        # camera.azimuth = -130
        # camera.lookat = (0, 0, 0.3)
        # self.renderer = 
        self.qcurr = np.zeros(self.model.nv)
        self.mocapid = self.model.body('target').mocapid
        self.target_pos = None
        self.target_quat = None
        self.sim = sim
        if not sim:
            self.arm = XArmAPI(arm_ip, is_radian=True)
            self.arm.set_mode(4)
            time.sleep(3)
        self.Kp = 5

        self.pos_bounds = pos_bounds
        self.ref_site_id = self.data.site('attachment_site').id
        self.target_site_id = self.data.site('target').id

        self.run = False


    def stop(self):
        self.arm.motion_enable(False)
        self.run = False

    def start(self):
        self.arm.set_state(state=0)
        time.sleep(1)
        self.arm.motion_enable(True)
        time.sleep(1)
        self.run = True

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
        mat_err = np.empty((3, 3))
        quat_err = np.empty(4)

        # Pos error
        error[:3] = data.site(target_site_id).xpos - data.site(ref_site_id).xpos
        # Quat error
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
        if not self.run or self.target_pos is None or self.target_quat is None:
            return
        if self.arm.has_error:
            #print(f"Exiting, error {self.arm.error_code}")
            return self.arm.error_code
        
        if not self.sim:
            code, state = self.arm.get_servo_angle()
            self.qcurr = np.array(state[:6])
            self.data.qpos = self.qcurr

        self.data.mocap_pos[self.mocapid] = self.target_pos
        self.data.mocap_quat[self.mocapid] = self.target_quat
        error = self.calculate_error(self.model, self.data, self.ref_site_id, self.target_site_id)
        dq = self.calculate_dq_damped(self.model, self.data, self.ref_site_id, error)
        
        if not self.sim:
            self.arm.vc_set_joint_velocity(speeds=list(self.Kp*dq), duration=0.1)

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


