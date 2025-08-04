#!/usr/bin/env python3
import asyncio
import websockets
import json
from datetime import datetime
import numpy as np
import mujoco
import time
import collections
import quaternion
from pathlib import Path
import gymnasium as gym
import gym_lite6.env


MENAGERIE_DIR = Path(__file__).parent.parent.resolve() / "mujoco_menagerie"  # note: absolute path

class JoyMessage:
    def __init__(self):
        self.header = {
            'stamp': datetime.now().isoformat(),
            'frame_id': 'joy'
        }
        self.axes = []
        self.buttons = []

    def from_dict(self, data):
        self.header = data.get('header', self.header)
        self.axes = data.get('axes', [])
        self.buttons = data.get('buttons', [])
        return self

    def to_dict(self):
        return {
            'header': self.header,
            'axes': self.axes,
            'buttons': self.buttons
        }

class WebSocketJoyServer:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.callbacks = []
        self.xarm_control = None

    async def register_client(self, websocket):
        self.clients.add(websocket)
        print(f"Client connected from {websocket.remote_address}")
        
        # Send initial state if available
        if self.xarm_control:
            state = self.xarm_control.get_current_state()
            if state:
                await self.send_to_client(websocket, state)

    async def unregister_client(self, websocket):
        self.clients.remove(websocket)
        print(f"Client disconnected from {websocket.remote_address}")

    async def handle_message(self, websocket):
        await self.register_client(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    joy_msg = JoyMessage().from_dict(data)
                    
                    print(f"Received Joy message:")
                    print(f"  Axes: {joy_msg.axes}")
                    print(f"  Buttons: {joy_msg.buttons}")
                    print(f"  Timestamp: {joy_msg.header['stamp']}")
                    
                    # Here you can process the joystick data
                    # For example, call a callback function or publish to ROS
                    await self.process_joy_message(joy_msg)
                    
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {message}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

    async def process_joy_message(self, joy_msg):
        # Override this method to handle joystick messages
        # For example, convert to ROS messages or control robots
        for cb in self.callbacks:
            cb(joy_msg)
    
    async def send_to_client(self, websocket, data):
        """Send data back to a specific client"""
        try:
            await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected while sending data")
    
    async def broadcast_to_clients(self, data):
        """Send data to all connected clients"""
        if self.clients:
            await asyncio.gather(
                *[self.send_to_client(client, data) for client in self.clients],
                return_exceptions=True
            )

    async def start_server(self):
        print(f"Starting WebSocket server on {self.host}:{self.port}")
        server = await websockets.serve(
            self.handle_message,
            self.host,
            self.port
        )
        print("WebSocket server started. Waiting for connections...")
        await server.wait_closed()
    
    def register_cb(self, fn):
        self.callbacks.append(fn)
    
    def set_xarm_control(self, xarm_control):
        """Set the XarmControl instance for state broadcasting"""
        self.xarm_control = xarm_control
    
    async def start_state_broadcast(self, rate_hz=10):
        """Start broadcasting Xarm state at regular intervals"""
        while True:
            await asyncio.sleep(1.0 / rate_hz)
            if self.xarm_control and self.clients:
                state = self.xarm_control.get_current_state()
                if state:
                    await self.broadcast_to_clients(state)


def quaternion_to_axis_angle(quat):
    """
    Converts a quaternion to axis-angle representation.

    Args:
        quat: A 4-element quaternion (w, x, y, z) or np.quaternion.

    Returns:
        A tuple containing the axis (a 3-element vector) and the angle (in radians).
    """

    # Extract the scalar and vector components of the quaternion
    if isinstance(quat, np.quaternion):
        w = quat.w
        x = quat.x
        y = quat.y
        z = quat.z
    else:
        w = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]

    # Calculate the angle
    angle = 2 * np.arccos(w)

    # Calculate the axis
    if angle == 0:
        axis = np.array([0, 0, 0])  # Zero vector if angle is zero
    else:
        axis = np.array([x, y, z]) * angle / np.linalg.norm(np.array([x, y, z]))

    return axis

def mujoco_to_xarm_pose(pos, quat):
    """
    Convert from position (in m) and quaternion to xarm axis-angle pose [x (mm), y (mm), z (mm), ax, ay, az]
    """
    pos_mm = np.array(pos)*1e3
    aang = quaternion_to_axis_angle(quat)
    return list(pos_mm) + list(aang)

def xarm_to_mujoco_pose(pos_aang):
    """
    Convert from xarm axis-angle pose [x (mm), y (mm), z (mm), ax, ay, az] to position (in m) and quaternion
    """
    angle = np.linalg.norm(pos_aang[3:])
    axis = pos_aang[3:]/angle

    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)

    pos = np.array(pos_aang)[:3]/1000.

    return pos, quat


class XarmControl:
    def __init__(self, ip="192.168.1.185", sim_mode=False):
        # model_xml = MENAGERIE_DIR / "ufactory_lite6/lite6_gripper_wide.xml"
        class DummyTask:
            max_reward = 1
            def __init__(self):
                pass
            def get_reward(*args):
                return 0

        self.env = env = gym.make(
            "UfactoryCubePickup-v0",
            task=DummyTask(),
            obs_type="pixels_state",
            max_episode_steps=500,
            visualization_width=320,
            visualization_height=240,
            render_fps=30,
            joint_noise_magnitude=0.0
        )
        self.sim_mode=sim_mode
        
        self.ref_frame = 'end_effector'

        if self.sim_mode:
            observation, info = self.env.reset()
            self.state = observation['state']['qpos']
            self.ee_setpos = observation['ee_pose']['pos']
            self.ee_setquat = observation['ee_pose']['quat']
        else:
            from xarm.wrapper import XArmAPI
            self.arm = XArmAPI(ip, is_radian=True)
            self.arm.reset()
            self.arm.set_mode(mode=0)
            code, self.state = self.arm.get_servo_angle()
            if code:
                print(f"Invalid pos reading, codes: {(code)}")
            self.ee_setpos = self.arm.get_position_aa()[1]
            # TODO
            self.ee_setquat = observation['ee_pose']['quat']
        print(f"Starting pos: {self.state}, {self.ee_setpos}, {self.ee_setquat}")
        self.max_speed = 0.01 # mm
        self.max_rot = 0.1 # mm
    
    def get_current_state(self):
        """Get current Xarm state including joint angles and end effector pose"""
        try:
            if not self.sim_mode:
                # Get joint angles
                code, joint_angles = self.arm.get_servo_angle()
                if code:
                    print(f"Failed to get joint angles, code: {code}")
                    return None
                
                # Get end effector position and orientation
                code, ee_pose = self.arm.get_position_aa()
                if code:
                    print(f"Failed to get end effector pose, code: {code}")
                    return None
            else:
                joint_angles = self.env.unwrapped.data.qpos[:6]
                ee_pose = None
            
            # Get forward kinematics using the environment
            qpos = np.array(joint_angles[:6]) if len(joint_angles) >= 6 else np.array(joint_angles)
            pos, w_q_b = self.env.unwrapped.forward_kinematics(qpos, self.ref_frame)
            
            # Convert all numpy arrays and numpy types to regular Python types
            joint_angles_list = joint_angles.tolist() if hasattr(joint_angles, 'tolist') else list(joint_angles) if joint_angles else []
            ee_pose_list = ee_pose.tolist() if hasattr(ee_pose, 'tolist') else list(ee_pose) if ee_pose else []
            
            state_data = {
                'type': 'xarm_state',
                'timestamp': datetime.now().isoformat(),
                'joint_angles': joint_angles_list,
                'end_effector_pose': ee_pose_list,
                'computed_position': pos.tolist(),
                'computed_quaternion': w_q_b.tolist()
            }
            
            return state_data
            
        except Exception as e:
            print(f"Error getting Xarm state: {e}")
            return None
    
    def joy_cb(self, joy_msg):
        """
        [x (mm), y (mm), z (mm), ax, ay, az]
        """

        if self.sim_mode:
            self.state = self.env.unwrapped.data.qpos[:6]
        else:
            code, self.state = self.arm.get_servo_angle()
            if code:
                print(f"Invalid pos reading, codes: {(code)}")
                return
        
        # qpos = np.array(self.state[:6])
        # pos, w_q_b = self.env.unwrapped.forward_kinematics(qpos, self.ref_frame)
        b_q_w = self.ee_setquat
        w_q_b = np.zeros(4)
        mujoco.mju_negQuat(w_q_b, b_q_w)

        dpos_b = np.zeros(3)
        dpos_b[0] += joy_msg.axes[1] * self.max_speed
        dpos_b[1] += joy_msg.axes[0] * self.max_speed
        dpos_b[2] += joy_msg.axes[2] * self.max_speed

        r, p, y = joy_msg.axes[5], joy_msg.axes[4], joy_msg.axes[3]
        dquat_b = quaternion.as_float_array(quaternion.from_euler_angles(y*self.max_rot, p*self.max_rot,r*self.max_rot,))

        # Convert to world frame
        # Find diff in world frame w_Rd
        # diff b_Rd = b_R2.inv(b_R1)
        # w_Rd = w_Rb.R1.inv(w_Rb.R2)
        # w_Rd = w_Rb.b_Rd.inv(w_Rb)
        dquat_w = np.zeros(4)
        mujoco.mju_mulQuat(dquat_w, b_q_w, dquat_b)
        mujoco.mju_mulQuat(dquat_w, dquat_w, w_q_b)

        dpos_w = np.zeros(3)
        mujoco.mju_rotVecQuat(dpos_w, dpos_b, w_q_b)

        next_pos = self.ee_setpos + dpos_w
        next_quat = np.zeros(4)
        mujoco.mju_mulQuat(next_quat, b_q_w, dquat_w)
        next_pos = np.clip(next_pos, [-1, -1, 0], [1, 1, 1])

        next_qpos = self.env.unwrapped.solve_ik(next_pos, next_quat, init=self.state)
        # print(f"{self.state=} {self.ee_setpos=}, {w_q_b=}, {dpos_w=}, {next_pos=}, {dquat_w=}, {next_quat=}, {next_qpos=}")
        # print(f"{w_q_b=} {w_q_b=}, {dquat_b=}, {dquat_w=}")

        self.ee_setquat = next_quat / np.linalg.norm(next_quat)
        self.ee_setpos = next_pos


        print(f"{self.ee_setpos=}, {self.ee_setquat=}")

        if self.sim_mode:
            self.state = next_qpos
            self.env.step({"qpos": next_qpos, "gripper": 0})

    

async def main():
    server = WebSocketJoyServer()
    xc = XarmControl(sim_mode=True)

    server.register_cb(xc.joy_cb)
    server.set_xarm_control(xc)

    # Start both the server and state broadcast
    await asyncio.gather(
        server.start_server(),
        server.start_state_broadcast(rate_hz=10)
    )

if __name__ == "__main__":
    asyncio.run(main())