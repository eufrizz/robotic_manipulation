#!/usr/bin/env python3
import asyncio
import websockets
import json
from datetime import datetime
import numpy as np
import mujoco
from xarm.wrapper import XArmAPI
import time
import collections
import quaternion
from pathlib import Path
import gymnasium as gym
import gym_lite6.env, gym_lite6.scripted_policy, gym_lite6.pickup_task


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

    async def register_client(self, websocket):
        self.clients.add(websocket)
        print(f"Client connected from {websocket.remote_address}")

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
    def __init__(self, ip="192.168.1.185"):
        # model_xml = MENAGERIE_DIR / "ufactory_lite6/lite6_gripper_wide.xml"
        self.env = env = gym.make(
            "UfactoryCubePickup-v0",
            task=None,
            obs_type="pixels_state",
            max_episode_steps=500,
            visualization_width=320,
            visualization_height=240,
            render_fps=30,
            joint_noise_magnitude=0.1
        )
        self.ref_frame = 'end_effector'

        self.arm = XArmAPI(ip, is_radian=True)
        self.arm.reset()
        self.arm.set_mode(mode=0)

        code, self.state = self.arm.get_servo_angle()
        if code:
            print(f"Invalid pos reading, codes: {(code)}")
        self.ee_setpos = self.arm.get_position_aa()[1]
        print(self.state, self.ee_setpos)
        self.max_speed = 1 # mm
    
    def joy_cb(self, joy_msg):
        """
        [x (mm), y (mm), z (mm), ax, ay, az]
        """

        code, self.state = self.arm.get_servo_angle()
        if code:
            print(f"Invalid pos reading, codes: {(code)}")
            return
        qpos = np.array(self.state[:6])
        pos, w_q_b = self.env.unwrapped.forward_kinematics(qpos, self.ref_frame)

        
        dpos_b = np.zeros(3)
        dpos_b[0] += joy_msg.axes[1] * self.max_speed
        dpos_b[1] += joy_msg.axes[0] * self.max_speed
        dpos_b[2] += joy_msg.axes[2] * self.max_speed

        r, p, y = joy_msg.axes[4], joy_msg.axes[5], joy_msg.axes[1]
        dquat_b = quaternion.as_float_array(quaternion.from_euler_angles(r,p,y))

        # Convert to world frame
        b_q_w = np.zeros(4)
        mujoco.mju_negQuat(b_q_w, w_q_b)
        dquat_w = np.zeros(4)
        mujoco.mju_mulQuat(dquat_w, dquat_b, b_q_w)

        dpos_w = np.zeros(3)
        mujoco.mju_rotVecQuat(dpos_w, dpos_b, b_q_w)

        next_pos = pos + dpos_w
        next_quat = np.zeros(4)
        mujoco.mju_mulQuat(next_quat, w_q_b, dquat_w)
        next_qpos = self.env.unwrapped.solve_ik(next_pos, next_quat, init=qpos)
        print(f"{qpos=} {pos=}, {w_q_b=}, {dpos_w=}, {next_pos=}, {next_quat=}")

    

if __name__ == "__main__":
    server = WebSocketJoyServer()
    xc = XarmControl()

    server.register_cb(xc.joy_cb)

    asyncio.run(server.start_server())