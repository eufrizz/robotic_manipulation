#!/usr/bin/env python3
import asyncio
import websockets
import json
from datetime import datetime
import sys
import threading
import gymnasium as gym
import mujoco 
import gym_lite6.env
import argparse

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not available")

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("pynput not available - install with: pip install pynput")

class JoyMessage:
    def __init__(self):
        self.header = {
            'stamp': datetime.now().isoformat(),
            'frame_id': 'joy'
        }
        self.axes = []
        self.buttons = []

    def to_dict(self):
        return {
            'header': self.header,
            'axes': self.axes,
            'buttons': self.buttons
        }

class WebSocketJoyClient:
    def __init__(self, server_uri='ws://localhost:8765'):
        self.server_uri = server_uri
        self.joystick = None
        self.running = False
        self.use_keyboard = False
        self.use_pynput = False
        self.state_callbacks = []
        
        # Keyboard state for pynput
        self.keys_pressed = set()
        
        # Initialize pygame joystick system independently of display
        if PYGAME_AVAILABLE:
            # Initialize only the joystick subsystem, not the full pygame
            pygame.joystick.init()
            # Also init the event system for joystick events
            pygame.init()

    def initialize_joystick(self):
        if not PYGAME_AVAILABLE:
            return self.initialize_keyboard_only()
            
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            return self.initialize_keyboard_only()
        
        print(f"Found {joystick_count} joystick(s)")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.use_keyboard = False
        
        print(f"Joystick name: {self.joystick.get_name()}")
        print(f"Number of axes: {self.joystick.get_numaxes()}")
        print(f"Number of buttons: {self.joystick.get_numbuttons()}")
        print(f"Number of hats: {self.joystick.get_numhats()}")
        
        return True
    
    def initialize_keyboard_only(self):
        print("No joysticks found! Falling back to keyboard control.")
        print("Keyboard controls:")
        print("  WASD - Left stick (axes 0,1)")
        print("  IJKL - Left stick (axes 2,3)")
        print("  Arrow keys - Right stick (axes 4,5)")
        print("  Space - Button 0")
        print("  Enter - Button 1")
        
        if PYNPUT_AVAILABLE:
            print("Using pynput for keyboard input (works in background)")
            self.use_pynput = True
            self.setup_pynput_listener()
        else:
            print("Using pygame for keyboard input (window focus required)")
            self.use_keyboard = True
            if not PYGAME_AVAILABLE:
                print("ERROR: Neither pygame nor pynput available!")
                return False
        
        return True
    
    def setup_pynput_listener(self):
        def on_press(key):
            self.keys_pressed.add(key)
        
        def on_release(key):
            self.keys_pressed.discard(key)
        
        self.listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.listener.start()

    def read_joystick(self):
        joy_msg = JoyMessage()
        joy_msg.header['stamp'] = datetime.now().isoformat()
        
        if self.use_pynput:
            return self.read_pynput_keyboard(joy_msg)
        elif self.use_keyboard:
            if PYGAME_AVAILABLE:
                pygame.event.pump()
            return self.read_keyboard(joy_msg)
        
        # Read pygame joystick
        if PYGAME_AVAILABLE and self.joystick:
            # Process ALL pygame events to ensure joystick state updates
            # This must be done on the main thread on macOS
            pygame.event.pump()
            events = pygame.event.get()  # Get all events
            
            # We don't need to handle the events, just processing them updates joystick state
            # Put back any non-joystick events for the display system
            for event in events:
                if event.type not in [pygame.JOYAXISMOTION, pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP, pygame.JOYHATMOTION]:
                    pygame.event.post(event)
            
            # Read axes
            for i in range(self.joystick.get_numaxes()):
                axis_value = self.joystick.get_axis(i)
                joy_msg.axes.append(round(axis_value, 4))
            
            # Read buttons
            for i in range(self.joystick.get_numbuttons()):
                button_value = self.joystick.get_button(i)
                joy_msg.buttons.append(int(button_value))
            
            # Read hat (D-pad) if available
            for i in range(self.joystick.get_numhats()):
                hat_value = self.joystick.get_hat(i)
                # Convert hat to additional axes (optional)
                joy_msg.axes.extend([float(hat_value[0]), float(hat_value[1])])
        
        return joy_msg

    def read_pynput_keyboard(self, joy_msg):
        # Initialize axes (4 axes: left stick x,y and right stick x,y)
        joy_msg.axes = [0.0 for _ in range(6)]
        
        # Left stick (WASD)
        if 'a' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[3] = 1.0
        if 'd' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[3] = -1.0
            
        if 'w' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[2] = 1.0
        if 's' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[2] = -1.0

        if 'j' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[5] = -1.0
        if 'l' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[5] = 1.0
            
        if 'i' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[4] = -1.0
        if 'k' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[4] = 1.0
        
        # Right stick (Arrow keys)
        if keyboard.Key.left in self.keys_pressed:
            joy_msg.axes[0] = -1.0
        if keyboard.Key.right in self.keys_pressed:
            joy_msg.axes[0] = 1.0
            
        if keyboard.Key.up in self.keys_pressed:
            joy_msg.axes[1] = 1.0
        if keyboard.Key.down in self.keys_pressed:
            joy_msg.axes[1] = -1.0
        
        # Buttons
        joy_msg.buttons = [
            int(keyboard.Key.space in self.keys_pressed),  # Button 0
            int(keyboard.Key.enter in self.keys_pressed)   # Button 1
        ]
        
        return joy_msg

    def read_keyboard(self, joy_msg):
        if not PYGAME_AVAILABLE:
            return joy_msg
            
        keys = pygame.key.get_pressed()
        
        # Initialize axes (4 axes: left stick x,y and right stick x,y)
        joy_msg.axes = [0.0, 0.0, 0.0, 0.0]
        
        # Left stick (WASD)
        if keys[pygame.K_a]:
            joy_msg.axes[3] = -1.0
        elif keys[pygame.K_d]:
            joy_msg.axes[3] = 1.0
            
        if keys[pygame.K_w]:
            joy_msg.axes[2] = -1.0
        elif keys[pygame.K_s]:
            joy_msg.axes[2] = 1.0
        
        # Right stick (Arrow keys)
        if keys[pygame.K_LEFT]:
            joy_msg.axes[1] = -1.0
        elif keys[pygame.K_RIGHT]:
            joy_msg.axes[1] = 1.0
            
        if keys[pygame.K_UP]:
            joy_msg.axes[0] = -1.0
        elif keys[pygame.K_DOWN]:
            joy_msg.axes[0] = 1.0
        
        # Buttons
        joy_msg.buttons = [
            int(keys[pygame.K_SPACE]),  # Button 0
            int(keys[pygame.K_RETURN])  # Button 1
        ]
        
        return joy_msg

    async def send_joystick_data(self, websocket):
        if self.use_pynput:
            print("Starting keyboard data transmission (pynput - works in background)...")
        elif self.use_keyboard:
            print("Starting keyboard data transmission (pygame - focus window)...")
        else:
            print("Starting joystick data transmission...")
        self.running = True
        
        try:
            while self.running:
                joy_msg = self.read_joystick()
                message = json.dumps(joy_msg.to_dict())
                
                await websocket.send(message)
                
                # Print periodic status
                if any(joy_msg.buttons) or any(abs(axis) > 0.01 for axis in joy_msg.axes):
                    if self.use_pynput:
                        input_type = "Keyboard (pynput)"
                    elif self.use_keyboard:
                        input_type = "Keyboard (pygame)"
                    else:
                        input_type = "Joystick"
                    print(f"{input_type} - Axes: {joy_msg.axes} Buttons: {joy_msg.buttons[:2]}")
                
                await asyncio.sleep(0.08)  # 10 Hz
                
        except asyncio.CancelledError:
            print("Input data transmission cancelled")
        except Exception as e:
            print(f"Error in input transmission: {e}")

    async def receive_messages(self, websocket):
        """Listen for incoming messages from the server"""
        print("Starting message receiver...")
        try:
            while self.running:
                try:
                    # Wait for message with a timeout to avoid blocking indefinitely
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    try:
                        data = json.loads(message)
                        await self.handle_server_message(data)
                    except json.JSONDecodeError:
                        print(f"Received invalid JSON: {message}")
                except asyncio.TimeoutError:
                    # Timeout is normal, just continue the loop
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("Server connection closed")
                    break
        except asyncio.CancelledError:
            print("Message receiving cancelled")
        except Exception as e:
            print(f"Error receiving messages: {e}")
        print("Message receiver ended")

    async def handle_server_message(self, data):
        """Handle different types of messages from the server"""
        message_type = data.get('type', 'unknown')
        if message_type == 'xarm_state':
            for cb in self.state_callbacks:
                cb(data)
        else:
            print(f"Unknown message type: {message_type}")

    def print_xarm_state(self, state_data):
        """Print the received Xarm state data"""
        timestamp = state_data.get('timestamp', 'N/A')
        joint_angles = state_data.get('joint_angles', [])
        ee_pose = state_data.get('end_effector_pose', [])
        computed_pos = state_data.get('computed_position', [])
        computed_quat = state_data.get('computed_quaternion', [])
        
        print(f"\n=== Xarm State ({timestamp}) ===")
        print(f"Joint angles: {[round(x, 3) if x is not None else None for x in joint_angles[:6]]}")
        if ee_pose:
            print(f"EE pose: [{', '.join([f'{x:.3f}' for x in ee_pose[:6]])}]")
        print(f"Computed pos: [{', '.join([f'{x:.3f}' for x in computed_pos])}]")
        print(f"Computed quat: [{', '.join([f'{x:.3f}' for x in computed_quat])}]")

    async def connect_and_run(self):
        try:
            print(f"Connecting to server at {self.server_uri}")
            async with websockets.connect(self.server_uri) as websocket:
                print("Connected to server!")
                print("Listening for Xarm state updates...")
                
                # Run both sending joystick data and receiving messages concurrently
                await asyncio.gather(
                    self.send_joystick_data(websocket),
                    self.receive_messages(websocket),
                    return_exceptions=True
                )
        except ConnectionRefusedError:
            print(f"Could not connect to server at {self.server_uri}")
            print("Make sure the server is running and the address is correct")
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Connection error: {e}")

    def run(self):
        if not self.initialize_joystick():
            return
        
        try:
            print("Press Ctrl+C to stop")
            if self.use_keyboard:
                print("Make sure this terminal window has focus for keyboard input")
            asyncio.run(self.connect_and_run())
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.running = False
            if hasattr(self, 'listener'):
                self.listener.stop()
            if self.joystick and PYGAME_AVAILABLE:
                self.joystick.quit()
            if PYGAME_AVAILABLE:
                pygame.quit()
    
    def registerStateCb(self, fn):
        self.state_callbacks.append(fn)

class XarmViz:
    def __init__(self):
        self.env = gym.make(
            "UfactoryCubePickup-v0",
            task=None,
            obs_type="pixels_state",
            max_episode_steps=500,
            visualization_width=640,
            visualization_height=480,
            render_fps=30,
            joint_noise_magnitude=0.1
        )
        self.env.reset()
        
        # Initialize pygame display separately from joystick system
        if PYGAME_AVAILABLE:
            # pygame.init() was already called by WebSocketJoyClient, so we can just create the display
            self.screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("Xarm Visualization")
            self.clock = pygame.time.Clock()
        else:
            print("Warning: pygame not available, visualization will not display")
            self.screen = None
        
    def update(self, state_data):
        """Update the visualization with new Xarm state"""
        try:
            joint_angles = state_data.get('joint_angles', [])
            if len(joint_angles) >= 6:
                # Update joint positions in the environment
                self.env.unwrapped.data.qpos[:6] = joint_angles[:6]
                
                # Forward dynamics to update the model
                mujoco.mj_forward(self.env.unwrapped.model, self.env.unwrapped.data)
                
                # Render the scene and get the image
                image = self.env.unwrapped.render(camera="gripper_cam")
                
                # Display the image if pygame is available
                if self.screen is not None and image is not None:
                    self.display_image(image)
                
        except Exception as e:
            print(f"Error updating visualization: {e}")
    
    def display_image(self, image):
        """Display the rendered image in pygame window"""
        try:
            # Handle pygame events to keep window responsive
            # Only handle QUIT events, don't consume all events
            events = pygame.event.get([pygame.QUIT])
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Convert numpy array to pygame surface
            # Image is typically in RGB format (height, width, channels)
            if len(image.shape) == 3:
                # Convert from RGB to pygame format
                surf = pygame.surfarray.make_surface(image.swapaxes(0, 1))
                # Scale to fit window if needed
                surf = pygame.transform.scale(surf, (640, 480))
                
                # Blit to screen
                self.screen.blit(surf, (0, 0))
                pygame.display.flip()
                
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.quit()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="localhost")
    parser.add_argument("--viz", action="store_true")

    args = parser.parse_args()
    server_uri = f'ws://{args.ip}:8765'
        
    print(f"Joystick WebSocket Client with Visualization")
    print(f"Server URI: {server_uri}")
    print("Make sure your joystick is connected before running this script")
    
    client = WebSocketJoyClient(server_uri)
    viz = None
    
    try:
        # Initialize visualization
        if args.viz:
            viz = XarmViz()
            client.registerStateCb(viz.update)
        # client.registerStateCb(client.print_xarm_state)
        
        client.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up visualization
        if viz is not None:
            viz.close()
