#!/usr/bin/env python3
import asyncio
import websockets
import json
from datetime import datetime
import sys
import threading

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
        
        # Keyboard state for pynput
        self.keys_pressed = set()
        
        # Initialize pygame if available
        if PYGAME_AVAILABLE:
            pygame.init()
            pygame.joystick.init()

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
        print("  Arrow keys - Right stick (axes 2,3)")
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
        if PYGAME_AVAILABLE:
            pygame.event.pump()
            
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
        joy_msg.axes = [0.0, 0.0, 0.0, 0.0]
        
        # Left stick (WASD)
        if 'a' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[0] = -1.0
        if 'd' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[0] = 1.0
            
        if 'w' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[1] = -1.0
        if 's' in [getattr(k, 'char', None) for k in self.keys_pressed]:
            joy_msg.axes[1] = 1.0
        
        # Right stick (Arrow keys)
        if keyboard.Key.left in self.keys_pressed:
            joy_msg.axes[2] = -1.0
        if keyboard.Key.right in self.keys_pressed:
            joy_msg.axes[2] = 1.0
            
        if keyboard.Key.up in self.keys_pressed:
            joy_msg.axes[3] = -1.0
        if keyboard.Key.down in self.keys_pressed:
            joy_msg.axes[3] = 1.0
        
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
            joy_msg.axes[0] = -1.0
        elif keys[pygame.K_d]:
            joy_msg.axes[0] = 1.0
            
        if keys[pygame.K_w]:
            joy_msg.axes[1] = -1.0
        elif keys[pygame.K_s]:
            joy_msg.axes[1] = 1.0
        
        # Right stick (Arrow keys)
        if keys[pygame.K_LEFT]:
            joy_msg.axes[2] = -1.0
        elif keys[pygame.K_RIGHT]:
            joy_msg.axes[2] = 1.0
            
        if keys[pygame.K_UP]:
            joy_msg.axes[3] = -1.0
        elif keys[pygame.K_DOWN]:
            joy_msg.axes[3] = 1.0
        
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
                if any(joy_msg.buttons) or any(abs(axis) > 0.1 for axis in joy_msg.axes):
                    if self.use_pynput:
                        input_type = "Keyboard (pynput)"
                    elif self.use_keyboard:
                        input_type = "Keyboard (pygame)"
                    else:
                        input_type = "Joystick"
                    print(f"{input_type} - Axes: {joy_msg.axes[:4]} Buttons: {joy_msg.buttons[:2]}")
                
                await asyncio.sleep(0.1)  # 10 Hz
                
        except asyncio.CancelledError:
            print("Input data transmission cancelled")
        except Exception as e:
            print(f"Error in input transmission: {e}")

    async def connect_and_run(self):
        try:
            print(f"Connecting to server at {self.server_uri}")
            async with websockets.connect(self.server_uri) as websocket:
                print("Connected to server!")
                await self.send_joystick_data(websocket)
        except websockets.exceptions.ConnectionRefused:
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

if __name__ == "__main__":
    server_uri = 'ws://localhost:8765'
    if len(sys.argv) > 1:
        server_uri = f'ws://{sys.argv[1]}:8765'
    
    print(f"Joystick WebSocket Client")
    print(f"Server URI: {server_uri}")
    print("Make sure your joystick is connected before running this script")
    
    client = WebSocketJoyClient(server_uri)
    client.run()