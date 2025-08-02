#!/usr/bin/env python3
import asyncio
import websockets
import json
import pygame
from datetime import datetime
import sys

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
        
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()

    def initialize_joystick(self):
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("No joysticks found!")
            return False
        
        print(f"Found {joystick_count} joystick(s)")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"Joystick name: {self.joystick.get_name()}")
        print(f"Number of axes: {self.joystick.get_numaxes()}")
        print(f"Number of buttons: {self.joystick.get_numbuttons()}")
        print(f"Number of hats: {self.joystick.get_numhats()}")
        
        return True

    def read_joystick(self):
        pygame.event.pump()
        
        joy_msg = JoyMessage()
        joy_msg.header['stamp'] = datetime.now().isoformat()
        
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

    async def send_joystick_data(self, websocket):
        print("Starting joystick data transmission...")
        self.running = True
        
        try:
            while self.running:
                joy_msg = self.read_joystick()
                message = json.dumps(joy_msg.to_dict())
                
                await websocket.send(message)
                
                # Print periodic status
                if any(joy_msg.buttons) or any(abs(axis) > 0.1 for axis in joy_msg.axes):
                    print(f"Sent - Axes: {joy_msg.axes[:4]}... Buttons: {joy_msg.buttons[:8]}...")
                
                await asyncio.sleep(0.05)  # 20 Hz
                
        except asyncio.CancelledError:
            print("Joystick data transmission cancelled")
        except Exception as e:
            print(f"Error in joystick transmission: {e}")

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
            asyncio.run(self.connect_and_run())
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.running = False
            if self.joystick:
                self.joystick.quit()
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