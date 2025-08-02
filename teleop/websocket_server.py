#!/usr/bin/env python3
import asyncio
import websockets
import json
from datetime import datetime

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

    async def register_client(self, websocket):
        self.clients.add(websocket)
        print(f"Client connected from {websocket.remote_address}")

    async def unregister_client(self, websocket):
        self.clients.remove(websocket)
        print(f"Client disconnected from {websocket.remote_address}")

    async def handle_message(self, websocket, path):
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
                except Exception as e:
                    print(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

    async def process_joy_message(self, joy_msg):
        # Override this method to handle joystick messages
        # For example, convert to ROS messages or control robots
        pass

    async def start_server(self):
        print(f"Starting WebSocket server on {self.host}:{self.port}")
        server = await websockets.serve(
            self.handle_message,
            self.host,
            self.port
        )
        print("WebSocket server started. Waiting for connections...")
        await server.wait_closed()

if __name__ == "__main__":
    server = WebSocketJoyServer()
    asyncio.run(server.start_server())