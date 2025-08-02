# WebSocket Joystick Teleop

Two Python nodes connected by WebSocket for passing joystick data from Mac client to Ubuntu server, similar to ROS sensor_msgs/Joy.

## Setup

### Dependencies
```bash
pip install -r requirements.txt
```

### Usage

1. **Start the server (Ubuntu side):**
```bash
python websocket_server.py
```

2. **Connect joystick client (Mac side):**
```bash
# Connect to localhost
python websocket_client.py

# Connect to remote server
python websocket_client.py <server_ip>
```

## Message Format

The Joy message format matches ROS sensor_msgs/Joy:
```json
{
  "header": {
    "stamp": "2025-08-02T...",
    "frame_id": "joy"
  },
  "axes": [0.0, 0.0, ...],
  "buttons": [0, 1, 0, ...]
}
```

## Files

- `websocket_server.py` - Server node for Ubuntu (receives joystick data)
- `websocket_client.py` - Client node for Mac (reads joystick, sends data)
- `requirements.txt` - Python dependencies