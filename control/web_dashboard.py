"""
web_dashboard.py
================
FastAPI web server for Archimedes Survey remote control.

Features:
  - WebSocket telemetry (GPS, IMU, battery, tidal sensor, arm state)
  - Live MJPEG camera stream (proxied from RPi camera)
  - REST commands: move, arm pose, autonomous mode toggle
  - LoRa fallback status indicator

Run on RPi4:
    pip install fastapi uvicorn[standard] aiohttp
    python control/web_dashboard.py

Then open http://<RPi_IP>:8080 on any browser (phone/PC).
"""

import asyncio
import json
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

# ---------------------------------------------------------------------------
# Optional ROS2 bridge import (graceful fallback for non-ROS environments)
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String, Float32, Bool
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

# ---------------------------------------------------------------------------
# Optional camera stream
# ---------------------------------------------------------------------------
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import pathlib, os

BASE_DIR = pathlib.Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Archimedes Survey Control")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ---------------------------------------------------------------------------
# Shared state (updated by ROS2 subscriber callbacks or mock data)
# ---------------------------------------------------------------------------
_state: dict = {
    "gps":      {"lat": 0.0, "lon": 0.0, "fix": False},
    "imu":      {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "battery":  {"voltage": 12.6, "pct": 100},
    "tidal":    {"water_cm": 0.0, "alert": False},
    "motors":   {"left_rpm": 0.0, "right_rpm": 0.0},
    "arm":      {"j1_deg": 0.0, "j2_deg": 0.0, "j3_deg": 0.0, "pose": "stow"},
    "lora":     {"connected": False, "rssi": -999},
    "auto":     {"enabled": False, "target_lat": 0.0, "target_lon": 0.0, "status": "idle"},
    "ts":       0.0,
}
_ws_clients: list[WebSocket] = []

# ---------------------------------------------------------------------------
# ROS2 bridge (optional)
# ---------------------------------------------------------------------------
_ros_node: Optional[object] = None
_cmd_vel_pub = None
_arm_pub = None
_auto_pub = None

def _ros_init():
    global _ros_node, _cmd_vel_pub, _arm_pub, _auto_pub
    if not HAS_ROS2:
        return
    rclpy.init()
    _ros_node = rclpy.create_node("web_dashboard")

    _cmd_vel_pub = _ros_node.create_publisher(Twist, "/cmd_vel", 10)
    _arm_pub     = _ros_node.create_publisher(String, "/arm_command", 10)
    _auto_pub    = _ros_node.create_publisher(String, "/auto_command", 10)

    def _on_telemetry(msg):
        try:
            d = json.loads(msg.data)
            _state.update(d)
            _state["ts"] = time.time()
        except Exception:
            pass

    _ros_node.create_subscription(String, "/telemetry", _on_telemetry, 10)

    import threading
    threading.Thread(target=rclpy.spin, args=(_ros_node,), daemon=True).start()

# ---------------------------------------------------------------------------
# WebSocket broadcast
# ---------------------------------------------------------------------------
async def _broadcast(data: dict):
    msg = json.dumps(data)
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)

async def _telemetry_loop():
    """Push state to all WebSocket clients at 5 Hz."""
    while True:
        _state["ts"] = time.time()
        await _broadcast({"type": "state", "data": _state})
        await asyncio.sleep(0.2)

# ---------------------------------------------------------------------------
# Camera MJPEG stream
# ---------------------------------------------------------------------------
def _gen_frames():
    if not HAS_CV2:
        return
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("dashboard.html", {"request": request})

@app.get("/video")
def video_feed():
    if not HAS_CV2:
        return HTMLResponse("<h3>OpenCV not available</h3>", status_code=503)
    return StreamingResponse(
        _gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            raw = await ws.receive_text()
            cmd = json.loads(raw)
            await _handle_command(cmd)
    except WebSocketDisconnect:
        _ws_clients.remove(ws)

async def _handle_command(cmd: dict):
    kind = cmd.get("type")

    if kind == "move":
        # {"type":"move","linear":0.5,"angular":0.0}
        if HAS_ROS2 and _cmd_vel_pub:
            msg = Twist()
            msg.linear.x  = float(cmd.get("linear", 0))
            msg.angular.z = float(cmd.get("angular", 0))
            _cmd_vel_pub.publish(msg)
        else:
            # Mock: update state
            _state["motors"]["left_rpm"]  = cmd.get("linear", 0) * 60
            _state["motors"]["right_rpm"] = cmd.get("linear", 0) * 60

    elif kind == "arm":
        # {"type":"arm","pose":"deploy"} or {"type":"arm","j1":15,"j2":45,"j3":30}
        if HAS_ROS2 and _arm_pub:
            _arm_pub.publish(String(data=json.dumps(cmd)))
        else:
            if "pose" in cmd:
                _state["arm"]["pose"] = cmd["pose"]
            for j in ("j1", "j2", "j3"):
                if j in cmd:
                    _state["arm"][f"{j}_deg"] = cmd[j]

    elif kind == "auto":
        # {"type":"auto","enabled":true,"target_lat":22.5,"target_lon":120.3}
        _state["auto"]["enabled"] = cmd.get("enabled", False)
        if "target_lat" in cmd:
            _state["auto"]["target_lat"] = cmd["target_lat"]
            _state["auto"]["target_lon"] = cmd["target_lon"]
        if HAS_ROS2 and _auto_pub:
            _auto_pub.publish(String(data=json.dumps(cmd)))

    elif kind == "stop":
        if HAS_ROS2 and _cmd_vel_pub:
            _cmd_vel_pub.publish(Twist())
        _state["motors"] = {"left_rpm": 0.0, "right_rpm": 0.0}

# ---------------------------------------------------------------------------
# REST fallback (for LoRa / low-bandwidth links)
# ---------------------------------------------------------------------------
@app.get("/api/state")
def api_state():
    return _state

@app.post("/api/cmd")
async def api_cmd(cmd: dict):
    await _handle_command(cmd)
    return {"ok": True}

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    _ros_init()
    asyncio.create_task(_telemetry_loop())

@app.on_event("shutdown")
def shutdown():
    if HAS_ROS2:
        rclpy.shutdown()

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("web_dashboard:app", host="0.0.0.0", port=8080, reload=False)
