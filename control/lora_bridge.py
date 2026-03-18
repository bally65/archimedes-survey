"""
lora_bridge.py
==============
LoRa 433MHz backup communication bridge for Archimedes Survey.

Hardware: Ra-02 (SX1278) on SPI → RPi4
Library: pip install pyLoRa  (or use rpi-lora via spidev)

Protocol (text, 9600 baud equivalent, max 64 bytes/packet):
  Uplink   (Ground → Robot):  CMD:<cmd_id>:<payload>\n
  Downlink (Robot → Ground):  TEL:<lat>:<lon>:<batt>:<tidal>:<mode>\n

Commands:
  CMD:S:0.5:0.0   -- move linear=0.5 angular=0.0
  CMD:X           -- emergency stop
  CMD:A:stow      -- arm pose
  CMD:R           -- RTH
  CMD:G:lat:lon   -- goto waypoint

Run on RPi:
    python control/lora_bridge.py --role robot
Run on ground station (laptop with USB LoRa dongle):
    python control/lora_bridge.py --role ground

If LoRa hardware unavailable: uses UDP simulation on localhost for testing.
"""

import argparse
import json
import sys
import threading
import time
import struct

# LoRa hardware (optional)
try:
    from SX127x.LoRa import LoRa as SX127xLoRa
    from SX127x.board_config import BOARD
    HAS_LORA_HW = True
except ImportError:
    HAS_LORA_HW = False

# ROS2 (optional)
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String, Float32
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

import socket

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LORA_FREQ_MHZ  = 433.0
LORA_SF        = 7        # spreading factor (lower = faster)
LORA_BW        = 125e3    # bandwidth
LORA_CR        = 5        # coding rate 4/5
TX_PERIOD_S    = 2.0      # telemetry downlink interval (robot side)
WATCHDOG_S     = 15.0     # no uplink for this long → RTH

# UDP simulation settings (when no LoRa HW)
UDP_ROBOT_PORT   = 5700
UDP_GROUND_PORT  = 5701
UDP_HOST         = "127.0.0.1"


# ---------------------------------------------------------------------------
# UDP simulation transport
# ---------------------------------------------------------------------------
class UDPTransport:
    def __init__(self, role: str):
        self._role = role
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if role == "robot":
            self._sock.bind((UDP_HOST, UDP_ROBOT_PORT))
            self._peer = (UDP_HOST, UDP_GROUND_PORT)
        else:
            self._sock.bind((UDP_HOST, UDP_GROUND_PORT))
            self._peer = (UDP_HOST, UDP_ROBOT_PORT)
        self._sock.settimeout(1.0)
        self._rssi = -70   # simulated

    def send(self, text: str):
        self._sock.sendto(text.encode(), self._peer)

    def recv(self, timeout=1.0) -> tuple[str | None, int]:
        try:
            data, _ = self._sock.recvfrom(256)
            return data.decode().strip(), self._rssi
        except socket.timeout:
            return None, -999

    def rssi(self):
        return self._rssi


# ---------------------------------------------------------------------------
# LoRa hardware transport (SX1278)
# ---------------------------------------------------------------------------
class LoRaTransport(SX127xLoRa if HAS_LORA_HW else object):
    def __init__(self, role: str):
        if not HAS_LORA_HW:
            raise RuntimeError("SX127x library not installed")
        BOARD.setup()
        super().__init__(verbose=False)
        self.set_mode(0x01)   # STDBY
        self.set_freq(LORA_FREQ_MHZ)
        self.set_spreading_factor(LORA_SF)
        self.set_bw(LORA_BW)
        self.set_coding_rate(LORA_CR)
        self.set_pa_config(pa_select=1, max_power=21, output_power=15)
        self._role   = role
        self._rx_buf = []
        self._last_rssi = -999

    def send(self, text: str):
        payload = list(text.encode("ascii"))[:64]
        self.write_payload(payload)
        self.set_mode(0x03)   # TX
        time.sleep(0.5)
        self.set_mode(0x05)   # RX_CONT

    def recv(self, timeout=1.0) -> tuple[str | None, int]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            flags = self.get_irq_flags()
            if flags.get("rx_done"):
                self.clear_irq_flags(rx_done=1)
                payload = self.read_payload(nocheck=True)
                self._last_rssi = self.get_pkt_rssi_value()
                return bytes(payload).decode("ascii", errors="ignore").strip(), self._last_rssi
            time.sleep(0.05)
        return None, -999

    def rssi(self):
        return self._last_rssi


# ---------------------------------------------------------------------------
# Robot-side bridge
# ---------------------------------------------------------------------------
class RobotBridge:
    def __init__(self, transport):
        self._tr    = transport
        self._state = {
            "lat": 0.0, "lon": 0.0, "batt": 100.0,
            "tidal": 0.0, "mode": "idle",
        }
        self._last_rx_t = time.monotonic()
        self._cmd_callback = None   # set by ROS integration

        # ROS2 setup
        if HAS_ROS2:
            rclpy.init()
            self._node = rclpy.create_node("lora_robot_bridge")
            self._pub_cmd = self._node.create_publisher(String, "/lora_cmd", 10)
            self._pub_lora_status = self._node.create_publisher(String, "/lora/status", 10)
            self._node.create_subscription(String, "/telemetry", self._cb_telemetry, 10)
            threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True).start()

    def _cb_telemetry(self, msg):
        try:
            d = json.loads(msg.data)
            gps = d.get("gps", {})
            self._state.update({
                "lat":   gps.get("lat", 0),
                "lon":   gps.get("lon", 0),
                "batt":  d.get("battery", {}).get("pct", 0),
                "tidal": d.get("tidal", {}).get("water_cm", 0),
                "mode":  d.get("auto", {}).get("status", "idle"),
            })
        except Exception:
            pass

    def run(self):
        last_tx = 0.0
        print(f"[Robot LoRa] Running (HW={HAS_LORA_HW})")

        # Start in RX continuous
        if HAS_LORA_HW:
            self._tr.set_mode(0x05)

        while True:
            now = time.monotonic()

            # --- Receive uplink commands ---
            raw, rssi = self._tr.recv(timeout=0.1)
            if raw:
                self._last_rx_t = now
                self._handle_uplink(raw, rssi)

            # --- Watchdog: no uplink → RTH ---
            if now - self._last_rx_t > WATCHDOG_S:
                print("[Robot LoRa] WATCHDOG: no uplink → RTH")
                if HAS_ROS2:
                    self._pub_cmd.publish(String(data='{"type":"auto","mode":"rth"}'))
                self._last_rx_t = now   # reset to avoid flood

            # --- Send downlink telemetry ---
            if now - last_tx >= TX_PERIOD_S:
                s = self._state
                packet = (f"TEL:{s['lat']:.5f}:{s['lon']:.5f}:"
                          f"{s['batt']:.0f}:{s['tidal']:.1f}:{s['mode']}\n")
                self._tr.send(packet)
                last_tx = now

                # Publish LoRa status to ROS
                if HAS_ROS2:
                    status = json.dumps({"connected": True, "rssi": rssi})
                    self._pub_lora_status.publish(String(data=status))

    def _handle_uplink(self, raw: str, rssi: int):
        print(f"[Robot LoRa] RX rssi={rssi}: {raw!r}")
        parts = raw.split(":")
        if len(parts) < 2 or parts[0] != "CMD":
            return
        sub = parts[1]
        if sub == "X":
            cmd = {"type": "stop"}
        elif sub == "S" and len(parts) >= 4:
            cmd = {"type": "move", "linear": float(parts[2]), "angular": float(parts[3])}
        elif sub == "A" and len(parts) >= 3:
            cmd = {"type": "arm", "pose": parts[2]}
        elif sub == "R":
            cmd = {"type": "auto", "mode": "rth", "enabled": True}
        elif sub == "G" and len(parts) >= 4:
            cmd = {"type": "auto", "enabled": True,
                   "target_lat": float(parts[2]), "target_lon": float(parts[3])}
        else:
            return

        if HAS_ROS2:
            self._pub_cmd.publish(String(data=json.dumps(cmd)))


# ---------------------------------------------------------------------------
# Ground-side bridge (laptop with USB dongle)
# ---------------------------------------------------------------------------
class GroundBridge:
    def __init__(self, transport):
        self._tr  = transport
        self._telem = {}

    def run(self):
        print("[Ground LoRa] Listening for robot telemetry...")
        print("Commands: move <v> <w> | stop | arm <pose> | rth | goto <lat> <lon> | quit")

        import select

        while True:
            # Non-blocking input
            ready = select.select([sys.stdin], [], [], 0.05)[0]
            if ready:
                line = sys.stdin.readline().strip()
                if line == "quit":
                    break
                pkt = self._parse_cmd(line)
                if pkt:
                    self._tr.send(pkt)
                    print(f"[Ground] Sent: {pkt!r}")

            # Receive telemetry
            raw, rssi = self._tr.recv(timeout=0.05)
            if raw and raw.startswith("TEL:"):
                parts = raw.split(":")
                if len(parts) >= 6:
                    self._telem = {
                        "lat":   parts[1], "lon":   parts[2],
                        "batt":  parts[3], "tidal": parts[4],
                        "mode":  parts[5], "rssi":  rssi,
                    }
                    print(f"[Robot] lat={parts[1]} lon={parts[2]} "
                          f"batt={parts[3]}% tidal={parts[4]}cm "
                          f"mode={parts[5]} RSSI={rssi}dBm")

    @staticmethod
    def _parse_cmd(line: str) -> str | None:
        parts = line.split()
        if not parts:
            return None
        cmd = parts[0].lower()
        if cmd == "stop":
            return "CMD:X"
        elif cmd == "move" and len(parts) >= 3:
            return f"CMD:S:{parts[1]}:{parts[2]}"
        elif cmd == "arm" and len(parts) >= 2:
            return f"CMD:A:{parts[1]}"
        elif cmd == "rth":
            return "CMD:R"
        elif cmd == "goto" and len(parts) >= 3:
            return f"CMD:G:{parts[1]}:{parts[2]}"
        print(f"Unknown command: {line!r}")
        return None


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LoRa bridge for Archimedes Survey")
    parser.add_argument("--role", choices=["robot", "ground"], default="robot")
    parser.add_argument("--sim",  action="store_true",
                        help="Force UDP simulation (no LoRa hardware)")
    args = parser.parse_args()

    if args.sim or not HAS_LORA_HW:
        print(f"[LoRa] Using UDP simulation ({args.role} mode)")
        transport = UDPTransport(args.role)
    else:
        print(f"[LoRa] Using SX1278 hardware ({args.role} mode)")
        transport = LoRaTransport(args.role)

    if args.role == "robot":
        RobotBridge(transport).run()
    else:
        GroundBridge(transport).run()


if __name__ == "__main__":
    main()
