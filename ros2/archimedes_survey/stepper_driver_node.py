"""
stepper_driver_node.py
======================
ROS2 node: converts /motor/left_pwm + /motor/right_pwm Float32 (-1~+1)
into GPIO step/direction pulses for two TB6600 stepper drivers.

Hardware wiring (RPi4 BCM numbering):
  TB6600 Left (Screw A):
    ENA-  → GND          ENA+ → 3.3V (always enabled)
    DIR-  → GND          DIR+ → GPIO 17
    PUL-  → GND          PUL+ → GPIO 27

  TB6600 Right (Screw B):
    ENA-  → GND          ENA+ → 3.3V
    DIR-  → GND          DIR+ → GPIO 22
    PUL-  → GND          PUL+ → GPIO 23

  TB6600 DIP switches (full step = max torque):
    SW1-SW3: 001  = 1/1 step  (or 110 = 1/4 for smoother at cost of torque)
    SW4-SW6: 011  = 2.0 A (match NEMA23 57BYGH78 rated current)

  NEMA23 57BYGH78 specs:
    Steps/rev: 200   (1.8 deg/step)
    Gear ratio: 10:1
    Effective steps/rev: 2000
    Max RPM (motor): 600 → after gearbox: 60 RPM

Subscribe:
  /motor/left_pwm   (std_msgs/Float32  -1.0 to +1.0)
  /motor/right_pwm  (std_msgs/Float32  -1.0 to +1.0)

Publish:
  (none — directly drives GPIO)

Run:
    ros2 run archimedes_survey stepper_driver
    # Must run as root or with gpio group:
    sudo ros2 run archimedes_survey stepper_driver
"""

import sys
import time
import threading

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32
except ImportError:
    sys.exit("ROS2 not found.")

# GPIO backend: prefer RPi.GPIO, fallback to pigpio, fallback to mock
try:
    import RPi.GPIO as GPIO
    _GPIO_BACKEND = "RPi.GPIO"
except ImportError:
    try:
        import pigpio
        _pi = pigpio.pi()
        _GPIO_BACKEND = "pigpio"
    except ImportError:
        _GPIO_BACKEND = "mock"
        print("[stepper_driver] WARNING: No GPIO library found, running in mock mode")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STEPS_PER_REV   = 200          # NEMA23 native
GEAR_RATIO      = 10           # 10:1 gearbox
EFF_STEPS_REV   = STEPS_PER_REV * GEAR_RATIO   # = 2000
MAX_RPM         = 60.0         # after gearbox
MAX_STEP_HZ     = MAX_RPM / 60.0 * EFF_STEPS_REV   # = 2000 steps/s
MIN_STEP_HZ     = 50.0         # below this, treat as stopped
PULSE_WIDTH_US  = 5            # TB6600 minimum pulse width: 5 us

# GPIO pin numbers (BCM)
PIN_LEFT_DIR  = 17
PIN_LEFT_PUL  = 27
PIN_RIGHT_DIR = 22
PIN_RIGHT_PUL = 23


# ---------------------------------------------------------------------------
# Low-level GPIO helpers
# ---------------------------------------------------------------------------
def _gpio_setup():
    if _GPIO_BACKEND == "RPi.GPIO":
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in (PIN_LEFT_DIR, PIN_LEFT_PUL, PIN_RIGHT_DIR, PIN_RIGHT_PUL):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    elif _GPIO_BACKEND == "pigpio":
        for pin in (PIN_LEFT_DIR, PIN_LEFT_PUL, PIN_RIGHT_DIR, PIN_RIGHT_PUL):
            _pi.set_mode(pin, pigpio.OUTPUT)
            _pi.write(pin, 0)


def _gpio_write(pin: int, val: int):
    if _GPIO_BACKEND == "RPi.GPIO":
        GPIO.output(pin, val)
    elif _GPIO_BACKEND == "pigpio":
        _pi.write(pin, val)
    # mock: no-op


def _gpio_cleanup():
    if _GPIO_BACKEND == "RPi.GPIO":
        GPIO.cleanup()
    elif _GPIO_BACKEND == "pigpio":
        _pi.stop()


# ---------------------------------------------------------------------------
# Stepper channel: runs in its own thread
# ---------------------------------------------------------------------------
class StepperChannel:
    """
    Background thread that sends step pulses at the requested frequency.
    Thread-safe: call set_pwm() from any thread.
    """
    def __init__(self, dir_pin: int, pul_pin: int, name: str):
        self._dir_pin = dir_pin
        self._pul_pin = pul_pin
        self._name    = name

        self._lock    = threading.Lock()
        self._hz      = 0.0      # pulse frequency (steps/s)
        self._forward = True     # True = forward, False = reverse
        self._running = True

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_pwm(self, pwm: float):
        """
        pwm: -1.0 (full reverse) to +1.0 (full forward), 0 = stop.
        """
        hz = abs(pwm) * MAX_STEP_HZ
        fwd = pwm >= 0
        with self._lock:
            self._hz      = hz if hz >= MIN_STEP_HZ else 0.0
            self._forward = fwd

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            with self._lock:
                hz  = self._hz
                fwd = self._forward

            if hz < MIN_STEP_HZ:
                time.sleep(0.01)
                continue

            period_s = 1.0 / hz
            half     = period_s / 2.0
            pw_s     = PULSE_WIDTH_US * 1e-6

            # Set direction
            _gpio_write(self._dir_pin, 1 if fwd else 0)

            # One step pulse
            _gpio_write(self._pul_pin, 1)
            time.sleep(pw_s)
            _gpio_write(self._pul_pin, 0)
            time.sleep(max(0, half - pw_s))


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------
class StepperDriverNode(Node):
    def __init__(self):
        super().__init__("stepper_driver")
        _gpio_setup()

        self._left  = StepperChannel(PIN_LEFT_DIR,  PIN_LEFT_PUL,  "left")
        self._right = StepperChannel(PIN_RIGHT_DIR, PIN_RIGHT_PUL, "right")

        self.create_subscription(Float32, "/motor/left_pwm",  self._cb_left,  10)
        self.create_subscription(Float32, "/motor/right_pwm", self._cb_right, 10)

        self.get_logger().info(
            f"StepperDriverNode ready (GPIO backend: {_GPIO_BACKEND})\n"
            f"  Left:  DIR=GPIO{PIN_LEFT_DIR}  PUL=GPIO{PIN_LEFT_PUL}\n"
            f"  Right: DIR=GPIO{PIN_RIGHT_DIR} PUL=GPIO{PIN_RIGHT_PUL}\n"
            f"  Max: {MAX_STEP_HZ:.0f} steps/s = {MAX_RPM:.0f} RPM (after {GEAR_RATIO}:1 gearbox)"
        )

    def _cb_left(self, msg: Float32):
        pwm = max(-1.0, min(1.0, msg.data))
        self._left.set_pwm(pwm)

    def _cb_right(self, msg: Float32):
        pwm = max(-1.0, min(1.0, msg.data))
        self._right.set_pwm(pwm)

    def destroy_node(self):
        self._left.stop()
        self._right.stop()
        _gpio_cleanup()
        super().destroy_node()


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = StepperDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
