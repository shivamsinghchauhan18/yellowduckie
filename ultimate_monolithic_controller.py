#!/usr/bin/env python3
# ultimate_monolithic_controller.py
# A single-file, end-to-end controller for Duckiebot:
# - Camera, lane pose, stop line, apriltags, vehicle detections, IMU (optional)
# - Optional ONNX lane-following (fallback to PD)
# - Optional Roboflow object detection via HTTP API
# - Following-distance control + basic safety overrides
# - Publishes directly to wheels (bypasses launch/remaps/arbiter)
#
# Designed to run inside dt-duckiebot-interface container.

import os
import sys
import time
import math
import json
import base64
import threading
import traceback
from collections import deque
from typing import Optional, Tuple, Dict

import numpy as np

# OpenCV is in the container
import cv2

# ROS
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Imu
from duckietown_msgs.msg import (
    Twist2DStamped, LanePose, VehicleCorners, StopLineReading, AprilTagDetectionArray
)

# Try to enable ONNX if available; otherwise gracefully degrade
ONNX_OK = False
try:
    import onnxruntime as ort
    ONNX_OK = True
except Exception:
    ONNX_OK = False

# Try to use requests for Roboflow, otherwise urllib
REQUESTS_OK = False
try:
    import requests
    REQUESTS_OK = True
except Exception:
    REQUESTS_OK = False
import urllib.request

# =========================
# ======= TUNABLES ========
# =========================
DEFAULT_RATE_HZ = 20

# PD fallback (if no ONNX lane model)
PD_KP_D = 3.0     # proportional on lateral offset d
PD_KD_D = 0.0     # derivative on d (not used here)
PD_KP_PHI = 8.0   # proportional on heading phi
PD_KD_PHI = 0.0

BASE_V = 0.30     # nominal linear velocity (m/s)
MIN_V = 0.05
MAX_V = 0.60

# Slowdown factors
VISIBILITY_MIN_BRIGHTNESS = 30
VISIBILITY_MAX_REDUCTION = 0.7   # reduce by up to 70%

# Following distance control
DESIRED_TIME_GAP = 0.8   # seconds
MIN_HEADWAY_V = 0.15     # below this speed, relax following
MAX_DECEL = 0.4          # m/s^2 clamp when reducing speed

# Stop line handling
STOP_LINE_STOP_TIME = 2.0  # seconds
STOP_LINE_DISTANCE_THRESH = 0.15  # meters

# Roboflow throttling
ROBOFLOW_PERIOD = 0.33  # seconds between inference calls (~3 Hz)

# Collision/safety
COLLISION_RISK_STOP = {"high", "critical", "emergency"}

# ONNX preprocessing defaults (adjust to your model)
ONNX_INPUT_W = 224
ONNX_INPUT_H = 224
ONNX_INPUT_NAME = None  # auto-detect
ONNX_OUTPUT_NAME = None # auto-detect
# Expected output: [linear, angular] or [steer] — handled below in postproc


def now():
    return rospy.get_time() if rospy.core.is_initialized() else time.time()


class UltimateMonolithicController:
    def __init__(self,
                 veh: str,
                 rate_hz: int = DEFAULT_RATE_HZ,
                 onnx_model_path: Optional[str] = None,
                 roboflow_model_id: Optional[str] = None):
        self.veh = veh
        self.rate_hz = rate_hz
        self.dt = 1.0 / max(1, rate_hz)

        # State
        self.lock = threading.Lock()
        self.last_img_bgr = None
        self.last_img_time = 0.0

        self.last_lane_pose: Optional[LanePose] = None
        self.last_stop_line: Optional[StopLineReading] = None
        self.last_apriltags: Optional[AprilTagDetectionArray] = None
        self.last_vehicle_corners: Optional[VehicleCorners] = None
        self.last_imu: Optional[Imu] = None

        self.last_collision_risk: Optional[str] = None
        self.last_safety_status: Optional[str] = None

        # Diagnostics
        self.cmd_hist = deque(maxlen=200)
        self.stop_until = 0.0
        self.last_roboflow_t = 0.0
        self.last_roboflow_det = []

        # ONNX (optional)
        self.onnx_session = None
        self.use_onnx = False
        if onnx_model_path and ONNX_OK and os.path.exists(onnx_model_path):
            try:
                ropts = ort.SessionOptions()
                ropts.intra_op_num_threads = 1
                ropts.inter_op_num_threads = 1
                self.onnx_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'], sess_options=ropts)
                self.use_onnx = True
                rospy.loginfo(f"[UMC] ONNX lane model loaded: {onnx_model_path}")
            except Exception as e:
                rospy.logwarn(f"[UMC] Failed to load ONNX model: {e}. Falling back to PD.")

        # Roboflow (optional)
        self.roboflow_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
        self.roboflow_model_id = roboflow_model_id
        self.use_roboflow = bool(self.roboflow_key and self.roboflow_model_id)

        if self.use_roboflow:
            rospy.loginfo(f"[UMC] Roboflow enabled: model={self.roboflow_model_id}")

        # ROS wiring (ABSOLUTE topics; no remaps)
        ns = f"/{self.veh}"
        self.pub_cmd = rospy.Publisher(f"{ns}/wheels_driver_node/wheels_cmd", Twist2DStamped, queue_size=1)

        rospy.Subscriber(f"{ns}/camera_node/image/compressed", CompressedImage, self.cb_image, queue_size=1)
        rospy.Subscriber(f"{ns}/lane_filter_node/lane_pose", LanePose, self.cb_lane_pose, queue_size=1)
        rospy.Subscriber(f"{ns}/stop_line_filter_node/stop_line_reading", StopLineReading, self.cb_stop_line, queue_size=1)
        rospy.Subscriber(f"{ns}/apriltag_detector_node/detections", AprilTagDetectionArray, self.cb_apriltags, queue_size=1)
        rospy.Subscriber(f"{ns}/vehicle_detection_node/centers", VehicleCorners, self.cb_vehicle_corners, queue_size=1)
        rospy.Subscriber(f"{ns}/imu_node/data", Imu, self.cb_imu, queue_size=1)

        # Optional safety feeds (if running)
        try:
            rospy.Subscriber(f"{ns}/collision_detection_manager_node/collision_risk", String, self.cb_collision_risk, queue_size=1)
        except Exception:
            pass
        try:
            rospy.Subscriber(f"{ns}/safety_fusion_manager_node/safety_status", String, self.cb_safety_status, queue_size=1)
        except Exception:
            pass

        rospy.loginfo("[UMC] UltimateMonolithicController initialized.")

    # ------------- Callbacks -------------
    def cb_image(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                with self.lock:
                    self.last_img_bgr = img
                    self.last_img_time = now()
        except Exception:
            rospy.logwarn("[UMC] Image decode failed:\n" + traceback.format_exc())

    def cb_lane_pose(self, msg: LanePose):
        with self.lock:
            self.last_lane_pose = msg

    def cb_stop_line(self, msg: StopLineReading):
        with self.lock:
            self.last_stop_line = msg

    def cb_apriltags(self, msg: AprilTagDetectionArray):
        with self.lock:
            self.last_apriltags = msg

    def cb_vehicle_corners(self, msg: VehicleCorners):
        with self.lock:
            self.last_vehicle_corners = msg

    def cb_imu(self, msg: Imu):
        with self.lock:
            self.last_imu = msg

    def cb_collision_risk(self, msg: String):
        with self.lock:
            self.last_collision_risk = msg.data.strip().lower()

    def cb_safety_status(self, msg: String):
        with self.lock:
            self.last_safety_status = msg.data.strip().lower()

    # ------------- Control Helpers -------------
    def compute_lane_control_pd(self, pose: LanePose) -> Tuple[float, float]:
        """Fallback PD lane follower when ONNX is not used."""
        d = pose.d
        phi = pose.phi
        omega = -(PD_KP_D * d + PD_KP_PHI * phi)
        v = BASE_V
        return v, omega

    def onnx_lane_control(self, img_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
        """Infer [v, omega] or [steer] from ONNX if supported."""
        if not self.use_onnx or self.onnx_session is None or img_bgr is None:
            return None
        try:
            # Preprocess
            resized = cv2.resize(img_bgr, (ONNX_INPUT_W, ONNX_INPUT_H))
            inp = resized.astype(np.float32) / 255.0
            inp = inp.transpose(2, 0, 1)  # CHW
            inp = np.expand_dims(inp, 0)  # NCHW

            input_name = ONNX_INPUT_NAME or self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: inp})
            # Heuristic: if 2 floats => [v, omega]; if 1 float => steer
            out = outputs[0]
            out = np.array(out).reshape(-1)
            if out.size >= 2:
                v = float(np.clip(out[0], -MAX_V, MAX_V))
                omega = float(out[1])
            else:
                steer = float(out[0])
                v = BASE_V
                omega = steer
            return v, omega
        except Exception:
            rospy.logwarn("[UMC] ONNX inference failed, falling back to PD once:\n" + traceback.format_exc())
            return None

    def estimate_visibility_factor(self, img_bgr: Optional[np.ndarray]) -> float:
        """Compute a [0..1] speed multiplier based on brightness/contrast."""
        if img_bgr is None:
            return 1.0
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_bright = float(np.mean(gray))
        if mean_bright < VISIBILITY_MIN_BRIGHTNESS:
            # strong reduction under poor light
            return max(0.0, 1.0 - VISIBILITY_MAX_REDUCTION)
        return 1.0

    def following_speed_limit(self, current_v: float) -> float:
        """Cap speed based on nearest vehicle distance if available."""
        vc = self.last_vehicle_corners
        if vc is None or len(vc.corners) == 0:
            return current_v
        # crude proxy: use the minimum y (forward distance) if encoded,
        # but VehicleCorners doesn't directly give z-dist; assume a conservative cap:
        # Slow if we detect any vehicle ahead.
        desired = max(MIN_V, min(current_v, 0.25))
        return desired

    def should_stop_for_stopline(self) -> bool:
        sl = self.last_stop_line
        if sl and sl.at_stop_line and sl.stop_line_detected:
            if sl.dist < STOP_LINE_DISTANCE_THRESH:
                return True
        return False

    def roboflow_detect(self, img_bgr: Optional[np.ndarray]) -> list:
        """Call Roboflow only every ROBOFLOW_PERIOD sec. Returns list of detections."""
        if not self.use_roboflow or img_bgr is None:
            return []
        t = now()
        if t - self.last_roboflow_t < ROBOFLOW_PERIOD:
            return self.last_roboflow_det
        self.last_roboflow_t = t

        try:
            _, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            url = f"https://detect.roboflow.com/{self.roboflow_model_id}"
            params = f"?api_key={self.roboflow_key}&format=json"
            payload = b64

            if REQUESTS_OK:
                resp = requests.post(url + params, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
                data = resp.json()
            else:
                req = urllib.request.Request(url + params, data=payload.encode("utf-8"),
                                             headers={"Content-Type": "application/x-www-form-urlencoded"})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

            preds = data.get("predictions", [])
            self.last_roboflow_det = preds
            return preds
        except Exception:
            # silent failure (network down etc.)
            return self.last_roboflow_det

    def roboflow_speed_cap(self, detections: list, img_shape: Tuple[int, int]) -> float:
        """Crude slowdown if big obstacle bounding boxes in front region."""
        if not detections:
            return MAX_V
        H, W = img_shape[:2]
        # consider central vertical band (middle 40%)
        x1 = int(W * 0.3)
        x2 = int(W * 0.7)
        severe = False
        moderate = False
        for det in detections:
            # Roboflow returns center-x,y and width/height OR x/y?
            # Handle both possible formats
            if all(k in det for k in ("x", "y", "width", "height")):
                cx, cy, w, h = det["x"], det["y"], det["width"], det["height"]
                left = cx - w / 2.0
                right = cx + w / 2.0
                if right >= x1 and left <= x2:
                    area = (w * h) / float(W * H + 1e-6)
                    if area > 0.08:
                        severe = True
                    elif area > 0.03:
                        moderate = True
        if severe:
            return 0.0
        if moderate:
            return 0.15
        return MAX_V

    def emergency_override(self) -> bool:
        """Stop if safety says emergency or collision risk is high/critical."""
        s = self.last_safety_status or ""
        if any(tok in s for tok in ["emergency", "stop"]):
            return True
        r = self.last_collision_risk or ""
        if r in COLLISION_RISK_STOP:
            return True
        return False

    # ------------- Main control loop -------------
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            try:
                with self.lock:
                    img = None if self.last_img_bgr is None else self.last_img_bgr.copy()
                    pose = self.last_lane_pose
                    stopline = self.last_stop_line

                # 1) Base lane command
                v_cmd, omega_cmd = BASE_V, 0.0
                used_onnx = False
                if self.use_onnx:
                    out = self.onnx_lane_control(img)
                    if out is not None:
                        v_cmd, omega_cmd = out
                        used_onnx = True

                if not used_onnx:
                    if pose is not None:
                        v_cmd, omega_cmd = self.compute_lane_control_pd(pose)
                    else:
                        v_cmd, omega_cmd = 0.15, 0.0  # limp

                # 2) Visibility adaptation
                vis_factor = self.estimate_visibility_factor(img)
                v_cmd *= vis_factor

                # 3) Following distance
                v_cmd = min(v_cmd, self.following_speed_limit(v_cmd))

                # 4) Stop line logic
                if self.should_stop_for_stopline():
                    self.stop_until = now() + STOP_LINE_STOP_TIME
                if now() < self.stop_until:
                    v_cmd = 0.0
                    omega_cmd = 0.0

                # 5) Roboflow object detection (optional)
                if self.use_roboflow and img is not None:
                    dets = self.roboflow_detect(img)
                    cap = self.roboflow_speed_cap(dets, img.shape)
                    v_cmd = min(v_cmd, cap)

                # 6) Safety emergency override
                if self.emergency_override():
                    v_cmd, omega_cmd = 0.0, 0.0

                # 7) Clamp + jerk limiting
                v_prev = self.cmd_hist[-1][0] if self.cmd_hist else v_cmd
                dv = np.clip(v_cmd - v_prev, -MAX_DECEL * self.dt, MAX_DECEL * self.dt)
                v_cmd = v_prev + dv
                v_cmd = float(np.clip(v_cmd, MIN_V if v_cmd > 0 else 0.0, MAX_V))

                # 8) Publish wheels command
                cmd = Twist2DStamped()
                cmd.header.stamp = rospy.Time.now()
                cmd.v = v_cmd
                cmd.omega = float(np.clip(omega_cmd, -8.0, 8.0))
                self.pub_cmd.publish(cmd)

                self.cmd_hist.append((v_cmd, omega_cmd, now()))

            except Exception:
                rospy.logerr("[UMC] Control loop error:\n" + traceback.format_exc())

            rate.sleep()


def parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(description="Ultimate Monolithic Duckiebot Controller")
    p.add_argument("--veh", default=os.environ.get("VEHICLE_NAME", "autobot"),
                   help="Vehicle name (namespace)")
    p.add_argument("--rate_hz", type=int, default=DEFAULT_RATE_HZ)
    p.add_argument("--onnx_model_path", type=str, default="",
                   help="Path to ONNX lane-following model (optional)")
    p.add_argument("--roboflow_model_id", type=str, default="",
                   help="Roboflow model id like 'workspace/model/1' (optional)")
    return p.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    rospy.init_node("ultimate_monolithic_controller", anonymous=False)

    ctrl = UltimateMonolithicController(
        veh=args.veh,
        rate_hz=args.rate_hz,
        onnx_model_path=args.onnx_model_path if args.onnx_model_path else None,
        roboflow_model_id=args.roboflow_model_id if args.roboflow_model_id else None
    )
    ctrl.spin()


if __name__ == "__main__":
    main()