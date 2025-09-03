#!/usr/bin/env python3
# ultimate_monolithic_controller_v3.py
# Single-file controller with ONNX, PD(lane_pose), and **OpenCV lane follower** fallback + rich logs & debug image.

import os, sys, time, json, math, base64, threading, traceback
from collections import deque
from typing import Optional, Tuple, Dict, List

import numpy as np
import cv2

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Imu
from duckietown_msgs.msg import (
    Twist2DStamped, LanePose, VehicleCorners, StopLineReading, AprilTagDetectionArray
)
from std_srvs.srv import SetBool, SetBoolResponse

# Optional deps
ONNX_OK = False
try:
    import onnxruntime as ort
    ONNX_OK = True
except Exception:
    ONNX_OK = False

REQUESTS_OK = False
try:
    import requests
    REQUESTS_OK = True
except Exception:
    REQUESTS_OK = False
import urllib.request

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x
def now(): return rospy.get_time() if rospy.core.is_initialized() else time.time()

class RateLimiter:
    def __init__(self, period_s: float):
        self.period = max(0.0, float(period_s))
        self.tlast = 0.0
    def ready(self) -> bool:
        t = now()
        if t - self.tlast >= self.period:
            self.tlast = t
            return True
        return False

class UMC:
    # ===== Tunables =====
    BASE_V = 0.32
    MIN_V = 0.05
    MAX_V = 0.65

    PD_KP_D = 3.2
    PD_KP_PHI = 9.0

    # CV steering
    CV_KP_PIX = 4.0          # steer gain on normalized pixel error
    CV_KD_PIX = 0.0
    CV_MIN_CONF = 0.08       # if mask coverage lower than this, reduce speed/steer
    CV_USE_HOUGH = True

    # Fallback timing
    POSE_STALE_S = 0.5
    CAM_STALE_S  = 1.0

    # Visibility adaptation
    VIS_BRIGHT_MIN = 35.0
    VIS_CONTRAST_MIN = 15.0
    VIS_MAX_REDUCTION = 0.7

    # Following distance heuristic
    FD_MAX_V_WITH_AHEAD = 0.25

    # Stop line
    STOP_DIST_THRESH = 0.14
    STOP_TIME_S = 2.0

    # Accel/Jerk limiting
    MAX_ACCEL = 0.45
    SMOOTH_WINDOW = 5
    OMEGA_CLAMP = 8.0

    RISK_STOP = {"high", "critical", "emergency"}
    STATUS_STOP_TOKENS = ("emergency", "stop")

    # ONNX preproc
    ONNX_W = 224
    ONNX_H = 224

    def __init__(self, veh: str, rate_hz: int, onnx_model_path: Optional[str],
                 roboflow_model_id: Optional[str], rf_period: float, log_interval: float,
                 debug_viz: bool):
        self.veh = veh
        self.rate_hz = max(1, int(rate_hz))
        self.dt = 1.0 / self.rate_hz

        self.lock = threading.Lock()
        self.img_bgr: Optional[np.ndarray] = None
        self.t_img = 0.0

        self.pose: Optional[LanePose] = None
        self.t_pose = 0.0

        self.stop_line: Optional[StopLineReading] = None
        self.apriltags: Optional[AprilTagDetectionArray] = None
        self.veh_corners: Optional[VehicleCorners] = None
        self.imu: Optional[Imu] = None
        self.collision_risk: Optional[str] = None
        self.safety_status: Optional[str] = None

        # Predictive perception (string)
        self.scene_analysis: Optional[String] = None
        self.predicted_traj: Optional[String] = None

        self.v_hist = deque(maxlen=50)
        self.cmd_hist = deque(maxlen=200)
        self.stop_until = 0.0

        # Behavior/Mode
        self.state = "LF"  # LF | STOP | OBSTACLE_STOP | INTERSECTION_WAIT | AVOID
        self.state_changed_t = now()
        self.mode = "auto"  # auto | manual | paused
        self.paused = False

        # Teleop override (best-effort passthrough)
        self.teleop_cmd: Optional[Twist2DStamped] = None
        self.t_teleop = 0.0
        self.TELEOP_TIMEOUT_S = 0.5

        # Lane biasing (for avoidance/lane-change feel)
        self.lane_bias = 0.0           # meters (desired center offset)
        self.lane_bias_target = 0.0    # meters
        self.LANE_BIAS_MAX = 0.07      # m
        self.LANE_BIAS_RATE = 0.05     # m/s slew

        # ONNX
        self.use_onnx = False
        self.onnx_session = None
        if onnx_model_path and ONNX_OK and os.path.exists(onnx_model_path):
            try:
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                self.onnx_session = ort.InferenceSession(
                    onnx_model_path,
                    providers=['CPUExecutionProvider'],
                    sess_options=so
                )
                self.use_onnx = True
                rospy.loginfo(f"[UMC] ONNX loaded: {onnx_model_path}")
            except Exception as e:
                rospy.logwarn(f"[UMC] ONNX failed: {e} (fallback to CV/PD)")

        # Roboflow
        self.rf_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
        self.rf_model = roboflow_model_id
        self.use_rf = bool(self.rf_key and self.rf_model)
        self.rf_rate = RateLimiter(period_s=rf_period)
        self.rf_last = []

        # Logging
        self.log_rate = RateLimiter(period_s=max(0.1, float(log_interval)))
        self.start_t = now()

        # CV state
        self.cv_prev_err = 0.0
        self.debug_viz = bool(debug_viz)

        # ROS wiring
        ns = f"/{self.veh}"
        self.pub_cmd = rospy.Publisher(f"{ns}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_status = rospy.Publisher(f"{ns}/umc/status", String, queue_size=1)
        self.pub_dbg = rospy.Publisher(f"{ns}/umc/debug_image/compressed", CompressedImage, queue_size=1) if self.debug_viz else None
        self.pub_diag = rospy.Publisher(f"{ns}/umc/diagnostics", String, queue_size=1)

        rospy.Subscriber(f"{ns}/camera_node/image/compressed", CompressedImage, self.cb_image, queue_size=1)
        rospy.Subscriber(f"{ns}/lane_filter_node/lane_pose", LanePose, self.cb_lane_pose, queue_size=1)
        rospy.Subscriber(f"{ns}/stop_line_filter_node/stop_line_reading", StopLineReading, self.cb_stop_line, queue_size=1)
        rospy.Subscriber(f"{ns}/apriltag_detector_node/detections", AprilTagDetectionArray, self.cb_apriltags, queue_size=1)
        rospy.Subscriber(f"{ns}/vehicle_detection_node/centers", VehicleCorners, self.cb_vehicle_corners, queue_size=1)
        rospy.Subscriber(f"{ns}/imu_node/data", Imu, self.cb_imu, queue_size=1)

        # Teleop passthrough (if present); harmless if topic absent
        try:
            rospy.Subscriber(f"{ns}/joy_mapper_node/car_cmd", Twist2DStamped, self.cb_teleop, queue_size=1)
        except Exception:
            pass

        # Safety (optional)
        try:
            rospy.Subscriber(f"{ns}/collision_detection_manager_node/collision_risk", String, self.cb_collision_risk, queue_size=1)
        except Exception:
            pass
        try:
            rospy.Subscriber(f"{ns}/safety_fusion_manager_node/safety_status", String, self.cb_safety_status, queue_size=1)
        except Exception:
            pass

        # Predictive perception strings (optional)
        try:
            rospy.Subscriber(f"{ns}/predictive_perception_manager_node/predicted_trajectories", String, self.cb_predicted_traj, queue_size=1)
        except Exception:
            pass
        try:
            rospy.Subscriber(f"{ns}/scene_understanding_module_node/scene_analysis", String, self.cb_scene_analysis, queue_size=1)
        except Exception:
            pass

        # Runtime parameter updates via JSON string
        try:
            rospy.Subscriber(f"{ns}/umc/param_update", String, self.cb_param_update, queue_size=1)
        except Exception:
            pass

        # Services: pause/resume (SetBool True=pause, False=resume)
        try:
            rospy.Service(f"{ns}/umc/pause", SetBool, self.srv_pause)
        except Exception:
            pass

        rospy.loginfo(f"[UMC] Initialized veh={self.veh} rate={self.rate_hz}Hz "
                      f"onnx={'yes' if self.use_onnx else 'no'} "
                      f"roboflow={'yes' if self.use_rf else 'no'}")

    # ==== Callbacks ====
    def cb_image(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                with self.lock:
                    self.img_bgr = img
                    self.t_img = now()
        except Exception:
            rospy.logwarn("[UMC] Image decode failed:\n" + traceback.format_exc())

    def cb_lane_pose(self, msg: LanePose):
        with self.lock:
            self.pose = msg
            self.t_pose = now()

    def cb_stop_line(self, msg: StopLineReading):
        with self.lock:
            self.stop_line = msg

    def cb_apriltags(self, msg: AprilTagDetectionArray):
        with self.lock:
            self.apriltags = msg

    def cb_vehicle_corners(self, msg: VehicleCorners):
        with self.lock:
            self.veh_corners = msg

    def cb_imu(self, msg: Imu):
        with self.lock:
            self.imu = msg

    def cb_collision_risk(self, msg: String):
        with self.lock:
            self.collision_risk = (msg.data or "").strip().lower()

    def cb_safety_status(self, msg: String):
        with self.lock:
            self.safety_status = (msg.data or "").strip().lower()

    def cb_predicted_traj(self, msg: String):
        with self.lock:
            self.predicted_traj = msg

    def cb_scene_analysis(self, msg: String):
        with self.lock:
            self.scene_analysis = msg

    def cb_teleop(self, msg: Twist2DStamped):
        # Record last teleop for manual override behavior
        with self.lock:
            self.teleop_cmd = msg
            self.t_teleop = now()

    def cb_param_update(self, msg: String):
        # Accept a JSON blob to update tunables at runtime (best-effort)
        try:
            data = json.loads(msg.data)
        except Exception:
            rospy.logwarn("[UMC] param_update: invalid JSON")
            return
        updated = {}
        with self.lock:
            for k, v in data.items():
                if not isinstance(k, str):
                    continue
                if not hasattr(self, k):
                    continue
                try:
                    # Guardrails: clamp some known fields
                    if k in ("BASE_V", "MIN_V", "MAX_V"):
                        v = float(v)
                        v = float(clamp(v, 0.0, 1.0))
                    if k in ("PD_KP_D", "PD_KP_PHI", "CV_KP_PIX", "CV_KD_PIX"):
                        v = float(v)
                    if k in ("LANE_BIAS_MAX", "LANE_BIAS_RATE"):
                        v = float(max(0.0, v))
                    setattr(self, k, v)
                    updated[k] = v
                except Exception:
                    pass
        if updated:
            rospy.loginfo(f"[UMC] param_update applied: {updated}")

    def srv_pause(self, req: SetBool):
        with self.lock:
            self.paused = bool(req.data)
            if self.paused:
                self.mode = "paused"
            else:
                self.mode = "auto"
        return SetBoolResponse(success=True, message=("paused" if self.paused else "resumed"))

    # ==== Perception-derived scalars ====
    def visibility_factor(self, img_bgr: Optional[np.ndarray]) -> float:
        if img_bgr is None:
            return 0.0
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))
        std_b = float(np.std(gray))
        if mean_b < self.VIS_BRIGHT_MIN or std_b < self.VIS_CONTRAST_MIN:
            return max(0.0, 1.0 - self.VIS_MAX_REDUCTION)
        return 1.0

    def lane_confidence_factor(self, pose: Optional[LanePose]) -> float:
        if pose is None:
            return 0.8
        d = abs(pose.d); phi = abs(pose.phi)
        factor = 1.0
        factor *= clamp(1.0 - d/0.12, 0.4, 1.0)
        factor *= clamp(1.0 - phi/0.6, 0.4, 1.0)
        return factor

    def imu_speed_factor(self) -> float:
        """Reduce speed when yaw rate is high or IMU is stale."""
        imu = self.imu
        if imu is None:
            return 1.0
        try:
            wz = float(imu.angular_velocity.z)
            # Downscale between ~0.5..1.0 as yaw rate rises
            fac = 1.0 / (1.0 + 0.35 * abs(wz))
            return clamp(fac, 0.5, 1.0)
        except Exception:
            return 1.0

    def curvature_speed_factor(self, omega: float) -> float:
        """Velocity reduction based on commanded curvature (omega)."""
        a = 0.20  # aggressiveness
        fac = 1.0 / (1.0 + a * abs(float(omega)))
        return clamp(fac, 0.35, 1.0)

    def following_cap(self) -> float:
        vc = self.veh_corners
        if vc and len(vc.corners) > 0:
            return self.FD_MAX_V_WITH_AHEAD
        return self.MAX_V

    def stopline_triggered(self) -> bool:
        sl = self.stop_line
        if not sl: return False
        if sl.at_stop_line and getattr(sl, "stop_line_detected", True):
            dist = getattr(sl, "dist", 1.0)
            return dist < self.STOP_DIST_THRESH
        return False

    def safety_emergency(self) -> bool:
        s = (self.safety_status or "")
        if any(tok in s for tok in self.STATUS_STOP_TOKENS):
            return True
        r = (self.collision_risk or "")
        return r in self.RISK_STOP

    def predictive_slowdown_factor(self) -> float:
        fac = 1.0
        try:
            sa = self.scene_analysis.data if isinstance(self.scene_analysis, String) else None
            pt = self.predicted_traj.data if isinstance(self.predicted_traj, String) else None
            txt = f"{sa or ''} {pt or ''}".lower()
            if "crossing" in txt or "pedestrian" in txt or "duck" in txt:
                fac = min(fac, 0.5)
            if "ttc" in txt:
                import re
                m = re.search(r"ttc[^0-9]*([0-9]+(\.[0-9]+)?)", txt)
                if m:
                    ttc = float(m.group(1))
                    if ttc < 1.0: fac = min(fac, 0.0)
                    elif ttc < 2.0: fac = min(fac, 0.4)
                    elif ttc < 3.0: fac = min(fac, 0.7)
        except Exception:
            pass
        return fac

    # ==== Controllers ====
    def onnx_lane(self, img_bgr: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        if not self.use_onnx or self.onnx_session is None or img_bgr is None:
            return None
        try:
            resized = cv2.resize(img_bgr, (self.ONNX_W, self.ONNX_H))
            x = resized.astype(np.float32) / 255.0
            x = x.transpose(2, 0, 1)[None, ...]
            input_name = self.onnx_session.get_inputs()[0].name
            y = self.onnx_session.run(None, {input_name: x})[0]
            y = np.array(y).reshape(-1)
            if y.size >= 2:
                v = float(clamp(y[0], -self.MAX_V, self.MAX_V))
                omega = float(clamp(y[1], -self.OMEGA_CLAMP, self.OMEGA_CLAMP))
            else:
                steer = float(clamp(y[0], -self.OMEGA_CLAMP, self.OMEGA_CLAMP))
                v = self.BASE_V
                omega = steer
            return v, omega
        except Exception:
            rospy.logwarn("[UMC] ONNX inference error; using PD/CV this tick:\n" + traceback.format_exc())
            return None

    def pd_lane(self, pose: Optional[LanePose], desired_d: float = 0.0) -> Tuple[float, float]:
        """PD control on lane pose with optional lateral bias (desired_d in meters)."""
        if pose is None:
            return 0.16, 0.0
        v = self.BASE_V
        d_err = (pose.d - float(desired_d))
        omega = -(self.PD_KP_D * d_err + self.PD_KP_PHI * pose.phi)
        return v, omega

    # ==== OpenCV lane follower (yellow/white) ====
    def cv_lane(self, img_bgr: Optional[np.ndarray], bias_m: float = 0.0) -> Tuple[Optional[Tuple[float,float]], float, Optional[np.ndarray]]:
        """
        Returns ((v, omega), conf, debug_image) or (None, 0.0, dbg) if not enough info.
        """
        if img_bgr is None:
            return (None, 0.0, None)

        H, W = img_bgr.shape[:2]
        # ROI: lower 45% of image
        y0 = int(H * 0.55)
        roi = img_bgr[y0:H, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Yellow mask
        # H: 15..40, S: 70..255, V: 70..255
        lower_y = np.array([15, 70, 70], dtype=np.uint8)
        upper_y = np.array([40, 255, 255], dtype=np.uint8)
        mask_y = cv2.inRange(hsv, lower_y, upper_y)

        # White mask (low saturation, high value)
        lower_w = np.array([0, 0, 200], dtype=np.uint8)
        upper_w = np.array([180, 60, 255], dtype=np.uint8)
        mask_w = cv2.inRange(hsv, lower_w, upper_w)

        mask = cv2.bitwise_or(mask_y, mask_w)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

        # Column histogram
        col_sum = np.sum(mask > 0, axis=0).astype(np.float32)
        area = float(np.sum(col_sum))
        roi_area = float(mask.shape[0]*mask.shape[1] + 1e-6)
        coverage = area / roi_area  # 0..1

        x_coords = np.arange(W, dtype=np.float32)
        if area > 1.0:
            cx = float(np.sum(x_coords * col_sum) / (np.sum(col_sum) + 1e-6))
        else:
            cx = W/2.0

        # Pixel error normalized to [-1,1]
        # Convert bias in meters to pixels approximately by mapping typical lane half-width ~0.11 m => ~W*0.25 px (rough heuristic)
        bias_px = clamp(bias_m / 0.11, -1.0, 1.0) * (W * 0.25)
        err = (((cx - bias_px) - (W/2.0)) / (W/2.0))
        derr = (err - self.cv_prev_err) / max(1e-3, self.dt)
        self.cv_prev_err = err

        # Optional heading via Hough to tweak omega
        heading_bias = 0.0
        if self.CV_USE_HOUGH:
            edges = cv2.Canny(mask, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=20)
            angles = []
            if lines is not None:
                for l in lines[:50]:
                    x1,y1,x2,y2 = l[0]
                    dx = x2-x1; dy = y2-y1
                    if dx == 0: continue
                    ang = math.atan2(dy, dx)  # rad
                    # consider near-vertical lines only (lane markings)
                    if abs(abs(ang) - math.pi/2) < math.radians(35):
                        angles.append(ang)
            if angles:
                # positive ang means leaning right, so steer accordingly
                mean_ang = float(np.mean(angles))
                heading_bias = clamp(-0.8 * mean_ang, -1.0, 1.0)  # small contribution

        omega = -(self.CV_KP_PIX * err + self.CV_KD_PIX * derr) + heading_bias

        # Confidence → speed scaling
        conf = float(coverage * 6.0)  # scale coverage; ~0.16 area → conf ~1
        conf = clamp(conf, 0.0, 1.0)
        v = self.BASE_V * (0.25 + 0.75 * conf)  # min 25% base when low conf

        # Debug image
        dbg = None
        if self.pub_dbg is not None:
            dbg = img_bgr.copy()
            # draw ROI box
            cv2.rectangle(dbg, (0, y0), (W-1, H-1), (80,80,80), 2)
            # overlay mask
            colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay = dbg[y0:H, :].copy()
            overlay = cv2.addWeighted(overlay, 0.6, colored, 0.4, 0)
            dbg[y0:H, :] = overlay
            # draw center and cx
            cv2.line(dbg, (W//2, y0), (W//2, H-1), (255,255,255), 1)
            cv2.line(dbg, (int(cx), y0), (int(cx), H-1), (0,255,255), 2)
            # text
            cv2.putText(dbg, f"CV err={err:+.2f} conf={conf:.2f}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,220,50), 2, cv2.LINE_AA)

        return ((v, float(clamp(omega, -self.OMEGA_CLAMP, self.OMEGA_CLAMP))), conf, dbg)

    # ==== Roboflow ====
    def rf_detect(self, img_bgr: Optional[np.ndarray]) -> List[dict]:
        if not self.use_rf or img_bgr is None or not self.rf_rate.ready():
            return self.rf_last
        try:
            ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return self.rf_last
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            url = f"https://detect.roboflow.com/{self.rf_model}?api_key={self.rf_key}&format=json"
            if REQUESTS_OK:
                r = requests.post(url, data=b64, headers={"Content-Type":"application/x-www-form-urlencoded"}, timeout=2.0)
                data = r.json()
            else:
                req = urllib.request.Request(url, data=b64.encode("utf-8"),
                                             headers={"Content-Type":"application/x-www-form-urlencoded"})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            preds = data.get("predictions", [])
            self.rf_last = preds
            return preds
        except Exception:
            return self.rf_last

    def rf_speed_cap(self, dets: List[dict], shape: Tuple[int,int]) -> float:
        if not dets:
            return self.MAX_V
        H, W = shape[:2]
        x1 = int(W * 0.35); x2 = int(W * 0.65)
        severe = False; moderate = False
        for d in dets:
            cx, cy, w, h = d.get("x"), d.get("y"), d.get("width"), d.get("height")
            if cx is None or cy is None or w is None or h is None: continue
            left, right = cx - w/2.0, cx + w/2.0
            if right >= x1 and left <= x2:
                area = (w*h) / float(W*H + 1e-6)
                if area > 0.10: severe = True
                elif area > 0.04: moderate = True
        if severe: return 0.0
        if moderate: return 0.18
        return self.MAX_V

    def update_lane_bias(self):
        """Slew lane_bias toward target."""
        if self.lane_bias_target > 0:
            self.lane_bias_target = min(self.lane_bias_target, self.LANE_BIAS_MAX)
        else:
            self.lane_bias_target = max(self.lane_bias_target, -self.LANE_BIAS_MAX)
        db = self.lane_bias_target - self.lane_bias
        max_step = self.LANE_BIAS_RATE * self.dt
        self.lane_bias += clamp(db, -max_step, max_step)

    # ==== Main loop ====
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        ema_v = self.BASE_V
        while not rospy.is_shutdown():
            try:
                with self.lock:
                    img = None if self.img_bgr is None else self.img_bgr.copy()
                    pose = self.pose
                    t_pose = self.t_pose
                    coll_risk = self.collision_risk or ""
                    safety = self.safety_status or ""
                    teleop_cmd = self.teleop_cmd
                    t_teleop = self.t_teleop
                    paused = self.paused

                # Camera staleness safety
                if img is None or (now() - self.t_img) > self.CAM_STALE_S:
                    self.publish_cmd(0.0, 0.0)
                    self.maybe_log(controller="none(cam_stale)", used_onnx=False, v_cmd=0.0, omega_cmd=0.0,
                                   vis_fac=0.0, lane_conf_fac=0.0, fd_cap=0.0, rf_active=self.use_rf,
                                   rf_last_n=0, safety=safety, collision_risk=coll_risk, stopped=True)
                    rate.sleep()
                    continue

                # Pause or Safety emergency stop overrides everything
                if paused or self.safety_emergency():
                    self.publish_cmd(0.0, 0.0)
                    self.maybe_log(controller=("paused" if paused else "safety_stop"), used_onnx=False, v_cmd=0.0, omega_cmd=0.0,
                                   vis_fac=0.0, lane_conf_fac=0.0, fd_cap=0.0, rf_active=self.use_rf,
                                   rf_last_n=0, safety=safety, collision_risk=coll_risk, stopped=True)
                    rate.sleep()
                    continue

                # Manual override if fresh teleop
                if teleop_cmd is not None and (now() - t_teleop) < self.TELEOP_TIMEOUT_S:
                    v_m = float(clamp(teleop_cmd.v, -self.MAX_V, self.MAX_V))
                    o_m = float(clamp(teleop_cmd.omega, -self.OMEGA_CLAMP, self.OMEGA_CLAMP))
                    self.publish_cmd(v_m, o_m)
                    self.maybe_log(controller="manual", used_onnx=False, v_cmd=v_m, omega_cmd=o_m,
                                   vis_fac=1.0, lane_conf_fac=1.0, fd_cap=self.MAX_V, rf_active=self.use_rf,
                                   rf_last_n=0, safety=safety, collision_risk=coll_risk, stopped=(abs(v_m) < 1e-3))
                    rate.sleep()
                    continue

                # 1) Base command selection: ONNX → PD(lane_pose + bias) → CV lane (with bias)
                used_onnx = False
                controller = "none"
                v_cmd, omega_cmd = self.BASE_V, 0.0

                out = self.onnx_lane(img)
                if out is not None:
                    v_cmd, omega_cmd = out
                    used_onnx = True
                    controller = "onnx"
                else:
                    if (now() - t_pose) <= self.POSE_STALE_S and pose is not None:
                        v_cmd, omega_cmd = self.pd_lane(pose, desired_d=self.lane_bias)
                        controller = "lane_pose_pd"
                    else:
                        cv_out, cv_conf, dbg = self.cv_lane(img, bias_m=self.lane_bias)
                        if cv_out is not None:
                            v_cmd, omega_cmd = cv_out
                            controller = f"cv(conf={cv_conf:.2f})"
                        else:
                            v_cmd, omega_cmd = 0.15, 0.0
                            controller = "cv_none"

                        # publish debug overlay if enabled
                        if self.pub_dbg is not None and dbg is not None:
                            try:
                                ok, buf = cv2.imencode(".jpg", dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                                if ok:
                                    msg = CompressedImage()
                                    msg.format = "jpeg"
                                    msg.data = np.array(buf).tobytes()
                                    self.pub_dbg.publish(msg)
                            except Exception:
                                pass

                # 2) Environmental adaptation
                vis_fac = self.visibility_factor(img)
                if controller.startswith("cv"):
                    lc_fac = 1.0  # CV already embeds its own conf scaling
                else:
                    lc_fac = self.lane_confidence_factor(pose)
                v_cmd *= vis_fac * lc_fac

                # 3) Following heuristics
                v_cap_fd = self.following_cap()
                v_cmd = min(v_cmd, v_cap_fd)

                # 4) Stop line
                if self.stopline_triggered():
                    self.stop_until = now() + self.STOP_TIME_S
                if now() < self.stop_until:
                    v_cmd = 0.0; omega_cmd = 0.0

                # 5) Predictive slowdown (strings)
                v_cmd *= self.predictive_slowdown_factor()

                # 6) Roboflow obstacle slow/stop
                dets = []
                if self.use_rf and img is not None:
                    dets = self.rf_detect(img)
                    v_cap_rf = self.rf_speed_cap(dets, img.shape)
                    v_cmd = min(v_cmd, v_cap_rf)

                # 6b) Lane bias strategy for moderate obstacles
                if self.use_rf and img is not None:
                    # If v_cap_rf moderate (not zero and < MAX_V), bias away temporarily
                    try:
                        if 'v_cap_rf' in locals() and v_cap_rf < self.MAX_V and v_cap_rf > 0.0:
                            self.lane_bias_target = 0.04  # shift right by default
                        else:
                            self.lane_bias_target = 0.0
                    except Exception:
                        self.lane_bias_target = 0.0
                else:
                    self.lane_bias_target = 0.0
                self.update_lane_bias()

                # 7) Safety emergency
                if self.safety_emergency():
                    v_cmd = 0.0; omega_cmd = 0.0

                # 8) Curvature/IMU-based speed caps
                v_cmd *= self.curvature_speed_factor(omega_cmd)
                v_cmd *= self.imu_speed_factor()

                # 9) Accel smoothing
                v_prev = self.cmd_hist[-1][0] if self.cmd_hist else v_cmd
                max_dv = self.MAX_ACCEL * self.dt
                dv = clamp(v_cmd - v_prev, -max_dv, max_dv)
                v_cmd = v_prev + dv

                # EMA
                alpha = clamp(2.0 / (self.SMOOTH_WINDOW + 1.0), 0.05, 0.6)
                ema_v = alpha * v_cmd + (1 - alpha) * ema_v
                v_cmd = ema_v

                # Clamp
                v_cmd = float(clamp(v_cmd, 0.0 if v_cmd < 0.08 else self.MIN_V, self.MAX_V))
                omega_cmd = float(clamp(omega_cmd, -self.OMEGA_CLAMP, self.OMEGA_CLAMP))

                # 10) Publish
                self.publish_cmd(v_cmd, omega_cmd)
                self.cmd_hist.append((v_cmd, omega_cmd, now()))

                # 11) Log + Diagnostics
                self.maybe_log(controller=controller, used_onnx=used_onnx, v_cmd=v_cmd, omega_cmd=omega_cmd,
                               vis_fac=vis_fac, lane_conf_fac=lc_fac, fd_cap=v_cap_fd, rf_active=self.use_rf,
                               rf_last_n=len(dets), safety=safety, collision_risk=coll_risk,
                               stopped=(now() < self.stop_until), lane_bias=self.lane_bias)

                # Lightweight diagnostics
                try:
                    diag = {
                        "t": now(),
                        "mode": self.mode,
                        "state": self.state,
                        "loop_hz": self.rate_hz,
                        "lanebias": self.lane_bias,
                        "img_age": now() - self.t_img,
                        "pose_age": now() - t_pose,
                    }
                    self.pub_diag.publish(String(data=json.dumps(diag, separators=(",", ":"))))
                except Exception:
                    pass

            except Exception:
                rospy.logerr("[UMC] Loop error:\n" + traceback.format_exc())

            rate.sleep()

    def publish_cmd(self, v, omega):
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = float(v)
        cmd.omega = float(omega)
        self.pub_cmd.publish(cmd)

    def maybe_log(self, **kw):
        if self.log_rate.ready():
            s = json.dumps(kw, default=lambda o: str(o), separators=(",", ":"))
            self.pub_status.publish(String(data=s))
            rospy.loginfo(f"[UMC] {s}")

# ===== CLI =====
def parse_args(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Ultimate Monolithic Controller (v3, with CV fallback)")
    ap.add_argument("--veh", default=os.environ.get("VEHICLE_NAME", "duckie"), help="Vehicle name/namespace")
    ap.add_argument("--rate_hz", type=int, default=20, help="Main control loop rate")
    ap.add_argument("--onnx_model_path", type=str, default="", help="Path to ONNX lane-following model (optional)")
    ap.add_argument("--roboflow_model_id", type=str, default="", help="Roboflow model ID 'workspace/model/1' (optional)")
    ap.add_argument("--rf_period", type=float, default=0.33, help="Min seconds between Roboflow calls")
    ap.add_argument("--log_interval", type=float, default=1.0, help="Seconds between status logs")
    ap.add_argument("--debug_viz", action="store_true", help="Publish /umc/debug_image/compressed with overlays")
    return ap.parse_args(argv)

def main():
    args = parse_args(sys.argv[1:])
    rospy.init_node("ultimate_monolithic_controller", anonymous=False)
    ctrl = UMC(
        veh=args.veh,
        rate_hz=args.rate_hz,
        onnx_model_path=args.onnx_model_path if args.onnx_model_path else None,
        roboflow_model_id=args.roboflow_model_id if args.roboflow_model_id else None,
        rf_period=args.rf_period,
        log_interval=args.log_interval,
        debug_viz=args.debug_viz
    )
    ctrl.spin()

if __name__ == "__main__":
    main()