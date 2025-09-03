#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaneFollowingONNXNode
- Subscribes: /{veh}/camera_node/image/compressed
- Publishes:  ~lane_pose (duckietown_msgs/LanePose)
Drop-in source to remap lane_controller_node's ~lane_pose input.
"""
import os
import math
import time
import traceback
import numpy as np
import cv2
import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import LanePose
from cv_bridge import CvBridge

# Optional dependency: onnxruntime (CPU provider by default)
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False


def focal_px_from_fov_deg(img_w: int, fov_deg: float) -> float:
    """Approximate focal length in pixels from horizontal FOV degrees."""
    fov_rad = math.radians(fov_deg)
    return (img_w * 0.5) / math.tan(0.5 * fov_rad)


class LaneFollowingONNXNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowingONNXNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.node_name = node_name
        self.bridge = CvBridge()

        # Parameters
        self.veh = os.environ.get("VEHICLE_NAME", rospy.get_param("~veh", "duckiebot"))
        self.model_path = rospy.get_param("~onnx_model_path", "")
        self.input_size = int(rospy.get_param("~input_size", 416))           # square input size
        self.postprocess_mode = rospy.get_param("~postprocess_mode", "seg_centerline")  # or "angle"
        self.fov_deg = float(rospy.get_param("~fov_deg", 80.0))
        self.meters_per_pixel = float(rospy.get_param("~meters_per_pixel", 0.0020))
        self.invert_d = bool(rospy.get_param("~invert_d", False))
        self.invert_phi = bool(rospy.get_param("~invert_phi", False))
        self.alpha_ema = float(rospy.get_param("~ema_alpha", 0.6))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 15.0))

        self.last_pub_ts = 0.0
        self._ema_d = None
        self._ema_phi = None

        # ONNX session init
        self.session = None
        self.input_name = None
        self.expected_layout = rospy.get_param("~input_layout", "NCHW")  # or "NHWC"
        self.normalize = bool(rospy.get_param("~normalize_01", True))

        if not ORT_AVAILABLE:
            rospy.logerr(f"[{self.node_name}] onnxruntime not available. Install `onnxruntime`.")
        else:
            if not self.model_path or not os.path.exists(self.model_path):
                rospy.logerr(f"[{self.node_name}] ONNX model not found at '{self.model_path}'. Set ~onnx_model_path.")
            else:
                try:
                    providers = ['CPUExecutionProvider']
                    self.session = ort.InferenceSession(self.model_path, providers=providers)
                    self.input_name = self.session.get_inputs()[0].name
                    rospy.loginfo(f"[{self.node_name}] Loaded ONNX model: {os.path.basename(self.model_path)}")
                except Exception as e:
                    rospy.logerr(f"[{self.node_name}] Failed to load ONNX model: {e}")
                    traceback.print_exc()

        # Topics
        cam_topic = f"/{self.veh}/camera_node/image/compressed"
        self.sub_img = rospy.Subscriber(cam_topic, CompressedImage, self.cb_image, queue_size=1, buff_size=2**22)
        self.pub_pose = rospy.Publisher("~lane_pose", LanePose, queue_size=1)

        rospy.loginfo(f"[{self.node_name}] Initialized (veh={self.veh}, postprocess={self.postprocess_mode}).")

    def _preprocess(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        if self.normalize:
            inp = resized.astype(np.float32) / 255.0
        else:
            inp = resized.astype(np.float32)
        if self.expected_layout.upper() == "NCHW":
            inp = np.transpose(inp, (2, 0, 1))[None, ...]
        else:  # NHWC
            inp = inp[None, ...]
        return inp

    def _postprocess_to_lane_pose(self, outputs, rgb):
        """
        - seg_centerline: expects a single-channel lane probability map (any reasonable shape).
        - angle: expects first two values [phi_rad, d_m] directly from model.
        """
        H, W, _ = rgb.shape
        f_px = focal_px_from_fov_deg(W, self.fov_deg)

        if self.postprocess_mode == "angle":
            arr = None
            for out in outputs:
                o = np.array(out) if not isinstance(out, np.ndarray) else out
                if o.size >= 2:
                    arr = o.reshape(-1)
                    break
            if arr is None:
                raise RuntimeError("No usable outputs for 'angle' mode.")
            phi = float(arr[0])
            d = float(arr[1])
        else:
            # Try to normalize an output to (H, W) probability map
            prob = None
            for out in outputs:
                o = np.array(out) if not isinstance(out, np.ndarray) else out
                if o.ndim == 4:               # (N,C,H,W) or (N,1,H,W)
                    p = o[0]
                    prob = p.max(axis=0) if p.shape[0] > 1 else p[0]
                elif o.ndim == 3:             # (C,H,W) or (H,W,1)
                    prob = o.max(axis=0) if o.shape[0] > 1 else (o[0] if o.shape[0] == 1 else None)
                    if prob is None and o.shape[2] == 1:
                        prob = o[..., 0]
                elif o.ndim == 2:             # (H,W)
                    prob = o
                if prob is not None:
                    break
            if prob is None:
                raise RuntimeError("Could not interpret ONNX outputs as segmentation probability.")

            prob = prob.astype(np.float32)
            prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)

            # Bottom ROI
            h_bot = max(10, H // 3)
            roi = prob[H - h_bot : H, :]

            cols = np.arange(W, dtype=np.float32)
            xs, ys = [], []
            for i in range(roi.shape[0]):
                row = roi[i]
                if row.max() < 0.2:
                    continue
                x = float((row * cols).sum() / (row.sum() + 1e-6))  # weighted center
                xs.append(x)
                ys.append(H - h_bot + i)

            if len(xs) < 5:
                phi, d = 0.0, 0.0
            else:
                xs = np.array(xs)
                ys = np.array(ys)
                A = np.vstack([ys, np.ones_like(ys)]).T
                a, b = np.linalg.lstsq(A, xs, rcond=None)[0]
                x_bottom = a * (H - 1) + b
                d_pixels = x_bottom - (W / 2.0)
                d = float(d_pixels * self.meters_per_pixel)
                dy = max(1.0, H * 0.25)
                x_top = a * (H - 1 - dy) + b
                dx = x_bottom - x_top
                phi = math.atan2(dx, f_px)

        if self.invert_d:
            d = -d
        if self.invert_phi:
            phi = -phi

        # EMA smoothing
        if self._ema_d is None:
            self._ema_d, self._ema_phi = d, phi
        else:
            self._ema_d = self.alpha_ema * d + (1 - self.alpha_ema) * self._ema_d
            self._ema_phi = self.alpha_ema * phi + (1 - self.alpha_ema) * self._ema_phi

        return float(self._ema_d), float(self._ema_phi)

    def cb_image(self, msg: CompressedImage):
        now = time.time()
        if self.publish_rate_hz > 0 and (now - self.last_pub_ts) < (1.0 / self.publish_rate_hz):
            return

        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            rospy.logwarn(f"[{self.node_name}] cv_bridge error: {e}")
            return

        if self.session is None:
            d, phi = 0.0, 0.0
        else:
            try:
                inp = self._preprocess(bgr)
                outputs = self.session.run(None, {self.input_name: inp})
            except Exception as e:
                rospy.logwarn(f"[{self.node_name}] ONNX inference failed: {e}")
                outputs = []
            try:
                d, phi = self._postprocess_to_lane_pose(outputs, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            except Exception as e:
                rospy.logwarn(f"[{self.node_name}] Postprocess failed: {e}")
                d, phi = 0.0, 0.0

        lp = LanePose()
        lp.header.stamp = rospy.Time.now()
        lp.header.frame_id = f"{self.veh}/camera"
        lp.d = d
        lp.phi = phi
        lp.status = 0
        self.pub_pose.publish(lp)
        self.last_pub_ts = now


if __name__ == "__main__":
    rospy.init_node("lane_following_onnx_node", anonymous=False)
    node = LaneFollowingONNXNode("lane_following_onnx_node")
    rospy.on_shutdown(lambda: rospy.loginfo("[lane_following_onnx_node] Shutting down."))
    rospy.spin()