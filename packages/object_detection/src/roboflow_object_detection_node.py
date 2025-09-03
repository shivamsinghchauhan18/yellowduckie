#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roboflow Object Detection Node
- Subscribes: /{veh}/camera_node/image/compressed
- Publishes:  ~enhanced_detections (std_msgs/String), ~image/compressed (sensor_msgs/CompressedImage)
API-pluggable and A/B testable alongside existing nodes.

Requires:
  pip install inference-sdk
"""
import os
import time
import json
import tempfile
import traceback
from typing import List

import cv2
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge

# Roboflow client (pip install inference-sdk)
try:
    from inference_sdk import InferenceHTTPClient
    ROBOFLOW_AVAILABLE = True
except Exception:
    ROBOFLOW_AVAILABLE = False


def draw_box(img, xyxy, color, label):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class RoboflowObjectDetectionNode(DTROS):
    def __init__(self, node_name):
        super(RoboflowObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.node_name = node_name
        self.bridge = CvBridge()

        # Params
        self.veh = os.environ.get("VEHICLE_NAME", rospy.get_param("~veh", "duckiebot"))
        self.api_url = rospy.get_param("~api_url", "https://serverless.roboflow.com")
        self.api_key = rospy.get_param("~api_key", os.environ.get("ROBOFLOW_API_KEY", ""))
        self.model_id = rospy.get_param("~model_id", "")
        self.confidence = float(rospy.get_param("~confidence", 0.25))
        self.max_fps = float(rospy.get_param("~max_fps", 4.0))
        self.enable_overlay = bool(rospy.get_param("~enable_overlay", True))
        self.timeout_s = float(rospy.get_param("~timeout_s", 3.5))

        self.last_call_ts = 0.0

        # Client
        self.client = None
        if not ROBOFLOW_AVAILABLE:
            rospy.logerr(f"[{self.node_name}] inference-sdk not available. Install via `pip install inference-sdk`.")
        else:
            if not self.api_key or not self.model_id:
                rospy.logwarn(f"[{self.node_name}] Missing api_key/model_id. Set ~api_key and ~model_id to enable detection.")
            else:
                try:
                    self.client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
                    rospy.loginfo(f"[{self.node_name}] Roboflow HTTP client ready (model_id={self.model_id}).")
                except Exception as e:
                    rospy.logerr(f"[{self.node_name}] Failed to init Roboflow client: {e}")
                    traceback.print_exc()

        # Pub/Sub
        cam_topic = f"/{self.veh}/camera_node/image/compressed"
        self.sub_img = rospy.Subscriber(cam_topic, CompressedImage, self.cb_image, queue_size=1, buff_size=2**22)
        self.pub_det_str = rospy.Publisher("~enhanced_detections", String, queue_size=1)
        self.pub_vis = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1)

        rospy.loginfo(f"[{self.node_name}] Initialized (veh={self.veh}).")

    def _throttle(self):
        if self.max_fps <= 0:
            return False
        now = time.time()
        if (now - self.last_call_ts) < (1.0 / self.max_fps):
            return True
        self.last_call_ts = now
        return False

    def _infer_roboflow(self, bgr: np.ndarray):
        if self.client is None:
            return []
        # Encode to JPEG in-memory
        success, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            return []
        # Roboflow SDK accepts bytes or file path; use temp file for wide compatibility
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tf:
            tf.write(enc.tobytes())
            tf.flush()
            try:
                res = self.client.infer(tf.name, model_id=self.model_id)
            except Exception as e:
                rospy.logwarn(f"[{self.node_name}] Roboflow HTTP error: {e}")
                return []
        # Expected structure: {'predictions': [ {x,y,width,height,class,confidence}, ... ]} (pixels)
        return res.get("predictions", []) if isinstance(res, dict) else []

    @staticmethod
    def _pred_to_bbox(pred, img_w, img_h):
        # Roboflow gives center x,y and w,h in pixels by default
        cx = float(pred.get("x", 0))
        cy = float(pred.get("y", 0))
        w = float(pred.get("width", 0))
        h = float(pred.get("height", 0))
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(img_w - 1, int(cx + w / 2))
        y2 = min(img_h - 1, int(cy + h / 2))
        return [x1, y1, x2, y2]

    @staticmethod
    def _format_string_msg(detections: List[dict]) -> str:
        """
        Keep compatibility with enhanced_object_detection_node's String formatting (human-readable key/vals).
        Example token: "class:duckie, conf:0.87, center:(123.4,56.7), area:999, priority:0.9"
        """
        parts = []
        for det in detections:
            parts.append(
                f"class:{det['class_name']}, "
                f"conf:{det['confidence']:.3f}, "
                f"center:({det['center'][0]:.1f},{det['center'][1]:.1f}), "
                f"area:{det['area']:.0f}, "
                f"priority:{det['priority']:.3f}"
            )
        return "; ".join(parts)

    def cb_image(self, msg: CompressedImage):
        if self._throttle():
            return

        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            rospy.logwarn(f"[{self.node_name}] cv_bridge error: {e}")
            return

        H, W = bgr.shape[:2]
        overlay = bgr.copy()

        preds = self._infer_roboflow(bgr)

        detections = []
        for p in preds:
            conf = float(p.get("confidence", 0.0))
            if conf < self.confidence:
                continue
            bbox = self._pred_to_bbox(p, W, H)
            cx = 0.5 * (bbox[0] + bbox[2])
            cy = 0.5 * (bbox[1] + bbox[3])
            area = max(0, (bbox[2] - bbox[0])) * max(0, (bbox[3] - bbox[1]))
            cls_name = str(p.get("class", "obj"))
            # Priority heuristic: centrality + confidence
            center_weight = 1.0 - min(1.0, abs((cx - W / 2.0) / (W / 2.0)))
            priority = float(conf) * (0.5 + 0.5 * center_weight)

            det = {
                "class_name": cls_name,
                "confidence": float(conf),
                "center": (float(cx), float(cy)),
                "area": float(area),
                "priority": float(priority),
                "bbox": bbox,
            }
            detections.append(det)

            if self.enable_overlay:
                draw_box(overlay, bbox, (0, 255, 255), f"{cls_name} {conf:.2f}")

        # Publish structured string compatible with existing consumer parsers
        det_msg = String()
        det_msg.data = self._format_string_msg(detections)
        self.pub_det_str.publish(det_msg)

        # Publish overlay image
        if self.enable_overlay:
            out = self.bridge.cv2_to_compressed_imgmsg(overlay)
            self.pub_vis.publish(out)


if __name__ == "__main__":
    rospy.init_node("roboflow_object_detection_node", anonymous=False)
    node = RoboflowObjectDetectionNode("roboflow_object_detection_node")
    rospy.on_shutdown(lambda: rospy.loginfo("[roboflow_object_detection_node] Shutting down."))
    rospy.spin()