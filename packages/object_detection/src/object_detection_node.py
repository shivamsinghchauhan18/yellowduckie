#!/usr/bin/env python3

# Force CPU-only mode before any imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'

import cv2
import numpy as np
import time
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

try:
    import torch
    torch.cuda.is_available = lambda: False
    torch.backends.cudnn.enabled = False
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from .nn_model.model import ModelWrapper
except Exception:
    ModelWrapper = None

try:
    from .constants import IMAGE_SIZE
except Exception:
    IMAGE_SIZE = 480

class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        self.initialized = False
        self.frame_id = 0
        # --- log throttling state ---
        self._last_state = None
        self._last_state_log_ts = 0.0
        self._log_throttle_sec = rospy.get_param("~log_throttle_sec", 2.0)
        
        self.veh = os.environ['VEHICLE_NAME']
        self.pub_vel = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_detections_image = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1)
        
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )

        self.bridge = CvBridge()
        self.model = None
        if ModelWrapper is not None and TORCH_AVAILABLE:
            try:
                self.model = ModelWrapper(aido_eval=False)
                self.log("Model wrapper initialized successfully")
            except Exception as e:
                rospy.logwarn(f"[{self.node_name}] Model init failed: {e}")

        self.log("Initialized!")
        self.initialized = True

    def _log_state(self, state_str: str):
        """Log only when state changes or throttle interval passes."""
        now = time.monotonic()
        if (state_str != self._last_state) or (now - self._last_state_log_ts) >= self._log_throttle_sec:
            try:
                # Use node's logging helper if present
                self.log(state_str)
            except Exception:
                rospy.loginfo(f"[{self.node_name}] {state_str}")
            self._last_state = state_str
            self._last_state_log_ts = now

    def image_cb(self, msg: CompressedImage):
        if not self.initialized:
            return

        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(msg)
            rgb = bgr[..., ::-1]
        except Exception as e:
            rospy.logwarn(f"[{self.node_name}] cv_bridge conversion failed: {e}")
            return

        bboxes, classes = [], []
        if self.model is not None:
            try:
                bboxes, classes = self.model.infer(rgb)
            except Exception as e:
                rospy.logwarn(f"[{self.node_name}] Inference failed: {e}")

        # Simple policy: stop if a large duck or duckiebot is in front
        stop_signal = False
        large_duck = False
        large_duckiebot = False

        # Example heuristic on bbox area (this is placeholder logic)
        for clas, box in zip(classes, bboxes):
            x1, y1, x2, y2 = box
            area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            if clas == 0 and area > 5000:
                stop_signal = True
                large_duck = True
                self.log(f"Duck detected. Area: {area}")
            if clas == 1 and area > 8000:
                stop_signal = True
                large_duckiebot = True
                self.log(f"Duckiebot detected. Area: {area}")

        # Create velocity command
        vel_cmd = Twist2DStamped()
        vel_cmd.header.stamp = rospy.Time.now()
        
        if stop_signal:
            vel_cmd.v = 0.0
            vel_cmd.omega = 0.0
            if large_duck and not large_duckiebot:
                self._log_state("STOP_DUCK")
            elif large_duckiebot and not large_duck:
                self._log_state("STOP_DUCKIEBOT")
            else:
                self._log_state("STOP_BOTH")
        else:
            vel_cmd.v = 0.2
            vel_cmd.omega = 0.0
            self._log_state("DRIVING")
        
        self.pub_vel.publish(vel_cmd)

        self.visualize_detections(rgb, bboxes, classes)

    def visualize_detections(self, rgb, bboxes, classes):
        colors = {0: (0, 255, 255), 1: (0, 165, 255)}
        names = {0: "duckie", 1: "duckiebot"}
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for clas, box in zip(classes, bboxes):
            if clas in [0, 1]:
                pt1 = tuple(map(int, box[:2]))
                pt2 = tuple(map(int, box[2:]))
                color = tuple(reversed(colors[clas]))
                name = names[clas]
                rgb = cv2.rectangle(rgb, pt1, pt2, color, 2)
                text_location = (pt1[0], min(pt2[1] + 30, IMAGE_SIZE))
                rgb = cv2.putText(rgb, name, text_location, font, 0.6, color, 2, cv2.LINE_AA)

        # Publish overlay image (optional)
        bgr = rgb[..., ::-1]
        obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
        self.pub_detections_image.publish(obj_det_img)

if __name__ == "__main__":
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    rospy.spin()