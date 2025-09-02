#!/usr/bin/env python3

# Force CPU-only mode before any imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'

import cv2
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

try:
    from nn_model.constants import IMAGE_SIZE
    from nn_model.model import Wrapper
    from solution.integration_activity import NUMBER_FRAMES_SKIPPED
except Exception as e:
    rospy.logwarn(f"Optional deps not available ({e}); running in passthrough mode.")
    IMAGE_SIZE = 416
    def NUMBER_FRAMES_SKIPPED():
        return 0
    class Wrapper:
        def __init__(self, *_args, **_kwargs):
            self.model = None
        def predict(self, _img):
            return [], [], []

class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        self.initialized = False
        self.frame_id = 0
        
        self.veh = os.environ['VEHICLE_NAME']
        self.pub_vel = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_detections_image = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1)
        
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1,
        )

        self.bridge = CvBridge()
        
        try:
            self.model_wrapper = Wrapper(rospy.get_param("~AIDO_eval", False))
            self.log("Model wrapper initialized successfully")
        except Exception as e:
            self.logwarn(f"Model wrapper unavailable: {e}. Running without detection.")
            self.model_wrapper = None
        
        self.initialized = True
        self.log("Initialized!")

    def image_cb(self, image_msg):
        if not self.initialized:
            return

        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED())
        if self.frame_id != 0:
            return

        try:
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr(f"Could not decode image: {e}")
            return

        rgb = bgr[..., ::-1]
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))

        # Skip object detection if model wrapper failed to initialize
        if self.model_wrapper is None:
            # Publish default driving command
            vel_cmd = Twist2DStamped()
            vel_cmd.header.stamp = rospy.Time.now()
            vel_cmd.v = 0.2
            vel_cmd.omega = 0.0
            self.pub_vel.publish(vel_cmd)
            return

        try:
            bboxes, classes, scores = self.model_wrapper.predict(rgb)
        except Exception as e:
            self.logerr(f"Model prediction failed: {e}")
            # Publish safe stop command on prediction failure
            vel_cmd = Twist2DStamped()
            vel_cmd.header.stamp = rospy.Time.now()
            vel_cmd.v = 0.0
            vel_cmd.omega = 0.0
            self.pub_vel.publish(vel_cmd)
            return

        # Stop logic for ducks and duckiebots
        stop_signal = False
        large_duck = False
        large_duckiebot = False

        # Define the left and right boundaries of the center region
        left_boundary = int(IMAGE_SIZE * 0.33)
        right_boundary = int(IMAGE_SIZE * 0.75)

        for cls, bbox, score in zip(classes, bboxes, scores):

            # Calculate center of bounding box
            center_x = (bbox[0] + bbox[2]) / 2

            if cls == 0 and score > 0.7:  # Duck
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > 2000 and left_boundary < center_x < right_boundary:
                    stop_signal = True
                    large_duck = True
                    self.log(f"Duck detected. Area: {area}")

            if cls == 1 and score > 0.7:  # Duckiebot
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > 10000 and left_boundary < center_x < right_boundary:
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
                self.log("Stopping for Duck.")
            elif large_duckiebot and not large_duck:
                self.log("Stopping for Duckiebot.")
            else:
                self.log("Stopping for duck and duckiebot.")
        else:
            vel_cmd.v = 0.2
            vel_cmd.omega = 0.0
            self.log("Driving...")
        
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
                rgb = cv2.putText(rgb, name, text_location, font, 1, color, thickness=2)
            
        bgr = rgb[..., ::-1]
        obj_det_img = self.bridge.cv2_to_compressed_imgmsg(bgr)
        self.pub_detections_image.publish(obj_det_img)

if __name__ == "__main__":
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    rospy.spin()