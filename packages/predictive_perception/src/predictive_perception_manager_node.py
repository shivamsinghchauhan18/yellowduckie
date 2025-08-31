#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType

# Standard ROS messages
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point, Vector3

# Custom messages
from predictive_perception.msg import (
    PredictedTrajectory, 
    PredictedTrajectoryArray, 
    TrackingConfidence,
    TrackedObject
)

# Internal modules
from predictive_perception.multi_object_tracker import MultiObjectTracker
from nn_model.constants import IMAGE_SIZE
from nn_model.model import Wrapper


class PredictivePerceptionManagerNode(DTROS):
    """
    Predictive Perception Manager Node for advanced object tracking and trajectory prediction.
    
    This node enhances the basic object detection with:
    - Multi-object tracking using Kalman filters
    - Trajectory prediction for tracked objects
    - Confidence estimation and tracking statistics
    """
    
    def __init__(self, node_name):
        super(PredictivePerceptionManagerNode, self).__init__(
            node_name=node_name, 
            node_type=NodeType.PERCEPTION
        )
        
        # Initialize parameters
        self.veh = rospy.get_param("~veh", "duckiebot")
        self.max_disappeared = rospy.get_param("~max_disappeared", 30)
        self.max_distance = rospy.get_param("~max_distance", 50.0)
        self.prediction_horizon = rospy.get_param("~prediction_horizon", 3.0)
        self.prediction_dt = rospy.get_param("~prediction_dt", 0.1)
        self.publish_rate = rospy.get_param("~publish_rate", 10.0)
        
        # Detection thresholds
        self.duck_threshold = rospy.get_param("~duck_threshold", 0.7)
        self.duckiebot_threshold = rospy.get_param("~duckiebot_threshold", 0.7)
        
        # Initialize components
        self.bridge = CvBridge()
        self.model_wrapper = Wrapper(rospy.get_param("~AIDO_eval", False))
        self.tracker = MultiObjectTracker(
            max_disappeared=self.max_disappeared,
            max_distance=self.max_distance
        )
        
        # Timing
        self.last_update_time = rospy.Time.now()
        
        # Publishers
        self.pub_trajectories = rospy.Publisher(
            "~predicted_trajectories", 
            PredictedTrajectoryArray, 
            queue_size=1
        )
        self.pub_confidence = rospy.Publisher(
            "~tracking_confidence", 
            TrackingConfidence, 
            queue_size=1
        )
        self.pub_tracked_objects = rospy.Publisher(
            "~tracked_objects", 
            TrackedObject, 
            queue_size=10
        )
        self.pub_debug_image = rospy.Publisher(
            "~debug_image/compressed", 
            CompressedImage, 
            queue_size=1
        )
        
        # Subscribers
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_callback,
            buff_size=10000000,
            queue_size=1
        )
        
        # Timer for regular publishing
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), 
            self.publish_predictions
        )
        
        self.loginfo("Predictive Perception Manager initialized")
    
    def image_callback(self, image_msg):
        """
        Process incoming camera images for object detection and tracking.
        """
        try:
            # Convert image
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
            rgb = bgr[..., ::-1]
            rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
            
            # Run object detection
            bboxes, classes, scores = self.model_wrapper.predict(rgb)
            
            # Convert detections to tracker format
            detections = []
            confidences = []
            
            for bbox, cls, score in zip(bboxes, classes, scores):
                # Filter by confidence thresholds
                if cls == 0 and score >= self.duck_threshold:  # Duck
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    detections.append((center_x, center_y, cls))
                    confidences.append(score)
                elif cls == 1 and score >= self.duckiebot_threshold:  # Duckiebot
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    detections.append((center_x, center_y, cls))
                    confidences.append(score)
            
            # Calculate time delta
            current_time = rospy.Time.now()
            dt = (current_time - self.last_update_time).to_sec()
            self.last_update_time = current_time
            
            # Update tracker
            self.tracker.update(detections, confidences, dt)
            
            # Publish tracked objects
            self.publish_tracked_objects(image_msg.header)
            
            # Create debug visualization
            self.publish_debug_visualization(rgb, bboxes, classes, scores)
            
        except Exception as e:
            self.logerr(f"Error in image callback: {e}")
    
    def publish_tracked_objects(self, header):
        """
        Publish information about currently tracked objects.
        """
        valid_trackers = self.tracker.get_valid_trackers()
        
        for tracker_id, tracker in valid_trackers.items():
            msg = TrackedObject()
            msg.object_id = tracker.object_id
            msg.object_class = tracker.object_class
            
            # Position
            pos = tracker.get_position()
            msg.position = Point(x=pos[0], y=pos[1], z=0.0)
            
            # Velocity
            vel = tracker.get_velocity()
            msg.velocity = Vector3(x=vel[0], y=vel[1], z=0.0)
            
            # Acceleration (simplified for constant velocity model)
            acc = tracker.get_acceleration()
            msg.acceleration = Vector3(x=acc[0], y=acc[1], z=0.0)
            
            msg.confidence = tracker.confidence
            msg.last_seen_time = rospy.Time.now().to_sec()
            
            self.pub_tracked_objects.publish(msg)
    
    def publish_predictions(self, event):
        """
        Publish trajectory predictions and tracking confidence.
        """
        try:
            # Get trajectory predictions
            trajectories = self.tracker.predict_trajectories(
                time_horizon=self.prediction_horizon,
                dt=self.prediction_dt
            )
            
            # Create trajectory array message
            traj_array_msg = PredictedTrajectoryArray()
            traj_array_msg.header = Header()
            traj_array_msg.header.stamp = rospy.Time.now()
            traj_array_msg.header.frame_id = "base_link"
            
            valid_trackers = self.tracker.get_valid_trackers()
            
            for tracker_id, trajectory_points in trajectories.items():
                if tracker_id in valid_trackers:
                    tracker = valid_trackers[tracker_id]
                    
                    traj_msg = PredictedTrajectory()
                    traj_msg.header = traj_array_msg.header
                    traj_msg.object_id = tracker.object_id
                    traj_msg.confidence = tracker.confidence
                    traj_msg.time_horizon = self.prediction_horizon
                    
                    # Convert trajectory points
                    for point in trajectory_points:
                        traj_point = Point()
                        traj_point.x = point[0]
                        traj_point.y = point[1]
                        traj_point.z = 0.0
                        traj_msg.trajectory_points.append(traj_point)
                    
                    # Add velocity and acceleration
                    vel = tracker.get_velocity()
                    acc = tracker.get_acceleration()
                    traj_msg.velocity = Vector3(x=vel[0], y=vel[1], z=0.0)
                    traj_msg.acceleration = Vector3(x=acc[0], y=acc[1], z=0.0)
                    
                    traj_array_msg.trajectories.append(traj_msg)
            
            self.pub_trajectories.publish(traj_array_msg)
            
            # Publish tracking confidence
            stats = self.tracker.get_tracking_stats()
            confidence_msg = TrackingConfidence()
            confidence_msg.header = traj_array_msg.header
            confidence_msg.total_tracked_objects = stats['total_tracked_objects']
            confidence_msg.average_confidence = stats['average_confidence']
            confidence_msg.tracking_fps = self.publish_rate
            confidence_msg.lost_tracks_count = stats['lost_tracks_count']
            confidence_msg.new_tracks_count = stats['new_tracks_count']
            
            self.pub_confidence.publish(confidence_msg)
            
        except Exception as e:
            self.logerr(f"Error in publish_predictions: {e}")
    
    def publish_debug_visualization(self, rgb_image, bboxes, classes, scores):
        """
        Publish debug visualization with tracking information.
        """
        try:
            debug_image = rgb_image.copy()
            
            # Draw detection bounding boxes
            colors = {0: (255, 255, 0), 1: (255, 165, 0), 2: (0, 255, 0)}  # BGR
            names = {0: "duck", 1: "duckiebot", 2: "other"}
            
            for bbox, cls, score in zip(bboxes, classes, scores):
                if cls in [0, 1] and score >= 0.5:
                    pt1 = (int(bbox[0]), int(bbox[1]))
                    pt2 = (int(bbox[2]), int(bbox[3]))
                    color = colors.get(cls, (0, 255, 0))
                    
                    cv2.rectangle(debug_image, pt1, pt2, color, 2)
                    
                    label = f"{names.get(cls, 'unknown')}: {score:.2f}"
                    cv2.putText(debug_image, label, 
                               (pt1[0], pt1[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw tracker information
            valid_trackers = self.tracker.get_valid_trackers()
            for tracker_id, tracker in valid_trackers.items():
                pos = tracker.get_position()
                vel = tracker.get_velocity()
                
                # Draw tracker position
                center = (int(pos[0]), int(pos[1]))
                cv2.circle(debug_image, center, 5, (0, 0, 255), -1)
                
                # Draw velocity vector
                vel_end = (int(pos[0] + vel[0] * 10), int(pos[1] + vel[1] * 10))
                cv2.arrowedLine(debug_image, center, vel_end, (0, 0, 255), 2)
                
                # Draw tracker ID and confidence
                label = f"ID:{tracker_id} C:{tracker.confidence:.2f}"
                cv2.putText(debug_image, label,
                           (center[0] + 10, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Convert back to BGR and publish
            bgr_debug = debug_image[..., ::-1]
            debug_msg = self.bridge.cv2_to_compressed_imgmsg(bgr_debug)
            self.pub_debug_image.publish(debug_msg)
            
        except Exception as e:
            self.logerr(f"Error in debug visualization: {e}")


if __name__ == "__main__":
    node = PredictivePerceptionManagerNode("predictive_perception_manager_node")
    rospy.spin()