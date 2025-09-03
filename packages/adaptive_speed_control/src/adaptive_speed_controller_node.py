#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from threading import Lock
from typing import Dict, List, Optional, Tuple

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Twist2DStamped, LanePose, VehicleCorners, 
    AprilTagDetectionArray, StopLineReading
)
from sensor_msgs.msg import Image, Imu, CompressedImage
from std_msgs.msg import String, Header, Float32
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge

# Import custom message types (these would be generated from .msg files)
# For now, we'll use standard messages and add custom fields in dictionaries

from adaptive_speed_control.environmental_analyzer import EnvironmentalAnalyzer
from adaptive_speed_control.following_distance_controller import FollowingDistanceController
from adaptive_speed_control.acceleration_profile_manager import AccelerationProfileManager


class AdaptiveSpeedControllerNode(DTROS):
    """
    Adaptive Speed Controller Node for Duckietown
    
    Implements intelligent speed adjustment based on environmental conditions,
    traffic density, following distance, and safety constraints. Provides
    smooth acceleration profiles and integrates with the safety system.
    
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node
    
    Configuration:
        ~base_linear_velocity (:obj:`float`): Base linear velocity in m/s
        ~base_angular_velocity (:obj:`float`): Base angular velocity in rad/s
        ~enable_environmental_adaptation (:obj:`bool`): Enable environmental speed adaptation
        ~enable_following_distance (:obj:`bool`): Enable following distance control
        ~enable_smooth_acceleration (:obj:`bool`): Enable smooth acceleration profiles
        ~safety_integration_enabled (:obj:`bool`): Enable safety system integration
    
    Subscribers:
        ~lane_pose (:obj:`LanePose`): Current lane pose from lane filter
        ~camera_image (:obj:`CompressedImage`): Camera image for environmental analysis
        ~vehicle_detections (:obj:`VehicleCorners`): Detected vehicles for following distance
        ~apriltag_detections (:obj:`AprilTagDetectionArray`): AprilTag detections for intersections
        ~stop_line_reading (:obj:`StopLineReading`): Stop line detection for intersection handling
        ~imu_data (:obj:`Imu`): IMU data for road condition analysis
        ~collision_risk (:obj:`String`): Collision risk assessment from safety system
        ~safety_status (:obj:`String`): Safety system status
        ~base_speed_command (:obj:`Twist2DStamped`): Base speed command from lane controller
    
    Publishers:
        ~adaptive_speed_command (:obj:`Twist2DStamped`): Adaptive speed command output
        ~speed_constraints (:obj:`String`): Current speed constraints (as string for now)
        ~environmental_factors (:obj:`String`): Environmental analysis results
        ~following_status (:obj:`String`): Following distance status
        ~acceleration_profile (:obj:`String`): Acceleration profile information
    """
    
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(AdaptiveSpeedControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize components
        self.environmental_analyzer = EnvironmentalAnalyzer(self.config)
        self.following_controller = FollowingDistanceController(self.config)
        self.acceleration_manager = AccelerationProfileManager(self.config)
        
        # CV Bridge for image processing
        self.bridge = CvBridge()
        
        # State variables
        self.current_lane_pose = None
        self.current_image = None
        self.current_imu_data = None
        self.detected_vehicles = []
        self.apriltag_detections = []
        self.stop_line_reading = None
        self.collision_risk_level = "NONE"
        self.safety_status = "SAFE"
        self.base_speed_command = None
        
        # Current speed state
        self.current_velocity = 0.0
        self.current_angular_velocity = 0.0
        
        # Thread safety
        self.state_lock = Lock()
        
        # Performance monitoring
        self.performance_stats = {
            "total_commands": 0,
            "environmental_adjustments": 0,
            "following_adjustments": 0,
            "safety_overrides": 0,
            "emergency_stops": 0
        }
        
        # Initialize publishers
        self._setup_publishers()
        
        # Initialize subscribers
        self._setup_subscribers()
        
        # Start control loop
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.config.get("control", {}).get("update_rate", 20)),
            self.control_loop
        )
        
        self.log("Adaptive Speed Controller initialized")
    
    def _load_configuration(self) -> Dict:
        """Load configuration parameters from ROS parameter server."""
        config = {}
        
        # Base speed parameters
        config["base_linear_velocity"] = rospy.get_param("~base_linear_velocity", 0.3)
        config["base_angular_velocity"] = rospy.get_param("~base_angular_velocity", 2.0)
        config["min_linear_velocity"] = rospy.get_param("~min_linear_velocity", 0.05)
        config["max_linear_velocity"] = rospy.get_param("~max_linear_velocity", 0.6)
        
        # Feature enables
        config["enable_environmental_adaptation"] = rospy.get_param("~enable_environmental_adaptation", True)
        config["enable_following_distance"] = rospy.get_param("~enable_following_distance", True)
        config["enable_smooth_acceleration"] = rospy.get_param("~enable_smooth_acceleration", True)
        config["safety_integration_enabled"] = rospy.get_param("~safety_integration_enabled", True)
        
        # Load subsystem configurations
        config["visibility"] = rospy.get_param("~visibility", {})
        config["traffic_density"] = rospy.get_param("~traffic_density", {})
        config["road_conditions"] = rospy.get_param("~road_conditions", {})
        config["following_distance"] = rospy.get_param("~following_distance", {})
        config["intersection"] = rospy.get_param("~intersection", {})
        config["acceleration"] = rospy.get_param("~acceleration", {})
        config["emergency_braking"] = rospy.get_param("~emergency_braking", {})
        config["safety_integration"] = rospy.get_param("~safety_integration", {})
        config["control"] = rospy.get_param("~control", {})
        config["logging"] = rospy.get_param("~logging", {})
        
        return config
    
    def _setup_publishers(self):
        """Setup ROS publishers."""
        self.pub_speed_command = rospy.Publisher(
            "~adaptive_speed_command",
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_speed_constraints = rospy.Publisher(
            "~speed_constraints",
            String,
            queue_size=1
        )
        
        self.pub_environmental_factors = rospy.Publisher(
            "~environmental_factors",
            String,
            queue_size=1
        )
        
        self.pub_following_status = rospy.Publisher(
            "~following_status",
            String,
            queue_size=1
        )
        
        self.pub_acceleration_profile = rospy.Publisher(
            "~acceleration_profile",
            String,
            queue_size=1
        )
        
        self.pub_performance_stats = rospy.Publisher(
            "~performance_stats",
            String,
            queue_size=1
        )
    
    def _setup_subscribers(self):
        """Setup ROS subscribers."""
        self.sub_lane_pose = rospy.Subscriber(
            "~lane_pose",
            LanePose,
            self.cb_lane_pose,
            queue_size=1
        )
        
        self.sub_camera_image = rospy.Subscriber(
            "~camera_image",
            CompressedImage,
            self.cb_camera_image,
            queue_size=1
        )
        
        self.sub_vehicle_detections = rospy.Subscriber(
            "~vehicle_detections",
            VehicleCorners,
            self.cb_vehicle_detections,
            queue_size=1
        )
        
        self.sub_apriltag_detections = rospy.Subscriber(
            "~apriltag_detections",
            AprilTagDetectionArray,
            self.cb_apriltag_detections,
            queue_size=1
        )
        
        self.sub_stop_line_reading = rospy.Subscriber(
            "~stop_line_reading",
            StopLineReading,
            self.cb_stop_line_reading,
            queue_size=1
        )
        
        self.sub_imu_data = rospy.Subscriber(
            "~imu_data",
            Imu,
            self.cb_imu_data,
            queue_size=1
        )
        
        self.sub_collision_risk = rospy.Subscriber(
            "~collision_risk",
            String,
            self.cb_collision_risk,
            queue_size=1
        )
        
        self.sub_safety_status = rospy.Subscriber(
            "~safety_status",
            String,
            self.cb_safety_status,
            queue_size=1
        )
        
        self.sub_base_speed_command = rospy.Subscriber(
            "~base_speed_command",
            Twist2DStamped,
            self.cb_base_speed_command,
            queue_size=1
        )
    
    def cb_lane_pose(self, msg):
        """Callback for lane pose updates."""
        with self.state_lock:
            self.current_lane_pose = msg
    
    def cb_camera_image(self, msg):
        """Callback for camera image updates."""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            with self.state_lock:
                self.current_image = cv_image
                
        except Exception as e:
            rospy.logwarn(f"Error processing camera image: {e}")
    
    def cb_vehicle_detections(self, msg):
        """Callback for vehicle detection updates."""
        with self.state_lock:
            # Convert VehicleCorners to our internal format
            self.detected_vehicles = []
            
            for i, corner in enumerate(msg.corners):
                # Estimate distance based on corner position (simplified)
                distance = np.sqrt(corner.x**2 + corner.y**2)
                
                vehicle_data = {
                    "id": f"vehicle_{i}",
                    "position": {"x": corner.x, "y": corner.y, "z": 0.0},
                    "distance": distance,
                    "velocity": {"x": 0.0, "y": 0.0},  # Not available from VehicleCorners
                    "confidence": 0.8,  # Default confidence
                    "type": "duckiebot",
                    "bbox": {"x": corner.x, "y": corner.y, "width": 0.1, "height": 0.1}
                }
                self.detected_vehicles.append(vehicle_data)
    
    def cb_apriltag_detections(self, msg):
        """Callback for AprilTag detection updates."""
        with self.state_lock:
            self.apriltag_detections = msg.detections
    
    def cb_stop_line_reading(self, msg):
        """Callback for stop line reading updates."""
        with self.state_lock:
            self.stop_line_reading = msg
    
    def cb_imu_data(self, msg):
        """Callback for IMU data updates."""
        with self.state_lock:
            self.current_imu_data = {
                "accel_x": msg.linear_acceleration.x,
                "accel_y": msg.linear_acceleration.y,
                "accel_z": msg.linear_acceleration.z,
                "gyro_x": msg.angular_velocity.x,
                "gyro_y": msg.angular_velocity.y,
                "gyro_z": msg.angular_velocity.z
            }
    
    def cb_collision_risk(self, msg):
        """Callback for collision risk updates."""
        with self.state_lock:
            # Parse collision risk message (format: LEVEL:TTC:ACTION)
            try:
                parts = msg.data.split(":")
                self.collision_risk_level = parts[0]
            except (ValueError, IndexError):
                self.collision_risk_level = "UNKNOWN"
    
    def cb_safety_status(self, msg):
        """Callback for safety status updates."""
        with self.state_lock:
            # Parse safety status message
            try:
                if "SAFETY_LEVEL:" in msg.data:
                    parts = msg.data.split(":")
                    self.safety_status = parts[1]
                else:
                    self.safety_status = msg.data
            except (ValueError, IndexError):
                self.safety_status = "UNKNOWN"
    
    def cb_base_speed_command(self, msg):
        """Callback for base speed command updates."""
        with self.state_lock:
            self.base_speed_command = msg
            # Update current velocity estimate
            self.current_velocity = msg.v
            self.current_angular_velocity = msg.omega
    
    def control_loop(self, event):
        """Main control loop for adaptive speed control."""
        with self.state_lock:
            # Check if we have necessary data
            if self.base_speed_command is None:
                return
            
            # Get base speed command
            base_linear_velocity = self.base_speed_command.v
            base_angular_velocity = self.base_speed_command.omega
            
            # Update acceleration manager with current state
            self.acceleration_manager.update_current_state(self.current_velocity)
            
            # Calculate environmental speed adjustments
            environmental_factors = self._calculate_environmental_adjustments()
            
            # Calculate following distance adjustments
            following_adjustment, following_status = self._calculate_following_adjustments(base_linear_velocity)
            
            # Calculate intersection adjustments
            intersection_adjustment = self._calculate_intersection_adjustments()
            
            # Apply safety constraints
            safety_adjustment = self._apply_safety_constraints()
            
            # Combine all adjustments
            target_velocity = self._combine_speed_adjustments(
                base_linear_velocity,
                environmental_factors,
                following_adjustment,
                intersection_adjustment,
                safety_adjustment
            )
            
            # Check for emergency conditions
            emergency_stop = self._check_emergency_conditions()
            
            # Apply smooth acceleration profile
            final_velocity, acceleration_profile = self.acceleration_manager.calculate_smooth_speed_command(
                target_velocity, emergency_stop
            )
            
            # Create and publish speed command
            self._publish_speed_command(final_velocity, base_angular_velocity)
            
            # Publish status information
            self._publish_status_information(
                environmental_factors, following_status, acceleration_profile
            )
            
            # Update performance statistics
            self.performance_stats["total_commands"] += 1
    
    def _calculate_environmental_adjustments(self) -> Dict[str, float]:
        """Calculate speed adjustments based on environmental conditions."""
        if not self.config.get("enable_environmental_adaptation", True):
            return {"overall_environmental_factor": 1.0}
        
        if self.current_image is None:
            return {"overall_environmental_factor": 0.8}  # Conservative default
        
        try:
            environmental_factors = self.environmental_analyzer.get_environmental_speed_factor(
                self.current_image,
                self.detected_vehicles,
                self.current_imu_data
            )
            
            if environmental_factors["overall_environmental_factor"] < 1.0:
                self.performance_stats["environmental_adjustments"] += 1
            
            return environmental_factors
            
        except Exception as e:
            rospy.logwarn(f"Error calculating environmental adjustments: {e}")
            return {"overall_environmental_factor": 0.8}
    
    def _calculate_following_adjustments(self, base_velocity: float) -> Tuple[float, Dict]:
        """Calculate speed adjustments for following distance control."""
        if not self.config.get("enable_following_distance", True):
            return base_velocity, {"is_following": False}
        
        try:
            adjusted_velocity, following_status = self.following_controller.get_following_distance_command(
                self.current_velocity,
                self.detected_vehicles,
                base_velocity
            )
            
            if following_status["is_following"]:
                self.performance_stats["following_adjustments"] += 1
            
            return adjusted_velocity, following_status
            
        except Exception as e:
            rospy.logwarn(f"Error calculating following adjustments: {e}")
            return base_velocity, {"is_following": False}
    
    def _calculate_intersection_adjustments(self) -> float:
        """Calculate speed adjustments for intersection approach."""
        intersection_factor = 1.0
        
        # Check for stop line detection
        if self.stop_line_reading and self.stop_line_reading.stop_line_detected:
            distance_to_stop = np.sqrt(
                self.stop_line_reading.stop_line_point.x**2 + 
                self.stop_line_reading.stop_line_point.y**2
            )
            
            approach_distance = self.config.get("intersection", {}).get("approach_distance", 1.0)
            approach_speed = self.config.get("intersection", {}).get("approach_speed", 0.1)
            
            if distance_to_stop <= approach_distance:
                # Linear speed reduction as approaching stop line
                intersection_factor = max(
                    approach_speed / self.config.get("base_linear_velocity", 0.3),
                    distance_to_stop / approach_distance
                )
        
        # Check for AprilTag detections (intersection markers)
        if self.apriltag_detections:
            # Reduce speed when AprilTags are detected
            intersection_factor = min(intersection_factor, 0.7)
        
        return intersection_factor
    
    def _apply_safety_constraints(self) -> float:
        """Apply safety-based speed constraints."""
        if not self.config.get("safety_integration_enabled", True):
            return 1.0
        
        safety_factor = 1.0
        
        # Apply constraints based on collision risk
        if self.collision_risk_level == "CRITICAL":
            safety_factor = 0.0  # Emergency stop
            self.performance_stats["safety_overrides"] += 1
        elif self.collision_risk_level == "HIGH":
            safety_factor = 0.3  # Severe speed reduction
            self.performance_stats["safety_overrides"] += 1
        elif self.collision_risk_level == "MEDIUM":
            safety_factor = 0.6  # Moderate speed reduction
        elif self.collision_risk_level == "LOW":
            safety_factor = 0.8  # Slight speed reduction
        
        # Apply constraints based on safety status
        if self.safety_status == "CRITICAL":
            safety_factor = min(safety_factor, 0.0)
        elif self.safety_status == "DANGER":
            safety_factor = min(safety_factor, 0.3)
        elif self.safety_status == "WARNING":
            safety_factor = min(safety_factor, 0.6)
        
        return safety_factor
    
    def _combine_speed_adjustments(self, base_velocity: float,
                                 environmental_factors: Dict[str, float],
                                 following_velocity: float,
                                 intersection_factor: float,
                                 safety_factor: float) -> float:
        """Combine all speed adjustments into final target velocity."""
        
        # Start with base velocity
        target_velocity = base_velocity
        
        # Apply environmental factors
        env_factor = environmental_factors.get("overall_environmental_factor", 1.0)
        target_velocity *= env_factor
        
        # Apply following distance adjustment (this is already an absolute velocity)
        if following_velocity != base_velocity:
            target_velocity = following_velocity
        
        # Apply intersection factor
        target_velocity *= intersection_factor
        
        # Apply safety factor (most restrictive)
        target_velocity *= safety_factor
        
        # Ensure velocity is within bounds
        min_vel = self.config.get("min_linear_velocity", 0.05)
        max_vel = self.config.get("max_linear_velocity", 0.6)
        
        target_velocity = np.clip(target_velocity, 0.0, max_vel)
        
        # If velocity is very low but not zero, ensure it's above minimum
        if 0 < target_velocity < min_vel:
            target_velocity = min_vel
        
        return target_velocity
    
    def _check_emergency_conditions(self) -> bool:
        """Check if emergency stop conditions are met."""
        # Emergency stop for critical collision risk
        if self.collision_risk_level == "CRITICAL":
            self.performance_stats["emergency_stops"] += 1
            return True
        
        # Emergency stop for critical safety status
        if self.safety_status == "CRITICAL":
            self.performance_stats["emergency_stops"] += 1
            return True
        
        # Emergency stop for very close obstacles
        if self.detected_vehicles:
            min_distance = min(vehicle["distance"] for vehicle in self.detected_vehicles)
            emergency_distance = self.config.get("emergency_braking", {}).get("trigger_distance", 0.3)
            
            if min_distance < emergency_distance:
                self.performance_stats["emergency_stops"] += 1
                return True
        
        return False
    
    def _publish_speed_command(self, linear_velocity: float, angular_velocity: float):
        """Publish the final adaptive speed command."""
        speed_cmd = Twist2DStamped()
        speed_cmd.header.stamp = rospy.Time.now()
        speed_cmd.v = linear_velocity
        speed_cmd.omega = angular_velocity
        
        self.pub_speed_command.publish(speed_cmd)
    
    def _publish_status_information(self, environmental_factors: Dict,
                                  following_status: Dict,
                                  acceleration_profile: Dict):
        """Publish status and diagnostic information."""
        
        # Environmental factors
        env_msg = String()
        env_msg.data = f"visibility:{environmental_factors.get('visibility_factor', 1.0):.3f}," \
                      f"traffic:{environmental_factors.get('traffic_density_factor', 1.0):.3f}," \
                      f"road:{environmental_factors.get('road_condition_factor', 1.0):.3f}," \
                      f"overall:{environmental_factors.get('overall_environmental_factor', 1.0):.3f}"
        self.pub_environmental_factors.publish(env_msg)
        
        # Following status
        following_msg = String()
        following_msg.data = f"following:{following_status.get('is_following', False)}," \
                           f"distance:{following_status.get('current_distance', 0.0):.3f}," \
                           f"safety:{following_status.get('following_safety_factor', 1.0):.3f}"
        self.pub_following_status.publish(following_msg)
        
        # Acceleration profile
        accel_msg = String()
        accel_msg.data = f"target_v:{acceleration_profile.get('target_velocity', 0.0):.3f}," \
                        f"current_v:{acceleration_profile.get('current_velocity', 0.0):.3f}," \
                        f"accel:{acceleration_profile.get('acceleration', 0.0):.3f}," \
                        f"emergency:{acceleration_profile.get('emergency_braking', False)}"
        self.pub_acceleration_profile.publish(accel_msg)
        
        # Performance statistics
        stats_msg = String()
        stats_msg.data = f"total:{self.performance_stats['total_commands']}," \
                        f"env_adj:{self.performance_stats['environmental_adjustments']}," \
                        f"follow_adj:{self.performance_stats['following_adjustments']}," \
                        f"safety_override:{self.performance_stats['safety_overrides']}," \
                        f"emergency:{self.performance_stats['emergency_stops']}"
        self.pub_performance_stats.publish(stats_msg)


if __name__ == "__main__":
    # Create and run the adaptive speed controller node
    adaptive_speed_node = AdaptiveSpeedControllerNode("adaptive_speed_controller_node")
    rospy.spin()