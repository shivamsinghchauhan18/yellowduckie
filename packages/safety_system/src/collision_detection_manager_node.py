#!/usr/bin/env python3

import rospy
import numpy as np
import math
from threading import Lock
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, VehicleCorners, BoolStamped, LanePose
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point32


class CollisionRisk:
    """Custom message type for collision risk assessment"""
    def __init__(self):
        self.header = Header()
        self.risk_level = 0  # 0=NONE, 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL
        self.time_to_collision = -1.0  # seconds, -1 if no collision predicted
        self.collision_point = Point32()
        self.involved_objects = []  # object IDs
        self.recommended_action = 0  # 0=CONTINUE, 1=SLOW, 2=STOP, 3=EVADE
        self.confidence = 0.0  # 0.0 to 1.0


class CollisionDetectionManagerNode(DTROS):
    """
    Collision Detection Manager Node for Duckietown Safety System
    
    Implements multi-layered collision detection with distance-based and velocity-based
    risk assessment. Provides real-time collision risk evaluation and recommendations.
    
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node
    
    Configuration:
        ~min_safe_distance (:obj:`float`): Minimum safe distance in meters
        ~critical_distance (:obj:`float`): Critical distance threshold in meters
        ~time_horizon (:obj:`float`): Time horizon for collision prediction in seconds
        ~confidence_threshold (:obj:`float`): Minimum confidence for risk assessment
        ~update_rate (:obj:`float`): Risk assessment update rate in Hz
    
    Subscribers:
        ~vehicle_detections (:obj:`VehicleCorners`): Detected vehicles from vehicle detection
        ~object_detections (:obj:`String`): General object detections (simplified)
        ~car_cmd (:obj:`Twist2DStamped`): Current vehicle commands for velocity estimation
        /<veh>/lane_filter_node/lane_pose (:obj:`LanePose`): Current lane pose for position estimation
    
    Publishers:
        ~collision_risk (:obj:`String`): Current collision risk assessment
        ~risk_visualization (:obj:`String`): Risk visualization data for debugging
        ~collision_alerts (:obj:`String`): High-priority collision alerts
    """
    
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(CollisionDetectionManagerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )
        # Vehicle namespace for absolute topics
        self.veh = rospy.get_param("~veh", rospy.get_namespace().strip("/"))
        
        # Parameters
        self.min_safe_distance = DTParam(
            "~min_safe_distance",
            param_type=ParamType.FLOAT,
            default=0.3,
            min_value=0.1,
            max_value=2.0
        )
        
        self.critical_distance = DTParam(
            "~critical_distance",
            param_type=ParamType.FLOAT,
            default=0.15,
            min_value=0.05,
            max_value=1.0
        )
        
        self.time_horizon = DTParam(
            "~time_horizon",
            param_type=ParamType.FLOAT,
            default=3.0,
            min_value=1.0,
            max_value=10.0
        )
        
        self.confidence_threshold = DTParam(
            "~confidence_threshold",
            param_type=ParamType.FLOAT,
            default=0.7,
            min_value=0.1,
            max_value=1.0
        )
        
        self.update_rate = DTParam(
            "~update_rate",
            param_type=ParamType.FLOAT,
            default=20.0,
            min_value=5.0,
            max_value=50.0
        )
        
        # State variables
        self.detected_vehicles = []
        self.detected_objects = []
        self.current_velocity = Twist2DStamped()
        self.current_position = None
        self.last_risk_assessment = CollisionRisk()
        
        # Thread safety
        self.state_lock = Lock()
        
        # Risk level constants
        self.RISK_NONE = 0
        self.RISK_LOW = 1
        self.RISK_MEDIUM = 2
        self.RISK_HIGH = 3
        self.RISK_CRITICAL = 4
        
        # Action constants
        self.ACTION_CONTINUE = 0
        self.ACTION_SLOW = 1
        self.ACTION_STOP = 2
        self.ACTION_EVADE = 3
        
        # Publishers
        self.pub_collision_risk = rospy.Publisher(
            "~collision_risk",
            String,  # Would be CollisionRisk in full implementation
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION
        )
        
        self.pub_risk_visualization = rospy.Publisher(
            "~risk_visualization",
            String,
            queue_size=1
        )
        
        self.pub_collision_alerts = rospy.Publisher(
            "~collision_alerts",
            String,
            queue_size=1,
            dt_topic_type=TopicType.PERCEPTION
        )
        
        # Subscribers
        self.sub_vehicle_detections = rospy.Subscriber(
            "~vehicle_detections",
            VehicleCorners,
            self.cb_vehicle_detections,
            queue_size=1
        )
        
        self.sub_object_detections = rospy.Subscriber(
            "~object_detections",
            String,  # Simplified for now
            self.cb_object_detections,
            queue_size=1
        )
        
        self.sub_car_cmd = rospy.Subscriber(
            "~car_cmd",
            Twist2DStamped,
            self.cb_car_cmd,
            queue_size=1
        )
        
        self.sub_lane_pose = rospy.Subscriber(
            f"/{self.veh}/lane_filter_node/lane_pose",
            LanePose,
            self.cb_lane_pose,
            queue_size=1
        )
        
        # Start collision detection timer
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.update_rate.value),
            self.assess_collision_risk
        )
        
        self.log("Collision Detection Manager initialized")
    
    def cb_vehicle_detections(self, msg):
        """
        Callback for vehicle detection messages
        
        Args:
            msg (:obj:`VehicleCorners`): Detected vehicle information
        """
        with self.state_lock:
            if msg.detection.data:
                # Process vehicle detection
                vehicle_info = {
                    'timestamp': msg.header.stamp,
                    'corners': msg.corners,
                    'confidence': 1.0,  # Assume high confidence for detected vehicles
                    'distance': self.estimate_vehicle_distance(msg.corners),
                    'relative_position': self.estimate_relative_position(msg.corners)
                }
                
                # Update detected vehicles list (keep only recent detections)
                self.detected_vehicles = [v for v in self.detected_vehicles 
                                        if (msg.header.stamp - v['timestamp']).to_sec() < 1.0]
                self.detected_vehicles.append(vehicle_info)
    
    def cb_object_detections(self, msg):
        """
        Callback for general object detection messages
        
        Args:
            msg (:obj:`String`): Object detection information (simplified)
        """
        with self.state_lock:
            # In full implementation, would process object detection messages
            # For now, simulate object detection
            if "OBSTACLE" in msg.data:
                object_info = {
                    'timestamp': rospy.Time.now(),
                    'type': 'obstacle',
                    'distance': 0.5,  # Simulated distance
                    'confidence': 0.8
                }
                
                # Update detected objects list
                current_time = rospy.Time.now()
                self.detected_objects = [o for o in self.detected_objects 
                                       if (current_time - o['timestamp']).to_sec() < 1.0]
                self.detected_objects.append(object_info)
    
    def cb_car_cmd(self, msg):
        """
        Callback for current vehicle commands
        
        Args:
            msg (:obj:`Twist2DStamped`): Current vehicle command
        """
        with self.state_lock:
            self.current_velocity = msg
    
    def cb_lane_pose(self, msg: LanePose):
        """
        Callback for current lane pose
        
        Args:
            msg (:obj:`LanePose`): Lane pose information
        """
        with self.state_lock:
            # Store only fields we might use downstream
            self.current_position = {
                'd': msg.d,
                'phi': msg.phi,
                'sigma_d': msg.sigma_d,
                'sigma_phi': msg.sigma_phi,
                'in_lane': msg.in_lane
            }
    
    def estimate_vehicle_distance(self, corners):
        """
        Estimate distance to detected vehicle based on corner positions
        
        Args:
            corners: List of corner points
            
        Returns:
            float: Estimated distance in meters
        """
        if not corners:
            return float('inf')
        
        # Simple distance estimation based on corner spread
        # In full implementation, would use camera calibration and geometry
        corner_spread = max([c.x for c in corners]) - min([c.x for c in corners])
        
        # Empirical relationship: larger spread = closer vehicle
        if corner_spread > 100:
            return 0.2  # Very close
        elif corner_spread > 50:
            return 0.5  # Close
        elif corner_spread > 20:
            return 1.0  # Medium distance
        else:
            return 2.0  # Far
    
    def estimate_relative_position(self, corners):
        """
        Estimate relative position of detected vehicle
        
        Args:
            corners: List of corner points
            
        Returns:
            dict: Relative position information
        """
        if not corners:
            return {'x': 0, 'y': 0}
        
        # Calculate center of detected pattern
        center_x = sum([c.x for c in corners]) / len(corners)
        center_y = sum([c.y for c in corners]) / len(corners)
        
        # Convert to relative position (simplified)
        # In full implementation, would use proper camera-to-world transformation
        return {
            'x': (center_x - 320) / 320.0,  # Normalized x position
            'y': (center_y - 240) / 240.0   # Normalized y position
        }
    
    def assess_collision_risk(self, event):
        """
        Main collision risk assessment function
        
        Args:
            event: Timer event
        """
        with self.state_lock:
            risk_assessment = CollisionRisk()
            risk_assessment.header.stamp = rospy.Time.now()
            
            # Initialize risk as none
            risk_assessment.risk_level = self.RISK_NONE
            risk_assessment.recommended_action = self.ACTION_CONTINUE
            risk_assessment.confidence = 1.0
            
            max_risk_level = self.RISK_NONE
            min_time_to_collision = float('inf')
            
            # Assess risk from detected vehicles
            for vehicle in self.detected_vehicles:
                vehicle_risk = self.assess_vehicle_risk(vehicle)
                if vehicle_risk['risk_level'] > max_risk_level:
                    max_risk_level = vehicle_risk['risk_level']
                    risk_assessment.time_to_collision = vehicle_risk['time_to_collision']
                    risk_assessment.collision_point.x = vehicle_risk['collision_point']['x']
                    risk_assessment.collision_point.y = vehicle_risk['collision_point']['y']
                
                if vehicle_risk['time_to_collision'] < min_time_to_collision:
                    min_time_to_collision = vehicle_risk['time_to_collision']
            
            # Assess risk from detected objects
            for obj in self.detected_objects:
                object_risk = self.assess_object_risk(obj)
                if object_risk['risk_level'] > max_risk_level:
                    max_risk_level = object_risk['risk_level']
                    risk_assessment.time_to_collision = object_risk['time_to_collision']
            
            # Set final risk assessment
            risk_assessment.risk_level = max_risk_level
            risk_assessment.recommended_action = self.determine_recommended_action(max_risk_level)
            
            # Store and publish risk assessment
            self.last_risk_assessment = risk_assessment
            self.publish_risk_assessment(risk_assessment)
    
    def assess_vehicle_risk(self, vehicle_info):
        """
        Assess collision risk for a specific vehicle
        
        Args:
            vehicle_info (dict): Vehicle detection information
            
        Returns:
            dict: Risk assessment for this vehicle
        """
        distance = vehicle_info['distance']
        
        # Distance-based risk assessment
        if distance < self.critical_distance.value:
            risk_level = self.RISK_CRITICAL
        elif distance < self.min_safe_distance.value:
            risk_level = self.RISK_HIGH
        elif distance < self.min_safe_distance.value * 1.5:
            risk_level = self.RISK_MEDIUM
        elif distance < self.min_safe_distance.value * 2.0:
            risk_level = self.RISK_LOW
        else:
            risk_level = self.RISK_NONE
        
        # Velocity-based time to collision
        current_speed = abs(self.current_velocity.v) if self.current_velocity else 0.0
        if current_speed > 0.01:  # Avoid division by zero
            time_to_collision = distance / current_speed
        else:
            time_to_collision = float('inf')
        
        # Adjust risk based on time to collision
        if time_to_collision < 1.0 and risk_level < self.RISK_HIGH:
            risk_level = self.RISK_HIGH
        elif time_to_collision < 2.0 and risk_level < self.RISK_MEDIUM:
            risk_level = self.RISK_MEDIUM
        
        return {
            'risk_level': risk_level,
            'time_to_collision': time_to_collision,
            'collision_point': vehicle_info['relative_position']
        }
    
    def assess_object_risk(self, object_info):
        """
        Assess collision risk for a detected object
        
        Args:
            object_info (dict): Object detection information
            
        Returns:
            dict: Risk assessment for this object
        """
        distance = object_info['distance']
        
        # Similar risk assessment as vehicles but more conservative
        if distance < self.critical_distance.value * 1.2:
            risk_level = self.RISK_CRITICAL
        elif distance < self.min_safe_distance.value * 1.2:
            risk_level = self.RISK_HIGH
        elif distance < self.min_safe_distance.value * 2.0:
            risk_level = self.RISK_MEDIUM
        else:
            risk_level = self.RISK_LOW
        
        # Time to collision calculation
        current_speed = abs(self.current_velocity.v) if self.current_velocity else 0.0
        if current_speed > 0.01:
            time_to_collision = distance / current_speed
        else:
            time_to_collision = float('inf')
        
        return {
            'risk_level': risk_level,
            'time_to_collision': time_to_collision
        }
    
    def determine_recommended_action(self, risk_level):
        """
        Determine recommended action based on risk level
        
        Args:
            risk_level (int): Current risk level
            
        Returns:
            int: Recommended action
        """
        if risk_level >= self.RISK_CRITICAL:
            return self.ACTION_STOP
        elif risk_level >= self.RISK_HIGH:
            return self.ACTION_SLOW
        elif risk_level >= self.RISK_MEDIUM:
            return self.ACTION_SLOW
        else:
            return self.ACTION_CONTINUE
    
    def publish_risk_assessment(self, risk_assessment):
        """
        Publish collision risk assessment
        
        Args:
            risk_assessment (CollisionRisk): Risk assessment to publish
        """
        # Convert risk level to string for simplified message
        risk_levels = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        actions = ["CONTINUE", "SLOW", "STOP", "EVADE"]
        
        risk_msg = String()
        risk_msg.data = f"{risk_levels[risk_assessment.risk_level]}:{risk_assessment.time_to_collision:.2f}:{actions[risk_assessment.recommended_action]}"
        
        self.pub_collision_risk.publish(risk_msg)
        
        # Publish high-priority alerts for critical risks
        if risk_assessment.risk_level >= self.RISK_HIGH:
            alert_msg = String()
            alert_msg.data = f"COLLISION_ALERT:{risk_levels[risk_assessment.risk_level]}:TTC_{risk_assessment.time_to_collision:.2f}"
            self.pub_collision_alerts.publish(alert_msg)
        
        # Publish visualization data
        viz_msg = String()
        viz_msg.data = f"RISK_VIZ:{risk_assessment.risk_level}:{len(self.detected_vehicles)}:{len(self.detected_objects)}"
        self.pub_risk_visualization.publish(viz_msg)
    
    def get_current_risk_level(self):
        """
        Get current collision risk level
        
        Returns:
            int: Current risk level
        """
        with self.state_lock:
            return self.last_risk_assessment.risk_level


if __name__ == "__main__":
    # Create and run the collision detection manager node
    collision_detection_node = CollisionDetectionManagerNode("collision_detection_manager_node")
    rospy.spin()