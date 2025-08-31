#!/usr/bin/env python3

import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType

# Standard ROS messages
from std_msgs.msg import Header, String, Float32
from sensor_msgs.msg import Imu, CompressedImage
from geometry_msgs.msg import Twist2DStamped

# Internal modules
from predictive_perception.sensor_fusion import (
    SensorFusionEngine, 
    CameraData, 
    IMUData, 
    EncoderData,
    FusedPerceptionData
)


class PerceptionDataFusionNode(DTROS):
    """
    Perception Data Fusion Node for combining camera, IMU, and encoder data.
    
    This node provides:
    - Multi-modal sensor fusion
    - Temporal consistency checks
    - Outlier detection and filtering
    - Unified perception output with confidence metrics
    """
    
    def __init__(self, node_name):
        super(PerceptionDataFusionNode, self).__init__(
            node_name=node_name, 
            node_type=NodeType.PERCEPTION
        )
        
        # Initialize parameters
        self.veh = rospy.get_param("~veh", "duckiebot")
        self.fusion_rate = rospy.get_param("~fusion_rate", 20.0)
        self.fusion_window = rospy.get_param("~fusion_window", 0.1)
        self.max_data_age = rospy.get_param("~max_data_age", 1.0)
        
        # Sensor weights
        self.camera_weight = rospy.get_param("~camera_weight", 0.6)
        self.imu_weight = rospy.get_param("~imu_weight", 0.3)
        self.encoder_weight = rospy.get_param("~encoder_weight", 0.1)
        
        # Initialize fusion engine
        self.fusion_engine = SensorFusionEngine(
            fusion_window=self.fusion_window,
            max_data_age=self.max_data_age
        )
        
        # Set fusion weights
        self.fusion_engine.camera_weight = self.camera_weight
        self.fusion_engine.imu_weight = self.imu_weight
        self.fusion_engine.encoder_weight = self.encoder_weight
        
        # Data storage
        self.last_encoder_ticks = {'left': 0, 'right': 0}
        self.last_encoder_time = None
        
        # Publishers
        self.pub_fused_perception = rospy.Publisher(
            "~fused_perception", 
            String,  # Would be custom FusedPerceptionData message in full implementation
            queue_size=1
        )
        self.pub_confidence_metrics = rospy.Publisher(
            "~confidence_metrics", 
            String, 
            queue_size=1
        )
        self.pub_sensor_health = rospy.Publisher(
            "~sensor_health", 
            String, 
            queue_size=1
        )
        self.pub_ego_motion = rospy.Publisher(
            "~ego_motion", 
            Twist2DStamped, 
            queue_size=1
        )
        
        # Subscribers
        
        # Camera data (enhanced detections from object detection)
        self.sub_camera_detections = rospy.Subscriber(
            f"/{self.veh}/enhanced_object_detection_node/enhanced_detections",
            String,
            self.camera_detections_callback,
            queue_size=1
        )
        
        # Lane pose from lane filter
        self.sub_lane_pose = rospy.Subscriber(
            f"/{self.veh}/lane_filter_node/lane_pose",
            String,  # Placeholder - would be LanePose message
            self.lane_pose_callback,
            queue_size=1
        )
        
        # IMU data
        self.sub_imu = rospy.Subscriber(
            f"/{self.veh}/imu_node/data",
            Imu,
            self.imu_callback,
            queue_size=10
        )
        
        # Wheel encoder data (simulated from wheel commands)
        self.sub_wheel_encoders = rospy.Subscriber(
            f"/{self.veh}/wheel_encoder_node/ticks",
            String,  # Placeholder - would be WheelEncoderStamped message
            self.encoder_callback,
            queue_size=10
        )
        
        # Car command for encoder simulation
        self.sub_car_cmd = rospy.Subscriber(
            f"/{self.veh}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            self.car_cmd_callback,
            queue_size=1
        )
        
        # Timer for fusion processing
        self.fusion_timer = rospy.Timer(
            rospy.Duration(1.0 / self.fusion_rate), 
            self.fusion_timer_callback
        )
        
        # Storage for latest sensor data
        self.latest_detections = []
        self.latest_lane_pose = None
        self.latest_car_cmd = None
        
        self.loginfo("Perception Data Fusion Node initialized")
    
    def camera_detections_callback(self, msg):
        """Process camera detection data."""
        try:
            # Parse detection data from string message
            # In full implementation, would parse custom DetectionArray message
            detections = self.parse_detection_string(msg.data)
            
            # Create camera data object
            camera_data = CameraData(
                timestamp=rospy.Time.now().to_sec(),
                detections=detections,
                lane_pose=self.latest_lane_pose,
                confidence=self.calculate_camera_confidence(detections)
            )
            
            # Add to fusion engine
            self.fusion_engine.add_camera_data(camera_data)
            
        except Exception as e:
            self.logerr(f"Error in camera detections callback: {e}")
    
    def lane_pose_callback(self, msg):
        """Process lane pose data."""
        try:
            # Store latest lane pose
            self.latest_lane_pose = msg.data
            
        except Exception as e:
            self.logerr(f"Error in lane pose callback: {e}")
    
    def imu_callback(self, msg):
        """Process IMU data."""
        try:
            # Extract IMU data
            angular_velocity = [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ]
            
            linear_acceleration = [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ]
            
            # Create IMU data object
            imu_data = IMUData(
                timestamp=msg.header.stamp.to_sec(),
                angular_velocity=angular_velocity,
                linear_acceleration=linear_acceleration,
                confidence=self.calculate_imu_confidence(msg)
            )
            
            # Add to fusion engine
            self.fusion_engine.add_imu_data(imu_data)
            
        except Exception as e:
            self.logerr(f"Error in IMU callback: {e}")
    
    def encoder_callback(self, msg):
        """Process wheel encoder data."""
        try:
            # Parse encoder ticks from string message
            # In full implementation, would parse WheelEncoderStamped message
            ticks = self.parse_encoder_string(msg.data)
            
            # Create encoder data object
            encoder_data = EncoderData(
                timestamp=rospy.Time.now().to_sec(),
                left_ticks=ticks['left'],
                right_ticks=ticks['right'],
                confidence=0.9  # High confidence for encoder data
            )
            
            # Add to fusion engine
            self.fusion_engine.add_encoder_data(encoder_data)
            
        except Exception as e:
            self.logerr(f"Error in encoder callback: {e}")
    
    def car_cmd_callback(self, msg):
        """Process car command for encoder simulation."""
        try:
            # Store latest car command
            self.latest_car_cmd = msg
            
            # Simulate encoder data from car commands
            self.simulate_encoder_data(msg)
            
        except Exception as e:
            self.logerr(f"Error in car command callback: {e}")
    
    def fusion_timer_callback(self, event):
        """Perform sensor fusion at regular intervals."""
        try:
            # Perform sensor fusion
            fused_data = self.fusion_engine.fuse_sensor_data()
            
            if fused_data:
                # Publish fused perception data
                self.publish_fused_perception(fused_data)
                
                # Publish confidence metrics
                self.publish_confidence_metrics(fused_data.confidence_metrics)
                
                # Publish sensor health
                self.publish_sensor_health(fused_data.sensor_health)
                
                # Publish ego motion
                self.publish_ego_motion(fused_data.ego_motion)
            
        except Exception as e:
            self.logerr(f"Error in fusion timer callback: {e}")
    
    def parse_detection_string(self, detection_str):
        """Parse detection data from string message."""
        detections = []
        
        if not detection_str.strip():
            return detections
        
        # Parse detection string format: "class:duck, conf:0.8, center:(100,200), area:5000, priority:0.9"
        detection_parts = detection_str.split(" | ")
        
        for part in detection_parts:
            try:
                detection = {}
                attributes = part.split(", ")
                
                for attr in attributes:
                    if ":" in attr:
                        key, value = attr.split(":", 1)
                        
                        if key == "class":
                            detection['class_name'] = value
                        elif key == "conf":
                            detection['confidence'] = float(value)
                        elif key == "center":
                            # Parse center coordinates
                            coords = value.strip("()").split(",")
                            detection['center'] = (float(coords[0]), float(coords[1]))
                        elif key == "area":
                            detection['area'] = float(value)
                        elif key == "priority":
                            detection['priority'] = float(value)
                
                if detection:
                    detections.append(detection)
                    
            except Exception as e:
                self.logwarn(f"Error parsing detection part '{part}': {e}")
        
        return detections
    
    def parse_encoder_string(self, encoder_str):
        """Parse encoder data from string message."""
        # Placeholder for encoder string parsing
        return {'left': 0, 'right': 0}
    
    def simulate_encoder_data(self, car_cmd):
        """Simulate encoder data from car commands."""
        try:
            current_time = rospy.Time.now().to_sec()
            
            if self.last_encoder_time is None:
                self.last_encoder_time = current_time
                return
            
            dt = current_time - self.last_encoder_time
            
            # Calculate wheel velocities from car command
            v = car_cmd.v  # Linear velocity
            omega = car_cmd.omega  # Angular velocity
            
            # Differential drive kinematics
            wheel_base = 0.1  # meters
            v_left = v - (omega * wheel_base / 2.0)
            v_right = v + (omega * wheel_base / 2.0)
            
            # Convert to encoder ticks
            wheel_radius = 0.0318  # meters
            ticks_per_revolution = 135
            
            ticks_per_meter = ticks_per_revolution / (2 * np.pi * wheel_radius)
            
            left_ticks = self.last_encoder_ticks['left'] + int(v_left * dt * ticks_per_meter)
            right_ticks = self.last_encoder_ticks['right'] + int(v_right * dt * ticks_per_meter)
            
            # Create encoder data
            encoder_data = EncoderData(
                timestamp=current_time,
                left_ticks=left_ticks,
                right_ticks=right_ticks,
                confidence=0.95
            )
            
            # Add to fusion engine
            self.fusion_engine.add_encoder_data(encoder_data)
            
            # Update state
            self.last_encoder_ticks = {'left': left_ticks, 'right': right_ticks}
            self.last_encoder_time = current_time
            
        except Exception as e:
            self.logerr(f"Error simulating encoder data: {e}")
    
    def calculate_camera_confidence(self, detections):
        """Calculate confidence for camera data based on detections."""
        if not detections:
            return 0.5
        
        # Average confidence of all detections
        avg_confidence = np.mean([det.get('confidence', 0.5) for det in detections])
        
        # Boost confidence if multiple detections
        detection_boost = min(len(detections) * 0.1, 0.3)
        
        return min(avg_confidence + detection_boost, 1.0)
    
    def calculate_imu_confidence(self, imu_msg):
        """Calculate confidence for IMU data."""
        # Check for reasonable acceleration and angular velocity values
        accel_magnitude = np.sqrt(
            imu_msg.linear_acceleration.x**2 + 
            imu_msg.linear_acceleration.y**2 + 
            imu_msg.linear_acceleration.z**2
        )
        
        angular_magnitude = np.sqrt(
            imu_msg.angular_velocity.x**2 + 
            imu_msg.angular_velocity.y**2 + 
            imu_msg.angular_velocity.z**2
        )
        
        # Reasonable ranges for Duckiebot
        if 8.0 < accel_magnitude < 12.0 and angular_magnitude < 5.0:
            return 0.9
        else:
            return 0.6
    
    def publish_fused_perception(self, fused_data):
        """Publish fused perception data."""
        msg = String()
        
        # Format fused data as string
        data_str = (
            f"timestamp: {fused_data.timestamp:.3f}, "
            f"objects: {len(fused_data.objects)}, "
            f"ego_motion_confidence: {fused_data.ego_motion.get('confidence', 0.0):.3f}"
        )
        
        if fused_data.objects:
            object_info = []
            for obj in fused_data.objects:
                obj_str = f"{obj.get('class_name', 'unknown')}({obj.get('confidence', 0.0):.2f})"
                object_info.append(obj_str)
            data_str += f", detected_objects: [{', '.join(object_info)}]"
        
        msg.data = data_str
        self.pub_fused_perception.publish(msg)
    
    def publish_confidence_metrics(self, confidence_metrics):
        """Publish confidence metrics."""
        msg = String()
        
        metrics_str = (
            f"overall: {confidence_metrics.get('overall_confidence', 0.0):.3f}, "
            f"completeness: {confidence_metrics.get('data_completeness', 0.0):.3f}, "
            f"agreement: {confidence_metrics.get('sensor_agreement', 0.0):.3f}, "
            f"consistency: {confidence_metrics.get('temporal_consistency', 0.0):.3f}"
        )
        
        msg.data = metrics_str
        self.pub_confidence_metrics.publish(msg)
    
    def publish_sensor_health(self, sensor_health):
        """Publish sensor health status."""
        msg = String()
        
        health_info = []
        for sensor, health in sensor_health.items():
            health_str = f"{sensor}: {health['status']}({health['confidence']:.2f})"
            health_info.append(health_str)
        
        msg.data = ", ".join(health_info)
        self.pub_sensor_health.publish(msg)
    
    def publish_ego_motion(self, ego_motion):
        """Publish ego motion estimate."""
        if not ego_motion:
            return
        
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        
        msg.v = ego_motion.get('linear_velocity', 0.0)
        msg.omega = ego_motion.get('angular_velocity', 0.0)
        
        self.pub_ego_motion.publish(msg)
    
    def get_fusion_status(self):
        """Get current fusion status for diagnostics."""
        return {
            'camera_buffer_size': len(self.fusion_engine.camera_buffer),
            'imu_buffer_size': len(self.fusion_engine.imu_buffer),
            'encoder_buffer_size': len(self.fusion_engine.encoder_buffer),
            'fusion_rate': self.fusion_rate,
            'fusion_window': self.fusion_window
        }


if __name__ == "__main__":
    node = PerceptionDataFusionNode("perception_data_fusion_node")
    rospy.spin()