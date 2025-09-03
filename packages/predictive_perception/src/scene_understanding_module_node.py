#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType

# Standard ROS messages
from std_msgs.msg import Header, String
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from duckietown_msgs.msg import LanePose
from duckietown_msgs.msg import AprilTagDetectionArray
# Custom messages (we'll create these)
try:
    from predictive_perception.msg import SceneAnalysis, EnvironmentalConditions
except ImportError:
    # Fallback if messages aren't built yet
    SceneAnalysis = None
    EnvironmentalConditions = None

# Internal modules
from predictive_perception.scene_analyzer import SceneAnalyzer, SceneClassifier, ScenarioType, EnvironmentalCondition


class SceneUnderstandingModuleNode(DTROS):
    """
    Scene Understanding Module Node for high-level scene interpretation.
    
    This node provides:
    - Traffic scenario classification (intersection, lane following, obstacles)
    - Environmental condition assessment (lighting, visibility, traffic density)
    - Scene context analysis for navigation decision making
    """
    
    def __init__(self, node_name):
        super(SceneUnderstandingModuleNode, self).__init__(
            node_name=node_name, 
            node_type=NodeType.PERCEPTION
        )
        
        # Initialize parameters
        self.veh = rospy.get_param("~veh", "duckiebot")
        self.publish_rate = rospy.get_param("~publish_rate", 5.0)
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        
        # Analysis parameters
        self.brightness_threshold_low = rospy.get_param("~brightness_threshold_low", 50)
        self.brightness_threshold_high = rospy.get_param("~brightness_threshold_high", 200)
        self.traffic_density_threshold = rospy.get_param("~traffic_density_threshold", 3)
        
        # Initialize components
        self.bridge = CvBridge()
        self.scene_analyzer = SceneAnalyzer()
        self.scene_classifier = SceneClassifier()
        
        # Configure analyzer thresholds
        self.scene_analyzer.brightness_threshold_low = self.brightness_threshold_low
        self.scene_analyzer.brightness_threshold_high = self.brightness_threshold_high
        self.scene_analyzer.traffic_density_threshold = self.traffic_density_threshold
        
        # Data storage
        self.current_image = None
        self.tracked_objects = {}
        self.lane_pose = None
        self.apriltags = []
        
        # Publishers
        self.pub_scene_analysis = rospy.Publisher(
            "~scene_analysis", 
            String,  # Using String for now, will create custom message later
            queue_size=1
        )
        self.pub_scenario_type = rospy.Publisher(
            "~scenario_type", 
            String, 
            queue_size=1
        )
        self.pub_environmental_conditions = rospy.Publisher(
            "~environmental_conditions", 
            String, 
            queue_size=1
        )
        
        if self.enable_visualization:
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
        
        # Subscribe to tracked objects (from predictive perception manager)
        self.sub_tracked_objects = rospy.Subscriber(
            f"/{self.veh}/predictive_perception_manager_node/tracked_objects",
            String,  # Placeholder - would be TrackedObject in real implementation
            self.tracked_objects_callback,
            queue_size=10
        )
        
        # Subscribe to lane pose
        self.sub_lane_pose = rospy.Subscriber(
            f"/{self.veh}/lane_filter_node/lane_pose",
            LanePose,
            self.lane_pose_callback,
            queue_size=1
        )
        
        # Subscribe to AprilTag detections
        self.sub_apriltags = rospy.Subscriber(
            f"/{self.veh}/apriltag_detector_node/detections",
            AprilTagDetectionArray,
            self.apriltags_callback,
    queue_size=1
        )
        
        # Timer for regular analysis
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), 
            self.analyze_scene_timer
        )
        
        self.loginfo("Scene Understanding Module initialized")
    
    def image_callback(self, image_msg):
        """
        Process incoming camera images.
        """
        try:
            # Convert and store image
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
            self.current_image = bgr
            
        except Exception as e:
            self.logerr(f"Error in image callback: {e}")
    
    def tracked_objects_callback(self, msg):
        """
        Process tracked objects information.
        Note: This is a placeholder - in real implementation would parse TrackedObject messages
        """
        # Placeholder implementation
        # In real system, would maintain dictionary of tracked objects
        pass
    
    def lane_pose_callback(self, msg: LanePose):
        """
        Process lane pose information.
        """
        self.lane_pose = msg
    
    def apriltags_callback(self, msg):
        """
        Process AprilTag detections.
        """
        # Placeholder implementation
        # In real system, would store AprilTag detection data
        pass
    
    def analyze_scene_timer(self, event):
        """
        Perform scene analysis at regular intervals.
        """
        if self.current_image is None:
            return
        
        try:
            # Perform scene analysis
            analysis = self.scene_analyzer.analyze_scene(
                self.current_image,
                self.tracked_objects,
                self.lane_pose,
                self.apriltags
            )
            
            # Publish analysis results
            self.publish_scene_analysis(analysis)
            
            # Create and publish visualization if enabled
            if self.enable_visualization:
                self.publish_scene_visualization(analysis)
                
        except Exception as e:
            self.logerr(f"Error in scene analysis: {e}")
    
    def publish_scene_analysis(self, analysis):
        """
        Publish comprehensive scene analysis results.
        """
        # Create scene analysis message (using String for now)
        analysis_msg = String()
        
        # Format analysis as JSON-like string
        analysis_str = (
            f"scenario_type: {analysis['scenario_type'].value}, "
            f"traffic_density: {analysis['traffic_density']['density_level']}, "
            f"visibility_score: {analysis['visibility_score']:.2f}, "
            f"object_count: {analysis['object_count']}, "
            f"confidence: {analysis['confidence']:.2f}"
        )
        
        analysis_msg.data = analysis_str
        self.pub_scene_analysis.publish(analysis_msg)
        
        # Publish individual components
        scenario_msg = String()
        scenario_msg.data = analysis['scenario_type'].value
        self.pub_scenario_type.publish(scenario_msg)
        
        conditions_msg = String()
        condition_names = [cond.value for cond in analysis['environmental_conditions']]
        conditions_msg.data = ", ".join(condition_names)
        self.pub_environmental_conditions.publish(conditions_msg)
        
        # Log analysis results
        self.loginfo(f"Scene Analysis: {analysis_str}")
    
    def publish_scene_visualization(self, analysis):
        """
        Create and publish scene analysis visualization.
        """
        try:
            if self.current_image is None:
                return
            
            # Create visualization image
            vis_image = self.current_image.copy()
            
            # Add analysis information as text overlay
            self._add_analysis_overlay(vis_image, analysis)
            
            # Add traffic density visualization
            self._add_traffic_density_overlay(vis_image, analysis)
            
            # Add visibility indicators
            self._add_visibility_overlay(vis_image, analysis)
            
            # Convert and publish
            debug_msg = self.bridge.cv2_to_compressed_imgmsg(vis_image)
            self.pub_debug_image.publish(debug_msg)
            
        except Exception as e:
            self.logerr(f"Error in scene visualization: {e}")
    
    def _add_analysis_overlay(self, image, analysis):
        """Add scene analysis text overlay to image."""
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # White
        thickness = 2
        
        # Background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Add text lines
        y_offset = 30
        line_height = 20
        
        lines = [
            f"Scenario: {analysis['scenario_type'].value}",
            f"Traffic: {analysis['traffic_density']['density_level']} ({analysis['object_count']} objects)",
            f"Visibility: {analysis['visibility_score']:.2f}",
            f"Confidence: {analysis['confidence']:.2f}"
        ]
        
        for i, line in enumerate(lines):
            y_pos = y_offset + i * line_height
            cv2.putText(image, line, (15, y_pos), font, font_scale, color, thickness)
    
    def _add_traffic_density_overlay(self, image, analysis):
        """Add traffic density visualization."""
        density_level = analysis['traffic_density']['density_level']
        
        # Color coding for density levels
        density_colors = {
            'none': (0, 255, 0),      # Green
            'low': (0, 255, 255),     # Yellow
            'medium': (0, 165, 255),  # Orange
            'high': (0, 0, 255)       # Red
        }
        
        color = density_colors.get(density_level, (128, 128, 128))
        
        # Draw density indicator
        cv2.circle(image, (image.shape[1] - 50, 50), 20, color, -1)
        cv2.circle(image, (image.shape[1] - 50, 50), 22, (255, 255, 255), 2)
        
        # Add density label
        cv2.putText(image, density_level.upper(), 
                   (image.shape[1] - 80, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _add_visibility_overlay(self, image, analysis):
        """Add visibility score visualization."""
        visibility = analysis['visibility_score']
        
        # Create visibility bar
        bar_width = 100
        bar_height = 10
        bar_x = image.shape[1] - bar_width - 20
        bar_y = image.shape[0] - 30
        
        # Background bar
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (64, 64, 64), -1)
        
        # Visibility level bar
        fill_width = int(bar_width * visibility)
        
        # Color based on visibility level
        if visibility > 0.7:
            color = (0, 255, 0)      # Green
        elif visibility > 0.4:
            color = (0, 255, 255)    # Yellow
        else:
            color = (0, 0, 255)      # Red
        
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        
        # Add visibility label
        cv2.putText(image, f"Visibility: {visibility:.2f}", 
                   (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def get_current_scene_summary(self):
        """
        Get current scene summary for other nodes to query.
        
        Returns:
            Dictionary with current scene information
        """
        if self.current_image is None:
            return {'status': 'no_data'}
        
        # Get scene summary from analyzer
        summary = self.scene_analyzer.get_scene_summary()
        
        # Add current analysis if available
        if hasattr(self, '_last_analysis'):
            summary.update({
                'current_scenario': self._last_analysis['scenario_type'].value,
                'current_visibility': self._last_analysis['visibility_score'],
                'current_traffic_density': self._last_analysis['traffic_density']['density_level']
            })
        
        return summary


if __name__ == "__main__":
    node = SceneUnderstandingModuleNode("scene_understanding_module_node")
    rospy.spin()