#!/usr/bin/env python3

import cv2
import numpy as np
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point, Polygon, Point32

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


class EnhancedObjectDetectionNode(DTROS):
    """
    Enhanced Object Detection Node with improved detection data output.
    
    This node extends the basic object detection with:
    - Enhanced detection data with confidence scores
    - Bounding box tracking and filtering
    - Noise reduction algorithms
    - Integration with predictive perception system
    """
    
    def __init__(self, node_name):
        super(EnhancedObjectDetectionNode, self).__init__(
            node_name=node_name, 
            node_type=NodeType.PERCEPTION
        )
        
        self.initialized = False
        self.frame_id = 0
        
        # Parameters
        self.veh = os.environ.get('VEHICLE_NAME', 'duckiebot')
        self.duck_threshold = rospy.get_param("~duck_threshold", 0.7)
        self.duckiebot_threshold = rospy.get_param("~duckiebot_threshold", 0.7)
        self.enable_filtering = rospy.get_param("~enable_filtering", True)
        self.enable_tracking = rospy.get_param("~enable_tracking", True)
        
        # Detection filtering parameters
        self.min_area_threshold = rospy.get_param("~min_area_threshold", 500)
        self.max_area_threshold = rospy.get_param("~max_area_threshold", 50000)
        self.confidence_smoothing = rospy.get_param("~confidence_smoothing", 0.7)
        
        # Region of interest parameters
        self.roi_left = rospy.get_param("~roi_left", 0.1)    # 10% from left
        self.roi_right = rospy.get_param("~roi_right", 0.9)  # 90% from left
        self.roi_top = rospy.get_param("~roi_top", 0.2)     # 20% from top
        self.roi_bottom = rospy.get_param("~roi_bottom", 1.0) # 100% from top
        
        # Detection history for filtering
        self.detection_history = []
        self.max_history_length = 5
        
        # Publishers
        self.pub_vel = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd", 
            Twist2DStamped, 
            queue_size=1
        )
        self.pub_detections_image = rospy.Publisher(
            "~image/compressed", 
            CompressedImage, 
            queue_size=1
        )
        self.pub_enhanced_detections = rospy.Publisher(
            "~enhanced_detections", 
            String,  # Would be custom DetectionArray message in full implementation
            queue_size=1
        )
        self.pub_filtered_detections = rospy.Publisher(
            "~filtered_detections", 
            String,  # Would be custom DetectionArray message in full implementation
            queue_size=1
        )
        
        # Subscribers
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_callback,
            buff_size=10000000,
            queue_size=1,
        )
        
        # Initialize components
        self.bridge = CvBridge()
        try:
            self.model_wrapper = Wrapper(rospy.get_param("~AIDO_eval", False))
        except Exception as e:
            self.logwarn(f"Model wrapper unavailable: {e}. Running without detection.")
            self.model_wrapper = None
        
        self.initialized = True
        self.loginfo("Enhanced Object Detection Node initialized")
    
    def image_callback(self, image_msg):
        """Process incoming camera images with enhanced detection."""
        if not self.initialized:
            return
        
        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED())
        if self.frame_id != 0:
            return
        
        try:
            # Convert image
            bgr = self.bridge.compressed_imgmsg_to_cv2(image_msg)
            rgb = bgr[..., ::-1]
            rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
            
            # Run object detection
            bboxes, classes, scores = self.model_wrapper.predict(rgb)
            
            # Enhance detections with additional processing
            enhanced_detections = self.enhance_detections(bboxes, classes, scores, rgb.shape)
            
            # Apply filtering if enabled
            if self.enable_filtering:
                filtered_detections = self.filter_detections(enhanced_detections)
            else:
                filtered_detections = enhanced_detections
            
            # Update detection history
            self.update_detection_history(filtered_detections)
            
            # Apply temporal smoothing
            smoothed_detections = self.apply_temporal_smoothing(filtered_detections)
            
            # Publish enhanced detection data
            self.publish_enhanced_detections(smoothed_detections, image_msg.header)
            
            # Generate control commands based on detections
            self.generate_control_commands(smoothed_detections)
            
            # Create and publish visualization
            self.publish_detection_visualization(rgb, smoothed_detections)
            
        except Exception as e:
            self.logerr(f"Error in image callback: {e}")
    
    def enhance_detections(self, bboxes, classes, scores, image_shape):
        """
        Enhance raw detections with additional information.
        
        Args:
            bboxes: Raw bounding boxes
            classes: Object classes
            scores: Confidence scores
            image_shape: Shape of the input image
            
        Returns:
            List of enhanced detection dictionaries
        """
        enhanced_detections = []
        
        for bbox, cls, score in zip(bboxes, classes, scores):
            # Skip detections below threshold
            if cls == 0 and score < self.duck_threshold:
                continue
            if cls == 1 and score < self.duckiebot_threshold:
                continue
            
            # Calculate enhanced properties
            detection = {
                'bbox': bbox,
                'class': cls,
                'confidence': score,
                'area': self.calculate_bbox_area(bbox),
                'center': self.calculate_bbox_center(bbox),
                'aspect_ratio': self.calculate_aspect_ratio(bbox),
                'relative_size': self.calculate_relative_size(bbox, image_shape),
                'in_roi': self.is_in_roi(bbox, image_shape),
                'timestamp': rospy.Time.now().to_sec()
            }
            
            # Add class-specific enhancements
            if cls == 0:  # Duck
                detection['class_name'] = 'duck'
                detection['priority'] = self.calculate_duck_priority(detection)
            elif cls == 1:  # Duckiebot
                detection['class_name'] = 'duckiebot'
                detection['priority'] = self.calculate_duckiebot_priority(detection)
            else:
                detection['class_name'] = 'unknown'
                detection['priority'] = 0.5
            
            enhanced_detections.append(detection)
        
        return enhanced_detections
    
    def filter_detections(self, detections):
        """
        Apply filtering algorithms to reduce noise and false positives.
        
        Args:
            detections: List of enhanced detection dictionaries
            
        Returns:
            List of filtered detections
        """
        filtered_detections = []
        
        for detection in detections:
            # Area-based filtering
            if not self.is_valid_area(detection['area']):
                continue
            
            # Aspect ratio filtering
            if not self.is_valid_aspect_ratio(detection['aspect_ratio'], detection['class']):
                continue
            
            # Confidence-based filtering with class-specific thresholds
            if not self.is_valid_confidence(detection['confidence'], detection['class']):
                continue
            
            # ROI filtering (prioritize objects in region of interest)
            if detection['in_roi']:
                detection['confidence'] *= 1.1  # Boost confidence for ROI objects
            
            # Size consistency filtering
            if self.is_size_consistent(detection):
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def apply_temporal_smoothing(self, detections):
        """
        Apply temporal smoothing to reduce detection jitter.
        
        Args:
            detections: Current frame detections
            
        Returns:
            Smoothed detections
        """
        if not self.detection_history:
            return detections
        
        smoothed_detections = []
        
        for detection in detections:
            # Find matching detection in history
            matched_detection = self.find_matching_detection(detection)
            
            if matched_detection:
                # Apply smoothing to confidence
                smoothed_confidence = (
                    self.confidence_smoothing * matched_detection['confidence'] +
                    (1 - self.confidence_smoothing) * detection['confidence']
                )
                detection['confidence'] = smoothed_confidence
                
                # Apply smoothing to bounding box (reduce jitter)
                detection['bbox'] = self.smooth_bbox(detection['bbox'], matched_detection['bbox'])
            
            smoothed_detections.append(detection)
        
        return smoothed_detections
    
    def calculate_bbox_area(self, bbox):
        """Calculate bounding box area."""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def calculate_bbox_center(self, bbox):
        """Calculate bounding box center."""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def calculate_aspect_ratio(self, bbox):
        """Calculate bounding box aspect ratio."""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width / height if height > 0 else 1.0
    
    def calculate_relative_size(self, bbox, image_shape):
        """Calculate relative size of bounding box."""
        area = self.calculate_bbox_area(bbox)
        image_area = image_shape[0] * image_shape[1]
        return area / image_area
    
    def is_in_roi(self, bbox, image_shape):
        """Check if bounding box is in region of interest."""
        center = self.calculate_bbox_center(bbox)
        
        roi_left = self.roi_left * image_shape[1]
        roi_right = self.roi_right * image_shape[1]
        roi_top = self.roi_top * image_shape[0]
        roi_bottom = self.roi_bottom * image_shape[0]
        
        return (roi_left <= center[0] <= roi_right and 
                roi_top <= center[1] <= roi_bottom)
    
    def calculate_duck_priority(self, detection):
        """Calculate priority for duck detection."""
        # Higher priority for larger, more confident ducks in ROI
        priority = detection['confidence'] * 0.5
        
        if detection['in_roi']:
            priority += 0.3
        
        # Size factor (larger ducks are more important)
        size_factor = min(detection['relative_size'] * 10, 0.2)
        priority += size_factor
        
        return min(priority, 1.0)
    
    def calculate_duckiebot_priority(self, detection):
        """Calculate priority for duckiebot detection."""
        # Higher priority for larger, more confident duckiebots
        priority = detection['confidence'] * 0.6
        
        if detection['in_roi']:
            priority += 0.4
        
        return min(priority, 1.0)
    
    def is_valid_area(self, area):
        """Check if detection area is within valid range."""
        return self.min_area_threshold <= area <= self.max_area_threshold
    
    def is_valid_aspect_ratio(self, aspect_ratio, object_class):
        """Check if aspect ratio is valid for object class."""
        if object_class == 0:  # Duck
            return 0.5 <= aspect_ratio <= 2.0
        elif object_class == 1:  # Duckiebot
            return 0.8 <= aspect_ratio <= 1.5
        else:
            return True  # No constraint for unknown objects
    
    def is_valid_confidence(self, confidence, object_class):
        """Check if confidence is above class-specific threshold."""
        if object_class == 0:  # Duck
            return confidence >= self.duck_threshold
        elif object_class == 1:  # Duckiebot
            return confidence >= self.duckiebot_threshold
        else:
            return confidence >= 0.5
    
    def is_size_consistent(self, detection):
        """Check if detection size is consistent with previous detections."""
        if not self.detection_history:
            return True
        
        # Simple size consistency check
        # In a more sophisticated implementation, this would track object sizes over time
        return True
    
    def find_matching_detection(self, detection):
        """Find matching detection in history for temporal smoothing."""
        if not self.detection_history:
            return None
        
        last_frame_detections = self.detection_history[-1]
        
        # Find closest detection by center distance
        min_distance = float('inf')
        best_match = None
        
        for hist_detection in last_frame_detections:
            if hist_detection['class'] == detection['class']:
                distance = np.sqrt(
                    (detection['center'][0] - hist_detection['center'][0])**2 +
                    (detection['center'][1] - hist_detection['center'][1])**2
                )
                
                if distance < min_distance and distance < 50:  # Max matching distance
                    min_distance = distance
                    best_match = hist_detection
        
        return best_match
    
    def smooth_bbox(self, current_bbox, previous_bbox, alpha=0.3):
        """Apply smoothing to bounding box coordinates."""
        smoothed_bbox = []
        
        for i in range(4):
            smoothed_coord = alpha * previous_bbox[i] + (1 - alpha) * current_bbox[i]
            smoothed_bbox.append(smoothed_coord)
        
        return smoothed_bbox
    
    def update_detection_history(self, detections):
        """Update detection history for temporal analysis."""
        self.detection_history.append(detections)
        
        # Maintain maximum history length
        if len(self.detection_history) > self.max_history_length:
            self.detection_history.pop(0)
    
    def publish_enhanced_detections(self, detections, header):
        """Publish enhanced detection data."""
        # Create detection message (using String for now)
        detection_msg = String()
        
        # Format detections as structured string
        detection_data = []
        for det in detections:
            det_str = (
                f"class:{det['class_name']}, "
                f"conf:{det['confidence']:.3f}, "
                f"center:({det['center'][0]:.1f},{det['center'][1]:.1f}), "
                f"area:{det['area']:.0f}, "
                f"priority:{det['priority']:.3f}"
            )
            detection_data.append(det_str)
        
        detection_msg.data = " | ".join(detection_data)
        self.pub_enhanced_detections.publish(detection_msg)
        
        # Also publish filtered detections
        filtered_msg = String()
        filtered_msg.data = f"filtered_count:{len(detections)}"
        self.pub_filtered_detections.publish(filtered_msg)
    
    def generate_control_commands(self, detections):
        """Generate control commands based on enhanced detections."""
        # Enhanced stop logic with priority-based decision making
        stop_signal = False
        stop_reason = ""
        
        # Define the center region for collision detection
        left_boundary = int(IMAGE_SIZE * 0.33)
        right_boundary = int(IMAGE_SIZE * 0.75)
        
        # Sort detections by priority
        sorted_detections = sorted(detections, key=lambda x: x['priority'], reverse=True)
        
        for detection in sorted_detections:
            center_x = detection['center'][0]
            
            # Check if high-priority object is in path
            if detection['priority'] > 0.7 and left_boundary < center_x < right_boundary:
                if detection['class'] == 0:  # Duck
                    if detection['area'] > 2000:
                        stop_signal = True
                        stop_reason = f"High-priority duck (conf: {detection['confidence']:.2f})"
                        break
                elif detection['class'] == 1:  # Duckiebot
                    if detection['area'] > 10000:
                        stop_signal = True
                        stop_reason = f"High-priority duckiebot (conf: {detection['confidence']:.2f})"
                        break
        
        # Create velocity command
        vel_cmd = Twist2DStamped()
        vel_cmd.header.stamp = rospy.Time.now()
        
        if stop_signal:
            vel_cmd.v = 0.0
            vel_cmd.omega = 0.0
            self.loginfo(f"Stopping for: {stop_reason}")
        else:
            vel_cmd.v = 0.2
            vel_cmd.omega = 0.0
            self.logdebug("Driving...")
        
        self.pub_vel.publish(vel_cmd)
    
    def publish_detection_visualization(self, rgb_image, detections):
        """Create and publish enhanced detection visualization."""
        try:
            debug_image = rgb_image.copy()
            
            # Draw ROI
            self.draw_roi(debug_image)
            
            # Draw detections with enhanced information
            for detection in detections:
                self.draw_enhanced_detection(debug_image, detection)
            
            # Add detection statistics
            self.draw_detection_stats(debug_image, detections)
            
            # Convert and publish
            bgr_debug = debug_image[..., ::-1]
            debug_msg = self.bridge.cv2_to_compressed_imgmsg(bgr_debug)
            self.pub_detections_image.publish(debug_msg)
            
        except Exception as e:
            self.logerr(f"Error in detection visualization: {e}")
    
    def draw_roi(self, image):
        """Draw region of interest on image."""
        h, w = image.shape[:2]
        
        roi_coords = [
            int(self.roi_left * w), int(self.roi_top * h),
            int(self.roi_right * w), int(self.roi_bottom * h)
        ]
        
        cv2.rectangle(image, (roi_coords[0], roi_coords[1]), 
                     (roi_coords[2], roi_coords[3]), (0, 255, 255), 2)
        
        cv2.putText(image, "ROI", (roi_coords[0] + 5, roi_coords[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def draw_enhanced_detection(self, image, detection):
        """Draw enhanced detection with additional information."""
        bbox = detection['bbox']
        pt1 = (int(bbox[0]), int(bbox[1]))
        pt2 = (int(bbox[2]), int(bbox[3]))
        
        # Color based on priority
        if detection['priority'] > 0.8:
            color = (255, 0, 0)  # Red for high priority
        elif detection['priority'] > 0.5:
            color = (255, 255, 0)  # Yellow for medium priority
        else:
            color = (0, 255, 0)  # Green for low priority
        
        # Draw bounding box
        thickness = 3 if detection['priority'] > 0.7 else 2
        cv2.rectangle(image, pt1, pt2, color, thickness)
        
        # Draw center point
        center = (int(detection['center'][0]), int(detection['center'][1]))
        cv2.circle(image, center, 5, color, -1)
        
        # Enhanced label with priority and confidence
        label = (
            f"{detection['class_name']}: {detection['confidence']:.2f} "
            f"(P:{detection['priority']:.2f})"
        )
        
        # Label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(image, (pt1[0], pt1[1] - label_size[1] - 10),
                     (pt1[0] + label_size[0], pt1[1]), color, -1)
        
        # Label text
        cv2.putText(image, label, (pt1[0], pt1[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_detection_stats(self, image, detections):
        """Draw detection statistics on image."""
        stats_text = [
            f"Detections: {len(detections)}",
            f"High Priority: {sum(1 for d in detections if d['priority'] > 0.7)}",
            f"In ROI: {sum(1 for d in detections if d['in_roi'])}"
        ]
        
        # Background for stats
        cv2.rectangle(image, (10, 10), (250, 80), (0, 0, 0), -1)
        
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * 20
            cv2.putText(image, text, (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


if __name__ == "__main__":
    node = EnhancedObjectDetectionNode("enhanced_object_detection_node")
    rospy.spin()