#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
import os


class CameraMockNode(DTROS):
    """
    Mock camera node that publishes synthetic images for testing.
    
    This node is useful when running the system without a physical camera
    or in simulation environments where camera data is not available.
    """
    
    def __init__(self, node_name):
        super(CameraMockNode, self).__init__(node_name=node_name, node_type=NodeType.DRIVER)
        
        # Parameters
        self.veh = os.environ.get('VEHICLE_NAME', 'duckiebot')
        self.publish_rate = rospy.get_param("~publish_rate", 10.0)  # Hz
        self.image_width = rospy.get_param("~image_width", 640)
        self.image_height = rospy.get_param("~image_height", 480)
        self.mock_type = rospy.get_param("~mock_type", "synthetic")  # synthetic, static, or noise
        
        # Initialize components
        self.bridge = CvBridge()
        self.frame_counter = 0
        
        # Publishers
        self.pub_image = rospy.Publisher(
            "~image/compressed", 
            CompressedImage, 
            queue_size=1
        )
        self.pub_camera_info = rospy.Publisher(
            "~camera_info", 
            CameraInfo, 
            queue_size=1
        )
        
        # Create timer for publishing
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), 
            self.publish_image
        )
        
        self.loginfo(f"Mock camera node initialized for vehicle: {self.veh}")
        self.loginfo(f"Publishing at {self.publish_rate} Hz with {self.mock_type} images")
    
    def generate_synthetic_image(self):
        """Generate a synthetic test image with moving elements."""
        # Create base image
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # Add a gradient background
        for y in range(self.image_height):
            intensity = int(50 + (y / self.image_height) * 100)
            image[y, :] = [intensity, intensity // 2, intensity // 3]
        
        # Add some moving elements based on frame counter
        time_factor = self.frame_counter * 0.1
        
        # Moving circle (simulates a duck or object)
        circle_x = int(self.image_width * 0.3 + 100 * np.sin(time_factor))
        circle_y = int(self.image_height * 0.6)
        cv2.circle(image, (circle_x, circle_y), 30, (0, 255, 255), -1)  # Yellow circle
        
        # Moving rectangle (simulates a duckiebot)
        rect_x = int(self.image_width * 0.7 + 80 * np.cos(time_factor * 0.7))
        rect_y = int(self.image_height * 0.4)
        cv2.rectangle(image, (rect_x - 25, rect_y - 15), (rect_x + 25, rect_y + 15), (255, 0, 0), -1)
        
        # Add some lane-like lines
        line_y = int(self.image_height * 0.8)
        cv2.line(image, (0, line_y), (self.image_width, line_y), (255, 255, 255), 3)
        
        # Add frame counter text
        cv2.putText(image, f"Frame: {self.frame_counter}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    def generate_static_image(self):
        """Generate a static test pattern."""
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # Create a checkerboard pattern
        square_size = 50
        for y in range(0, self.image_height, square_size):
            for x in range(0, self.image_width, square_size):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    image[y:y+square_size, x:x+square_size] = [200, 200, 200]
                else:
                    image[y:y+square_size, x:x+square_size] = [50, 50, 50]
        
        # Add some colored objects
        cv2.circle(image, (200, 200), 40, (0, 255, 255), -1)  # Yellow circle
        cv2.rectangle(image, (400, 150), (500, 250), (255, 0, 0), -1)  # Blue rectangle
        
        return image
    
    def generate_noise_image(self):
        """Generate a random noise image."""
        return np.random.randint(0, 256, (self.image_height, self.image_width, 3), dtype=np.uint8)
    
    def publish_image(self, event):
        """Publish a mock camera image."""
        try:
            # Generate image based on mock type
            if self.mock_type == "synthetic":
                image = self.generate_synthetic_image()
            elif self.mock_type == "static":
                image = self.generate_static_image()
            elif self.mock_type == "noise":
                image = self.generate_noise_image()
            else:
                self.logwarn(f"Unknown mock type: {self.mock_type}, using synthetic")
                image = self.generate_synthetic_image()
            
            # Convert to ROS message
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image, dst_format='jpg')
            compressed_msg.header.stamp = rospy.Time.now()
            compressed_msg.header.frame_id = f"{self.veh}/camera_optical_frame"
            
            # Publish image
            self.pub_image.publish(compressed_msg)
            
            # Publish camera info
            camera_info = CameraInfo()
            camera_info.header = compressed_msg.header
            camera_info.width = self.image_width
            camera_info.height = self.image_height
            
            # Basic camera matrix (approximate values)
            camera_info.K = [
                300.0, 0.0, self.image_width / 2.0,
                0.0, 300.0, self.image_height / 2.0,
                0.0, 0.0, 1.0
            ]
            camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
            camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            camera_info.P = [
                300.0, 0.0, self.image_width / 2.0, 0.0,
                0.0, 300.0, self.image_height / 2.0, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
            
            self.pub_camera_info.publish(camera_info)
            
            self.frame_counter += 1
            
            if self.frame_counter % 100 == 0:
                self.loginfo(f"Published {self.frame_counter} mock camera frames")
                
        except Exception as e:
            self.logerr(f"Error publishing mock camera image: {e}")


if __name__ == "__main__":
    node = CameraMockNode("camera_node")
    rospy.spin()