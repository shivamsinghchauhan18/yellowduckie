#!/usr/bin/env python3
"""
Test script to verify camera mock functionality
"""

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
import sys

class CameraMockTest:
    def __init__(self):
        rospy.init_node('camera_mock_test', anonymous=True)
        
        self.image_count = 0
        self.camera_info_count = 0
        
        # Subscribe to camera topics
        self.image_sub = rospy.Subscriber(
            '/blueduckie/camera_node/image/compressed',
            CompressedImage,
            self.image_callback
        )
        
        self.camera_info_sub = rospy.Subscriber(
            '/blueduckie/camera_node/camera_info',
            CameraInfo,
            self.camera_info_callback
        )
        
        print("Testing camera mock node...")
        print("Waiting for camera data...")
        
    def image_callback(self, msg):
        self.image_count += 1
        if self.image_count == 1:
            print(f"✓ First image received!")
            print(f"  - Format: {msg.format}")
            print(f"  - Frame ID: {msg.header.frame_id}")
            print(f"  - Data size: {len(msg.data)} bytes")
        
        if self.image_count % 10 == 0:
            print(f"✓ Received {self.image_count} images")
    
    def camera_info_callback(self, msg):
        self.camera_info_count += 1
        if self.camera_info_count == 1:
            print(f"✓ First camera info received!")
            print(f"  - Resolution: {msg.width}x{msg.height}")
            print(f"  - Frame ID: {msg.header.frame_id}")
    
    def run_test(self):
        # Wait for some messages
        timeout = rospy.Time.now() + rospy.Duration(10.0)
        
        while rospy.Time.now() < timeout and not rospy.is_shutdown():
            if self.image_count > 5 and self.camera_info_count > 5:
                print("✓ Camera mock test PASSED!")
                print(f"  - Images received: {self.image_count}")
                print(f"  - Camera info received: {self.camera_info_count}")
                return True
            
            rospy.sleep(0.1)
        
        print("✗ Camera mock test FAILED!")
        print(f"  - Images received: {self.image_count}")
        print(f"  - Camera info received: {self.camera_info_count}")
        return False

if __name__ == "__main__":
    try:
        test = CameraMockTest()
        success = test.run_test()
        sys.exit(0 if success else 1)
    except rospy.ROSInterruptException:
        print("Test interrupted")
        sys.exit(1)