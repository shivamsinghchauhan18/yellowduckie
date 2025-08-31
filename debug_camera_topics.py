#!/usr/bin/env python3
"""
Debug script to check camera topics and connectivity
"""

import rospy
import subprocess
import sys
from sensor_msgs.msg import CompressedImage

def check_ros_topics():
    """Check what ROS topics are available"""
    print("=== Checking ROS Topics ===")
    try:
        result = subprocess.run(['rostopic', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            topics = result.stdout.strip().split('\n')
            camera_topics = [t for t in topics if 'camera' in t or 'image' in t]
            
            print(f"Total topics: {len(topics)}")
            print(f"Camera-related topics: {len(camera_topics)}")
            
            if camera_topics:
                print("\nCamera topics found:")
                for topic in camera_topics:
                    print(f"  - {topic}")
            else:
                print("\n❌ No camera topics found!")
                
            return camera_topics
        else:
            print(f"❌ Failed to list topics: {result.stderr}")
            return []
    except Exception as e:
        print(f"❌ Error checking topics: {e}")
        return []

def check_topic_info(topic):
    """Check information about a specific topic"""
    print(f"\n=== Checking Topic: {topic} ===")
    try:
        # Check topic info
        result = subprocess.run(['rostopic', 'info', topic], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ Failed to get topic info: {result.stderr}")
            
        # Check message rate
        result = subprocess.run(['rostopic', 'hz', topic], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"Topic rate info:\n{result.stdout}")
        else:
            print(f"❌ No messages on topic or timeout")
            
    except Exception as e:
        print(f"❌ Error checking topic {topic}: {e}")

def test_camera_subscription():
    """Test subscribing to camera topic"""
    print("\n=== Testing Camera Subscription ===")
    
    # Try common camera topic names
    possible_topics = [
        f"/{rospy.get_param('/vehicle_name', 'blueduckie')}/camera_node/image/compressed",
        "/camera_node/image/compressed",
        "/camera/image/compressed",
        "/image/compressed"
    ]
    
    for topic in possible_topics:
        print(f"\nTrying to subscribe to: {topic}")
        try:
            def callback(msg):
                print(f"✅ Received image on {topic}!")
                print(f"   Format: {msg.format}")
                print(f"   Size: {len(msg.data)} bytes")
                print(f"   Frame: {msg.header.frame_id}")
                
            sub = rospy.Subscriber(topic, CompressedImage, callback)
            
            # Wait for a message
            timeout = rospy.Time.now() + rospy.Duration(5.0)
            received = False
            
            while rospy.Time.now() < timeout and not rospy.is_shutdown():
                if sub.get_num_connections() > 0:
                    received = True
                    break
                rospy.sleep(0.1)
            
            if not received:
                print(f"❌ No messages received on {topic}")
            
            sub.unregister()
            
        except Exception as e:
            print(f"❌ Error subscribing to {topic}: {e}")

def check_environment():
    """Check ROS environment variables"""
    print("\n=== Checking ROS Environment ===")
    
    import os
    ros_vars = ['ROS_MASTER_URI', 'ROS_IP', 'ROS_HOSTNAME', 'VEHICLE_NAME']
    
    for var in ros_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"{var}: {value}")

def main():
    print("🔍 Duckiebot Camera Debug Tool")
    print("=" * 50)
    
    try:
        rospy.init_node('camera_debug', anonymous=True)
        
        check_environment()
        camera_topics = check_ros_topics()
        
        # Check specific camera topics
        for topic in camera_topics:
            if 'compressed' in topic:
                check_topic_info(topic)
        
        # Test subscription
        test_camera_subscription()
        
        print("\n" + "=" * 50)
        print("🏁 Debug complete!")
        
        if not camera_topics:
            print("\n💡 Troubleshooting suggestions:")
            print("1. Make sure your Duckiebot is powered on and connected")
            print("2. Check that the camera_node is running on the robot")
            print("3. Verify ROS_MASTER_URI points to your robot")
            print("4. Check network connectivity to your robot")
            
    except rospy.ROSInterruptException:
        print("Debug interrupted")
    except Exception as e:
        print(f"❌ Debug failed: {e}")

if __name__ == "__main__":
    main()