#!/usr/bin/env python3
"""
Comprehensive camera setup test for Duckiebot
"""

import cv2
import os
import subprocess
import sys

def test_camera_hardware():
    """Test if camera hardware is accessible."""
    print("🔍 Testing camera hardware...")
    
    # Check for video devices
    video_devices = []
    for i in range(5):
        device_path = f'/dev/video{i}'
        if os.path.exists(device_path):
            video_devices.append(device_path)
    
    if video_devices:
        print(f"✅ Found video devices: {video_devices}")
    else:
        print("❌ No video devices found")
        return False
    
    # Test OpenCV access
    for i in range(len(video_devices)):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                
                if ret and frame is not None:
                    print(f"✅ Camera {i} is accessible via OpenCV")
                    print(f"   Frame shape: {frame.shape}")
                    return True
                else:
                    print(f"❌ Camera {i} opened but no frame captured")
            else:
                print(f"❌ Could not open camera {i}")
        except Exception as e:
            print(f"❌ Error testing camera {i}: {e}")
    
    return False

def test_camera_permissions():
    """Test camera device permissions."""
    print("\n🔐 Testing camera permissions...")
    
    device_path = '/dev/video0'
    if not os.path.exists(device_path):
        print(f"❌ {device_path} does not exist")
        return False
    
    # Check read/write permissions
    readable = os.access(device_path, os.R_OK)
    writable = os.access(device_path, os.W_OK)
    
    if readable and writable:
        print(f"✅ {device_path} has proper permissions")
        return True
    else:
        print(f"❌ {device_path} permissions insufficient")
        print(f"   Readable: {readable}, Writable: {writable}")
        print("💡 Try: sudo chmod 666 /dev/video0")
        return False

def test_camera_capture():
    """Test actual camera capture."""
    print("\n📸 Testing camera capture...")
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        # Set properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 10)
        
        # Capture a few frames
        success_count = 0
        for i in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                success_count += 1
                print(f"✅ Frame {i+1}: {frame.shape}")
            else:
                print(f"❌ Frame {i+1}: Failed to capture")
        
        cap.release()
        
        if success_count >= 3:
            print(f"✅ Camera capture successful ({success_count}/5 frames)")
            return True
        else:
            print(f"❌ Camera capture unreliable ({success_count}/5 frames)")
            return False
            
    except Exception as e:
        print(f"❌ Camera capture test failed: {e}")
        return False

def test_ros_environment():
    """Test ROS environment setup."""
    print("\n🤖 Testing ROS environment...")
    
    # Check environment variables
    ros_vars = {
        'ROS_MASTER_URI': os.environ.get('ROS_MASTER_URI'),
        'VEHICLE_NAME': os.environ.get('VEHICLE_NAME'),
        'ROS_DISTRO': os.environ.get('ROS_DISTRO')
    }
    
    for var, value in ros_vars.items():
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    # Test ROS commands
    try:
        result = subprocess.run(['which', 'rosnode'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ ROS commands available")
        else:
            print("❌ ROS commands not found")
            return False
    except:
        print("❌ Could not check ROS commands")
        return False
    
    return True

def provide_recommendations():
    """Provide setup recommendations."""
    print("\n💡 Setup Recommendations:")
    print("=" * 50)
    
    print("1. Camera Hardware:")
    print("   - Ensure camera is properly connected")
    print("   - Check camera LED (should be on when active)")
    print("   - Try: lsusb (for USB cameras) or vcgencmd get_camera (for Pi camera)")
    
    print("\n2. Permissions:")
    print("   - Run: sudo chmod 666 /dev/video0")
    print("   - Add user to video group: sudo usermod -a -G video $USER")
    
    print("\n3. ROS Environment:")
    print("   - Set VEHICLE_NAME: export VEHICLE_NAME=blueduckie")
    print("   - Set ROS_MASTER_URI: export ROS_MASTER_URI=http://localhost:11311")
    print("   - Source ROS: source /opt/ros/noetic/setup.bash")
    
    print("\n4. Test Camera:")
    print("   - Manual test: ./start_camera.sh")
    print("   - Check topics: rostopic list | grep camera")
    print("   - View images: rqt_image_view")

def main():
    print("🎥 Duckiebot Camera Setup Test")
    print("=" * 50)
    
    tests = [
        ("Camera Hardware", test_camera_hardware),
        ("Camera Permissions", test_camera_permissions),
        ("Camera Capture", test_camera_capture),
        ("ROS Environment", test_ros_environment)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 50)
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your camera should work.")
    else:
        provide_recommendations()
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)