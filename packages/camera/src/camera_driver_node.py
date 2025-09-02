#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import os
import subprocess
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType


class CameraDriverNode(DTROS):
    """
    Universal camera driver node for Duckiebot.
    
    Supports multiple camera types:
    - Raspberry Pi camera (via raspistill/libcamera)
    - USB cameras (via OpenCV)
    - Jetson CSI camera (via GStreamer nvarguscamerasrc)
    - Custom camera interfaces
    """
    
    def __init__(self, node_name):
        super(CameraDriverNode, self).__init__(node_name=node_name, node_type=NodeType.DRIVER)
        
        # Parameters
        self.veh_name = rospy.get_param("~veh_name", os.environ.get('VEHICLE_NAME', 'duckiebot'))
        self.framerate = rospy.get_param("~framerate", 30)
        self.res_w = rospy.get_param("~res_w", 640)
        self.res_h = rospy.get_param("~res_h", 480)
        self.exposure_mode = rospy.get_param("~exposure_mode", "auto")
        self.camera_type = rospy.get_param("~camera_type", "auto")
    # Optional Jetson-specific params
        self.flip_method = rospy.get_param("~flip_method", 0)  # Jetson nvvidconv flip-method
        self.sensor_mode = rospy.get_param("~sensor_mode", None)  # Jetson nvarguscamerasrc sensor-mode
        
        # Initialize components
        self.bridge = CvBridge()
        self.camera = None
        self.camera_info = self.create_camera_info()
        
        # Publishers
        self.pub_image = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1)
        self.pub_camera_info = rospy.Publisher("~camera_info", CameraInfo, queue_size=1)
        
        # Initialize camera
        self.initialize_camera()
        
        if self.camera is not None:
            # Start publishing timer
            self.timer = rospy.Timer(rospy.Duration(1.0 / self.framerate), self.capture_and_publish)
            self.loginfo(f"Camera driver initialized successfully for {self.veh_name}")
        else:
            self.logerr("Failed to initialize camera")
    
    def detect_camera_type(self):
        """Auto-detect available camera type."""
        self.loginfo("Auto-detecting camera type...")
        
        # Check for Jetson CSI first (to avoid mis-detecting as USB)
        if self.check_jetson_csi():
            return "jetson_csi"
        
        # Check for Raspberry Pi camera
        if self.check_pi_camera():
            return "pi"
        
        # Check for USB cameras
        if self.check_usb_camera():
            return "usb"
        
        self.logwarn("No camera detected")
        return None

    def check_jetson_csi(self):
        """Heuristics to detect a Jetson platform with CSI camera support."""
        try:
            model_path = "/proc/device-tree/model"
            if os.path.exists(model_path):
                with open(model_path, 'r') as f:
                    model = f.read().lower()
                if "nvidia" in model or "jetson" in model or "tegra" in model:
                    # nvargus-daemon is the CSI camera service; presence is a strong signal
                    try:
                        result = subprocess.run(["pgrep", "-x", "nvargus-daemon"], capture_output=True)
                        if result.returncode == 0:
                            self.loginfo("Detected Jetson platform with nvargus-daemon running")
                            return True
                    except Exception:
                        # Even if pgrep fails, being on Jetson is indicative
                        self.loginfo("Detected Jetson platform (no pgrep)")
                        return True
        except Exception:
            pass
        return False
    
    def check_pi_camera(self):
        """Check if Raspberry Pi camera is available."""
        try:
            # Try modern libcamera approach first
            result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and "Available cameras" in result.stdout:
                self.loginfo("Detected Raspberry Pi camera (libcamera)")
                return True
        except:
            pass
        
        try:
            # Try legacy raspistill approach
            result = subprocess.run(['raspistill', '--help'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.loginfo("Detected Raspberry Pi camera (raspistill)")
                return True
        except:
            pass
        
        # Check for camera device files
        pi_camera_paths = ['/dev/video0', '/sys/class/video4linux/video0']
        for path in pi_camera_paths:
            if os.path.exists(path):
                self.loginfo(f"Found camera device at {path}")
                return True
        
        return False
    
    def check_usb_camera(self):
        """Check if USB camera is available."""
        # Check for video devices
        for i in range(5):  # Check /dev/video0 through /dev/video4
            device_path = f'/dev/video{i}'
            if os.path.exists(device_path):
                try:
                    # Try to open with OpenCV
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None:
                            self.loginfo(f"Detected USB camera at {device_path}")
                            return True
                except:
                    pass
        
        return False
    
    def initialize_camera(self):
        """Initialize the appropriate camera interface."""
        # Determine camera type
        if self.camera_type == "auto":
            detected_type = self.detect_camera_type()
            if detected_type:
                self.camera_type = detected_type
            else:
                self.logerr("No camera detected and no manual type specified")
                return
        
        self.loginfo(f"Initializing {self.camera_type} camera...")
        
        if self.camera_type == "pi":
            self.camera = self.initialize_pi_camera()
        elif self.camera_type == "usb":
            self.camera = self.initialize_usb_camera()
        elif self.camera_type == "jetson_csi":
            self.camera = self.initialize_jetson_csi_camera()
        else:
            self.logerr(f"Unsupported camera type: {self.camera_type}")
    
    def initialize_pi_camera(self):
        """Initialize Raspberry Pi camera."""
        try:
            # Try OpenCV first (works with most Pi camera setups)
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.res_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res_h)
                cap.set(cv2.CAP_PROP_FPS, self.framerate)
                
                # Test capture
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.loginfo("Raspberry Pi camera initialized via OpenCV")
                    return cap
                else:
                    cap.release()
            
            # If OpenCV fails, we could implement libcamera or raspistill capture here
            self.logwarn("OpenCV camera initialization failed")
            return None
        except Exception as e:
            self.logerr(f"Failed to initialize Pi camera: {e}")
            return None
            
    def initialize_jetson_csi_camera(self):
        """Initialize Jetson CSI camera via GStreamer (nvarguscamerasrc)."""
        try:
            # Build nvarguscamerasrc pipeline
            sensor_mode_part = f" sensor-mode={int(self.sensor_mode)}" if self.sensor_mode is not None else ""
            gst_pipeline = (
                f"nvarguscamerasrc{sensor_mode_part} ! "
                f"video/x-raw(memory:NVMM), width=(int){self.res_w}, height=(int){self.res_h}, framerate=(fraction){self.framerate}/1 ! "
                f"nvvidconv flip-method={int(self.flip_method)} ! "
                "video/x-raw, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! "
                "appsink drop=1 max-buffers=1"
            )

            self.loginfo(f"Using GStreamer pipeline: {gst_pipeline}")
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                # Test capture
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.loginfo("Jetson CSI camera initialized via GStreamer")
                    return cap
                else:
                    cap.release()
                    self.logwarn("GStreamer opened but failed to read a frame")
            else:
                self.logerr("Failed to open GStreamer pipeline. Ensure OpenCV is built with GStreamer support.")
            return None
        except Exception as e:
            self.logerr(f"Failed to initialize Jetson CSI camera: {e}")
            return None
    
    def initialize_usb_camera(self):
        """Initialize USB camera."""
        try:
            # Try different video devices
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.res_w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res_h)
                    cap.set(cv2.CAP_PROP_FPS, self.framerate)
                    
                    # Test capture
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.loginfo(f"USB camera initialized on /dev/video{i}")
                        return cap
                    else:
                        cap.release()
            
            self.logwarn("No working USB camera found")
            return None
            
        except Exception as e:
            self.logerr(f"Failed to initialize USB camera: {e}")
            return None
    
    def capture_and_publish(self, event):
        """Capture image and publish to ROS topics."""
        if self.camera is None:
            return
        
        try:
            # Capture frame
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                self.logwarn("Failed to capture frame")
                return
            
            # Resize if needed
            if frame.shape[1] != self.res_w or frame.shape[0] != self.res_h:
                frame = cv2.resize(frame, (self.res_w, self.res_h))
            
            # Convert to ROS message
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
            compressed_msg.header.stamp = rospy.Time.now()
            compressed_msg.header.frame_id = f"{self.veh_name}/camera_optical_frame"
            
            # Publish image
            self.pub_image.publish(compressed_msg)
            
            # Publish camera info
            self.camera_info.header = compressed_msg.header
            self.pub_camera_info.publish(self.camera_info)
            
        except Exception as e:
            self.logerr(f"Error capturing/publishing image: {e}")
    
    def create_camera_info(self):
        """Create camera info message with basic calibration."""
        camera_info = CameraInfo()
        camera_info.width = self.res_w
        camera_info.height = self.res_h
        
        # Basic camera matrix (you should calibrate your specific camera)
        fx = fy = self.res_w * 0.8  # Approximate focal length
        cx = self.res_w / 2.0
        cy = self.res_h / 2.0
        
        camera_info.K = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]
        
        camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
        camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Identity
        
        camera_info.P = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        
        return camera_info
    
    def __del__(self):
        """Cleanup camera resources."""
        if self.camera is not None:
            self.camera.release()


if __name__ == "__main__":
    node = CameraDriverNode("camera_node")
    rospy.spin()