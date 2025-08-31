#!/bin/bash

# Script to manually start camera node
# Usage: ./start_camera.sh [camera_type]

CAMERA_TYPE=${1:-auto}
VEHICLE_NAME=${VEHICLE_NAME:-blueduckie}

echo "🎥 Starting camera node for $VEHICLE_NAME"
echo "📷 Camera type: $CAMERA_TYPE"

# Set ROS environment if not already set
if [ -z "$ROS_MASTER_URI" ]; then
    echo "⚠️  ROS_MASTER_URI not set. Using localhost."
    export ROS_MASTER_URI=http://localhost:11311
fi

# Check if camera devices exist
echo "🔍 Checking for camera devices..."
if [ -e /dev/video0 ]; then
    echo "✅ Found camera device: /dev/video0"
else
    echo "❌ No camera device found at /dev/video0"
    echo "💡 Make sure your camera is connected and recognized by the system"
fi

# Check camera permissions
if [ -e /dev/video0 ]; then
    if [ -r /dev/video0 ] && [ -w /dev/video0 ]; then
        echo "✅ Camera device permissions OK"
    else
        echo "⚠️  Camera device permissions may be insufficient"
        echo "💡 You may need to run: sudo chmod 666 /dev/video0"
    fi
fi

# Start camera node
echo "🚀 Starting camera node..."
rosrun camera camera_driver_node.py \
    _veh_name:=$VEHICLE_NAME \
    _camera_type:=$CAMERA_TYPE \
    _framerate:=10 \
    _res_w:=640 \
    _res_h:=480 \
    __name:=camera_node \
    __ns:=/$VEHICLE_NAME