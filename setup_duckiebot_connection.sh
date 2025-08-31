#!/bin/bash

# Setup script to connect to Duckiebot camera
# Usage: ./setup_duckiebot_connection.sh [ROBOT_NAME] [ROBOT_IP]

ROBOT_NAME=${1:-blueduckie}
ROBOT_IP=${2:-"auto"}

echo "🤖 Setting up connection to Duckiebot: $ROBOT_NAME"

# Set environment variables
export VEHICLE_NAME=$ROBOT_NAME

if [ "$ROBOT_IP" = "auto" ]; then
    echo "🔍 Auto-detecting robot IP..."
    # Try to ping the robot using .local domain
    ROBOT_HOSTNAME="${ROBOT_NAME}.local"
    
    if ping -c 1 $ROBOT_HOSTNAME > /dev/null 2>&1; then
        ROBOT_IP=$(ping -c 1 $ROBOT_HOSTNAME | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -1)
        echo "✅ Found robot at: $ROBOT_IP"
    else
        echo "❌ Could not find robot at $ROBOT_HOSTNAME"
        echo "Please provide the robot IP manually:"
        echo "  ./setup_duckiebot_connection.sh $ROBOT_NAME <ROBOT_IP>"
        exit 1
    fi
else
    echo "📡 Using provided IP: $ROBOT_IP"
fi

# Set ROS Master URI
export ROS_MASTER_URI="http://${ROBOT_IP}:11311"
export ROS_IP=$(hostname -I | awk '{print $1}')

echo "🔧 ROS Configuration:"
echo "  ROS_MASTER_URI: $ROS_MASTER_URI"
echo "  ROS_IP: $ROS_IP"
echo "  VEHICLE_NAME: $VEHICLE_NAME"

# Test connection
echo "🧪 Testing ROS connection..."
if timeout 10 rostopic list > /dev/null 2>&1; then
    echo "✅ ROS connection successful!"
    
    # Check for camera topics
    echo "📷 Checking for camera topics..."
    CAMERA_TOPICS=$(rostopic list | grep -E "(camera|image)" | head -5)
    
    if [ -n "$CAMERA_TOPICS" ]; then
        echo "✅ Camera topics found:"
        echo "$CAMERA_TOPICS"
        
        # Test camera data
        CAMERA_TOPIC="/${VEHICLE_NAME}/camera_node/image/compressed"
        echo "🎥 Testing camera data on: $CAMERA_TOPIC"
        
        if timeout 5 rostopic echo $CAMERA_TOPIC -n 1 > /dev/null 2>&1; then
            echo "✅ Camera is publishing data!"
        else
            echo "❌ No camera data received"
            echo "💡 Make sure the camera_node is running on your robot"
        fi
    else
        echo "❌ No camera topics found"
        echo "💡 Make sure your Duckiebot's camera stack is running"
    fi
else
    echo "❌ Could not connect to ROS master"
    echo "💡 Check that:"
    echo "  - Your robot is powered on and connected to the network"
    echo "  - The IP address is correct: $ROBOT_IP"
    echo "  - ROS is running on the robot"
fi

echo ""
echo "🚀 To use these settings in your current shell, run:"
echo "  source setup_duckiebot_connection.sh $ROBOT_NAME $ROBOT_IP"
echo ""
echo "🔧 Or set them manually:"
echo "  export VEHICLE_NAME=$VEHICLE_NAME"
echo "  export ROS_MASTER_URI=$ROS_MASTER_URI"
echo "  export ROS_IP=$ROS_IP"