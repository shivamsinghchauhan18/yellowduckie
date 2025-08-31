#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# Set environment variables to force CPU-only mode
export CUDA_VISIBLE_DEVICES=""
export TORCH_USE_CUDA_DSA=0

# Ensure ROS environment is properly sourced
source /opt/ros/noetic/setup.bash
source ${CATKIN_WS_DIR}/devel/setup.bash

# launching app
dt-exec roslaunch --wait duckietown_demos lane_following.launch


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
