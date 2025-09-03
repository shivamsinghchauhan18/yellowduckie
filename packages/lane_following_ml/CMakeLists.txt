cmake_minimum_required(VERSION 3.0.2)
project(lane_following_ml)

find_package(catkin REQUIRED COMPONENTS
  rospy
  duckietown_msgs
  sensor_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  src/lane_following_onnx_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)