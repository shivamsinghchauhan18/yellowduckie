cmake_minimum_required(VERSION 3.0.2)
project(camera_mock)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  duckietown_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  src/camera_mock_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)