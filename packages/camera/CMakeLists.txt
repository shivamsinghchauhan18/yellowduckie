cmake_minimum_required(VERSION 3.0.2)
project(camera)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  duckietown_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  src/camera_driver_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)