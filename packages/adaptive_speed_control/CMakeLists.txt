cmake_minimum_required(VERSION 3.0.2)
project(adaptive_speed_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  duckietown_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  src/adaptive_speed_controller_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)