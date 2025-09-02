cmake_minimum_required(VERSION 3.0.2)
project(custom_camera)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  duckietown_msgs
  cv_bridge
)

catkin_package()

# Intentionally not installing any executables; use official Duckietown camera package