cmake_minimum_required(VERSION 3.0.2)
project(lane_change_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  duckietown_msgs
  message_generation
)

add_message_files(
  FILES
  LaneChangeRequest.msg
  LaneChangeStatus.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs duckietown_msgs message_runtime
)