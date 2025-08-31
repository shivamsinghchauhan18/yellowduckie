cmake_minimum_required(VERSION 3.0.2)
project(predictive_perception)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  PredictedTrajectory.msg
  PredictedTrajectoryArray.msg
  TrackingConfidence.msg
  TrackedObject.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs message_runtime
)