# Predicted trajectory for a tracked object
uint32 object_id
geometry_msgs/Point[] trajectory_points
float32 confidence
float32 time_horizon
geometry_msgs/Vector3 velocity
geometry_msgs/Vector3 acceleration
std_msgs/Header header