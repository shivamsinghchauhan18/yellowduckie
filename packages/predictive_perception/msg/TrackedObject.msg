# Tracked object information
uint32 object_id
uint8 object_class  # 0=duck, 1=duckiebot, 2=other
geometry_msgs/Point position
geometry_msgs/Vector3 velocity
geometry_msgs/Vector3 acceleration
float32 confidence
float64 last_seen_time
geometry_msgs/Polygon bounding_box