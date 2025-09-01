# Speed constraints based on environmental and safety conditions
Header header

# Base speed limits
float32 base_max_linear_velocity   # m/s
float32 base_max_angular_velocity  # rad/s

# Environmental constraints
float32 visibility_speed_limit     # m/s, based on visibility conditions
float32 traffic_speed_limit        # m/s, based on traffic density
float32 road_condition_speed_limit # m/s, based on road conditions

# Safety constraints
float32 following_distance_limit   # m/s, based on following distance
float32 collision_avoidance_limit  # m/s, based on collision risk
float32 intersection_speed_limit   # m/s, near intersections

# Dynamic constraints
float32 acceleration_limit         # m/s²
float32 deceleration_limit        # m/s²
float32 jerk_limit                # m/s³

# Constraint sources and confidence
string[] active_constraints       # list of active constraint types
float32[] constraint_confidence   # confidence in each constraint (0.0-1.0)
bool emergency_constraints_active # true if emergency constraints are active

# Timing information
float32 constraint_update_rate    # Hz, rate at which constraints are updated
float32 constraint_validity_time  # seconds, how long constraints are valid