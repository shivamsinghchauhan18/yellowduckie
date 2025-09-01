# Speed command with environmental and safety constraints
Header header

# Commanded speeds
float32 linear_velocity      # m/s
float32 angular_velocity     # rad/s

# Speed constraints applied
float32 max_linear_velocity  # m/s
float32 max_angular_velocity # rad/s

# Environmental factors
float32 visibility_factor    # 0.0-1.0, 1.0 = perfect visibility
float32 traffic_density_factor # 0.0-1.0, 1.0 = no traffic
float32 road_condition_factor  # 0.0-1.0, 1.0 = perfect conditions

# Safety factors
float32 following_distance   # meters
float32 collision_risk_factor # 0.0-1.0, 1.0 = no risk
float32 safety_margin_factor # 0.0-1.0, 1.0 = maximum safety margin

# Control factors
float32 acceleration_limit   # m/s²
float32 jerk_limit          # m/s³
bool emergency_braking      # true if emergency braking is active

# Source information
string command_source       # source of the speed command
string constraint_reason    # reason for any applied constraints