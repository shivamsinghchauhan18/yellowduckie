# Lane change request message
Header header

# Direction: -1=LEFT, 1=RIGHT
int8 direction

# Reason for lane change
string reason

# Urgency level: 1=LOW, 2=MEDIUM, 3=HIGH
int8 urgency

# Feasibility score (0.0 to 1.0)
float32 feasibility_score

# Estimated duration in seconds
float32 estimated_duration

# Current lane pose when request was made
geometry_msgs/Point32 current_position
float32 current_heading