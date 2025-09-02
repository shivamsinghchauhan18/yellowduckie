# Lane change status message
Header header

# Current state of lane change controller
# 0=LANE_FOLLOWING, 1=ASSESSING_CHANGE, 2=SIGNALING, 3=EXECUTING, 4=VERIFYING, 5=ABORTING
int8 state

# Active lane change request (if any)
LaneChangeRequest active_request

# Progress of current maneuver (0.0 to 1.0)
float32 progress

# Time remaining for current phase (seconds)
float32 time_remaining

# Safety status
bool is_safe_to_proceed

# Abort reason (if aborting)
string abort_reason