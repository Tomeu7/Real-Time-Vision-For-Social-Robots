# Detailed geometry metrics for engagement computation
# Contains low-level geometric values used in Webb & Lemaignan formula

std_msgs/Header header

# ID of the person being observed
string person_id

# ID of the target (robot, tablet, or another person)
string target_id

# Distance between person and target (meters)
float32 distance

# Logarithmic distance term: log(-d + max_distance + 1) / log(max_distance + 1)
float32 log_distance

# Gaze from A to B (person looking at target)
# 0.0 = not looking, 1.0 = directly looking
float32 gaze_ab

# Gaze from B to A (target looking at person)
# For robot: based on camera direction
# For persons: based on their gaze
# 0.0 = not looking, 1.0 = directly looking
float32 gaze_ba

# Mutual gaze: gaze_ab * gaze_ba
float32 mutual_gaze

# Final engagement score: min(1, mutual_gaze * log_distance)
float32 score

# Position coordinates (for debugging)
# xa, ya, za: position of target in person's frame
float32 xa
float32 ya
float32 za

# xb, yb, zb: position of person in target's frame
float32 xb
float32 yb
float32 zb
