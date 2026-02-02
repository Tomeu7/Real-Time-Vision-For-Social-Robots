# Multi-user engagement level message
# Describes the overall engagement of a person based on maximum engagement with any target

std_msgs/Header header

# ID of the person being observed
string person_id

# ID of the target with highest engagement score
# - "robot" for robot engagement
# - person_id for person-to-person engagement
string primary_target_id

# Engagement level constants (same as hri_msgs/EngagementLevel)
uint8 UNKNOWN=0
uint8 DISENGAGED=1
uint8 ENGAGING=2
uint8 ENGAGED=3
uint8 DISENGAGING=4

# Overall engagement level (based on highest engagement score)
uint8 level

# Highest engagement score (0.0 to 1.0) from Webb & Lemaignan formula
float32 max_score

# Array of all engagement scores for debugging
# Shows engagement with robot and all other persons
EngagementDetail[] engagement_details
