# Individual engagement detail for debugging
# Used within MultiUserEngagementLevel to show all computed engagements

# Target ID (can be "robot" or person_id)
string target_id

# Raw engagement score with this target (0.0 to 1.0)
# This is the Webb & Lemaignan score for the current frame
float32 score

# History average (-1.0 to 1.0)
# Average of the binary engagement history (+1/-1) over the observation window
# This is what's actually used to determine the primary target
float32 history_average
