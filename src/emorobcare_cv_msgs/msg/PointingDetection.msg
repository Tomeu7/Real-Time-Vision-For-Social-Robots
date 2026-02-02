# PointingDetection.msg
std_msgs/Header header

# Person identification
string body_id

# Gesture recognition
bool is_pointing
string hand

# Pointing direction
float32 base_x
float32 base_y
float32 direction_x
float32 direction_y
float32 base_z
float32 direction_z

# Intersection
string target_label 
float32 target_x1
float32 target_y1
float32 target_x2
float32 target_y2

# Hand bounding box (possibly removed later)
float32 hand_x1
float32 hand_y1
float32 hand_x2
float32 hand_y2