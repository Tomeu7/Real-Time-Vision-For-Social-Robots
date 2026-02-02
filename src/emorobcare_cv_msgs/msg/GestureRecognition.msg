# GestureRecognition.msg
std_msgs/Header header
string handedness           # "left" or "right"
string gesture              
float32[4] bbox             # [x_min, y_min, width, height]
geometry_msgs/Point[] landmarks  # mediapipe landmarks
