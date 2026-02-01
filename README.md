# Real-Time Vision for Socially Aware Robots

Implementation of Real-Time Vision for Socially Aware Robots: Gesture, Pointing,
and Visual Engagement Estimation. It is a compact ROS4HRI extension providing lightweight, real-time visual social cues for embedded robots.

**Key features**
- Head-gesture detection (yes/no via 2D facial spans)
- Hand gesture and hybrid pointing (finger-based close range; elbow/eyes fallback at distance)
- Multi-target visual engagement (person⇄person, person⇄robot, person⇄object)
- Optimised for embedded platforms (tested on Jetson Orin Nano)
- Integrates MediaPipe for landmarks and YOLOv8 for object detection

**Requirements**
- Linux (Ubuntu recommended)
- ROS / ROS4HRI (place this package inside your ROS workspace)
- Python 3.8+ with OpenCV, MediaPipe, PyTorch/Ultralytics for YOLOv8