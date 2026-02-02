# EMOROBCARE_CV_OBJECT_DETECTION

This repository provides an implementation of a ROS2 node for object detection via ultralytics YOLO models.

## Features

- Real-time object detection using YOLO models
- Support for USB cameras and RealSense cameras
- Knowledge Base integration for object tracking
- Human Radar integration for spatial object visualization
- TF2 broadcasting for object transforms
- Depth-based 3D positioning (optional)
- Temporal filtering for stable detections
- On-demand model loading/unloading via ROS2 services
- Debug image visualization with bounding boxes

## Requirements

### 1. Python packages

- pytorch
- ultralytics
- cv_bridge
- numpy
- opencv-python

### 2. ROS2 packages

- emorobcare_cv_messages
- sensor_msgs
- std_srvs
- tf2_ros
- geometry_msgs
- knowledge_core (optional, for Knowledge Base integration)
- my_game_interface (optional, for Human Radar integration)

## Configuration

The configuration file is located at `config/config.yaml`. All parameters are documented below:

### Camera Settings

- **camera_type** (default=`usb_cam`, options: `usb_cam`, `realsense`)
  The camera type used for image input.

- **tf_reference_frame** (default=`camera_color_optical_frame`)
  TF reference frame for object transforms (e.g., `map`, `odom`, `base_link`, or camera frame).

### YOLO Model Settings

- **yolo_model_path** (default=`best_v5.pt`)
  Filename of the YOLO model (must be placed in the `models/` directory).

- **score_threshold** (default=`0.5`)
  Confidence threshold to accept a detection (0.0 to 1.0).

- **yolo_device** (default=`cuda:0`)
  Device for YOLO inference (e.g., `cpu`, `cuda:0`, `cuda:1`).

### Performance Settings

- **N_detection_interval** (default=`2`)
  Number of frames to skip between detections (1 = process every frame, 2 = every other frame).

### Visualization Settings

- **draw_image** (default=`false`)
  If `true`, publishes debug images with bounding boxes to `/debug/object_detection`.

### Integration Settings

- **use_human_radar** (default=`false`)
  If `true`, detected objects are displayed in the Human Radar visualization.
  **Note:** Cannot be enabled simultaneously with `use_knowledge_base`.

- **use_knowledge_base** (default=`true`)
  If `true`, detected objects are added to the Knowledge Base for reasoning.
  **Note:** Cannot be enabled simultaneously with `use_human_radar`.

- **use_depth** (default=`false`)
  If `true`, uses depth information for accurate 3D TF positioning.
  If `false`, objects are placed at a fixed distance with 2D pixel-based lateral/vertical positioning.

### Trigger Service Settings

- **use_trigger_service** (default=`false`)
  If `true`, the YOLO model is **not** loaded on startup. Instead, it must be loaded/unloaded via ROS2 services.
  If `false`, the model loads automatically on node startup (default behavior).

### Temporal Filtering Settings

- **ADD_THRESHOLD** (default=`0.5`)
  Percentage of past frames (0.0 to 1.0) in which an object must appear before being added to the Knowledge Base or Human Radar.

- **REMOVE_THRESHOLD** (default=`0.02`)
  Percentage of past frames (0.0 to 1.0) in which an object must be absent before being removed from the Knowledge Base or Human Radar.

- **N_kb_object_detection** (default=`30`)
  Length of detection history (in frames) used for `ADD_THRESHOLD` and `REMOVE_THRESHOLD` calculations.

## Topics

### Subscribed Topics

- `/camera/image_raw` (sensor_msgs/Image) - Camera feed for USB cameras
- `/realsense/color/image_raw` (sensor_msgs/Image) - Camera feed for RealSense cameras
- `/depth_map` (sensor_msgs/Image) - Depth map for 3D positioning (optional)

### Published Topics

- `/detected_objects` (emorobcare_cv_msgs/ObjectDetections) - Detected objects with bounding boxes and labels
- `/debug/object_detection` (sensor_msgs/Image) - Debug visualization with bounding boxes (if `draw_image` is enabled)
- `/processing_time` (emorobcare_cv_msgs/ProcessingTime) - Processing time metrics
- **TF transforms** - Object positions broadcasted as TF transforms (e.g., `object_tomato`, `object_pear`)

## Services

### Trigger Services (when `use_trigger_service: true`)

- **`/start_detection`** (std_srvs/Trigger)
  Loads the YOLO model and activates object detection.

  ```bash
  ros2 service call /start_detection std_srvs/srv/Trigger
  ```

  **Response:**
  - `success: true` - Model loaded successfully
  - `success: false` - Model already active or loading failed
  - `message` - Detailed status message

- **`/stop_detection`** (std_srvs/Trigger)
  Unloads the YOLO model and deactivates object detection (frees GPU/CPU memory).

  ```bash
  ros2 service call /stop_detection std_srvs/srv/Trigger
  ```

  **Response:**
  - `success: true` - Model unloaded successfully
  - `success: false` - Model not active or unloading failed
  - `message` - Detailed status message

### Service Integration (when `use_human_radar: true`)

- `/sim_scene/place_object` (my_game_interface/PlaceObject) - Used to place objects in Human Radar
- `/sim_scene/remove_object` (my_game_interface/EditObject) - Used to remove objects from Human Radar

## Usage Examples

### Default Mode (Auto-load on startup)

```yaml
# config/config.yaml
use_trigger_service: false
```

Launch the node:
```bash
ros2 run emorobcare_cv_object_detection object_detection_node
```

The model loads automatically and starts detecting immediately.

### Trigger Mode (On-demand loading)

```yaml
# config/config.yaml
use_trigger_service: true
```

Launch the node:
```bash
ros2 run emorobcare_cv_object_detection object_detection_node
```

The node starts but detection is inactive. Activate it when needed:
```bash
# Start detection
ros2 service call /start_detection std_srvs/srv/Trigger

# Stop detection (frees GPU memory)
ros2 service call /stop_detection std_srvs/srv/Trigger
```

**Use case:** This is useful for saving GPU memory when object detection is not continuously needed, or for coordinating detection with other modules in the system.

## Node Architecture

The node follows this processing pipeline:

1. **Image Acquisition**: Receives camera frames from ROS2 topics
2. **Model Inference**: Runs YOLO detection (if model is active)
3. **Temporal Filtering**: Tracks detections over time to reduce noise
4. **Depth Processing**: Computes 3D positions using depth maps (optional)
5. **Integration**: Updates Knowledge Base or Human Radar
6. **TF Broadcasting**: Publishes object transforms continuously
7. **Visualization**: Publishes debug images (optional)

## Notes

- **Mutual Exclusivity**: `use_knowledge_base` and `use_human_radar` cannot both be enabled simultaneously.
- **Depth Maps**: The `/depth_map` topic is expected to provide single-channel depth values. If using MiDaS with BGR8 output, adjust the encoding in `depth_callback()`.
- **Model Location**: YOLO models must be placed in the `models/` directory within the package.
- **GPU Memory Management**: When using trigger services, stopping detection properly frees GPU memory through garbage collection and CUDA cache clearing.
