# emorobcare_hri_yesno_recognition

## Overview
ROS2 node for detecting head gestures (yes/no) from detected faces using HRI (Human-Robot Interaction) framework.

## How it works
1. **Face Detection**: Uses `HRIListener` to subscribe to detected faces from the HRI framework
2. **Tracking Method Selection** (configurable):
   - **ROI Center**: Tracks the center of face bounding box (simple, fast)
   - **Landmark Span** (recommended): Tracks facial landmark distances (robust to body movement)
     - For "No": Tracks ear-to-ear horizontal span (changes with head rotation)
     - For "Yes": Tracks nose-to-chin vertical distance (changes with head pitch)
3. **Movement Analysis**: Maintains a history buffer (default: 16 frames) for each tracking method
4. **Gesture Detection with Robustness Checks**:
   - **Yes**: Vertical movement exceeds horizontal movement by threshold ratio
     - ROI method: Y-axis movement of face center
     - Landmark method: Nose-to-chin distance oscillation (perspective change)
     - Optional: Requires oscillation pattern (up-down-up) to avoid false positives from drift
   - **No**: Horizontal movement exceeds vertical movement by threshold ratio
     - ROI method: X-axis movement of face center
     - Landmark method: Ear-to-ear span oscillation (rotation-based)
     - Optional: Requires center-crossing (left-right-left pattern) to ensure actual head shake
     - Optional: Requires oscillation pattern to avoid false positives from drift
5. **Publishing**: Publishes detected gestures to `/humans/faces/{face_id}/head_gesture`
6. **Debug Mode**: Shows both methods simultaneously for comparison

## Key Features
- Per-face gesture detection (supports multiple faces simultaneously)
- **Two tracking methods**:
  - ROI center tracking (simple, fast)
  - Landmark span tracking (robust to body movement, recommended)
- **Improved robustness**:
  - Center-crossing detection for "no" gesture (ensures left-right-left pattern)
  - Oscillation detection (counts peaks/valleys to verify gesture pattern)
  - Both checks are configurable and can be disabled
- Configurable thresholds and history length
- Debug visualization showing both methods simultaneously for comparison
- Throttled logging to avoid spam
- **Future-ready**: Feature extraction structured for easy migration to ML classifier

## Configuration Options

### Tracking Method
- `tracking_method`: Choose between "roi_center" or "landmark_span" (default: "landmark_span")
- `yes_threshold_span_change`: Relative change threshold for landmark-based "yes" (default: 0.05)
- `no_threshold_span_change`: Relative change threshold for landmark-based "no" (default: 0.10)

### Robustness Settings
- `enable_center_crossing`: Require head to cross center point for "no" (default: true)
- `min_crossings`: Minimum center crossings for "no" gesture (default: 2)
- `enable_oscillation_check`: Require oscillation pattern (default: true)
- `min_oscillations`: Minimum peaks/valleys for gesture (default: 2)

### Traditional ROI Method (still available)
- `yes_threshold_y`: Pixel threshold for ROI-based "yes" (default: 20)
- `no_threshold_x`: Pixel threshold for ROI-based "no" (default: 20)
- `ratio_threshold`: Movement ratio threshold (default: 2.0)
