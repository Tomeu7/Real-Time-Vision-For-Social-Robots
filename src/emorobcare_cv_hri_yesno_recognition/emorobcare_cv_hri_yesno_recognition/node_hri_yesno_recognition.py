import os
import yaml
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Header
from builtin_interfaces.msg import Time, Duration
from emorobcare_cv_msgs.msg import ProcessingTime
from ament_index_python.packages import get_package_share_directory
from hri import HRIListener
from collections import deque, defaultdict
import time

ENABLE_PROCESSING_TIME = False
ENABLE_HEARTBEAT = False


class NodeHRIYesNoRecognition(Node):
    """ROS2 node for detecting yes/no head gestures using HRI face tracking."""

    def __init__(self):
        """Initialize the yes/no head gesture recognition node."""
        super().__init__('hri_yesno_recognition_node')

        config_path = os.path.join(
            get_package_share_directory('emorobcare_cv_hri_yesno_recognition'),
            'config', 'config.yaml'
        )
        self.config = self.load_config(config_path)

        self.camera_type = self.config.get("camera_type", "usb_cam")
        self.camera_topic = "/camera/image_raw" if self.camera_type == "usb_cam" else "/realsense/color/image_raw"

        # Params
        self.history_length = self.config.get("history_length", 16)
        self.yes_threshold_y = self.config.get("yes_threshold_y", 20)
        self.no_threshold_x = self.config.get("no_threshold_x", 20)
        self.ratio_threshold = self.config.get("ratio_threshold", 2.0)

        # Span-based detection thresholds
        self.yes_threshold_span_change = self.config.get("yes_threshold_span_change", 0.05)
        self.no_threshold_span_change = self.config.get("no_threshold_span_change", 0.10)

        # Robustness params
        self.enable_center_crossing = self.config.get("enable_center_crossing", True)
        self.min_crossings = self.config.get("min_crossings", 2)
        self.enable_oscillation_check = self.config.get("enable_oscillation_check", True)
        self.min_oscillations = self.config.get("min_oscillations", 2)

        self.debug_enabled = self.config.get("debug", False)
        self.debug_topic = self.config.get("debug_topic", "/head_gesture/debug_image")

        self.center_crossing_method = self.config.get("center_crossing_method", "median")  # "median" or "first_point"

        self.gesture_cooldown = self.config.get("gesture_cooldown", 0.0)  # seconds, 0 disables
        self._last_detect_time = defaultdict(float)  # face_id -> timestamp

        # Logger to know which backend (YuNet vs MediaPipe) is used
        self.backend_log_period = self.config.get("backend_log_period", 5.0)  # seconds
        self._last_backend_log_time = 0.0

        self.publish_heartbeat = self.config.get("publish_heartbeat", True)

        # Gesture persistence for debug overlay
        self.gesture_display_duration = self.config.get("gesture_display_duration", 2.0)  # seconds
        self.clear_history_on_detection = self.config.get("clear_history_on_detection", True)
        self.last_detected_gesture = {}  # face_id -> (gesture, timestamp)

        # Timestamp validation
        self.enable_timestamp_check = self.config.get("enable_timestamp_check", True)
        self.max_frame_interval = self.config.get("max_frame_interval", 0.5)  # seconds
        self.min_valid_frames = self.config.get("min_valid_frames", 12)
        self.timestamp_discard_count = defaultdict(int)

        # HRI listener
        self.hri = HRIListener("hri_yesno_recognition_node")

        # Store timestamps in history
        # ROI: deque of ((cx, cy), timestamp)
        self.history_roi = defaultdict(lambda: deque(maxlen=self.history_length))

        # Spans: per-face dict of deques holding (value, timestamp)
        self.history_span = defaultdict(lambda: {
            # "classic" spans (kept for backwards compatibility)
            'ear_span': deque(maxlen=self.history_length),
            'nose_chin': deque(maxlen=self.history_length),

            # advanced mediapipe spans (python advanced_landmark_span)
            'eye_span': deque(maxlen=self.history_length),
            'forehead_chin': deque(maxlen=self.history_length),
            'nose_bridge_chin': deque(maxlen=self.history_length),

            # yunet fallback spans (5 keypoints)
            'mouth_span': deque(maxlen=self.history_length),
            'nose_mouth': deque(maxlen=self.history_length),
        })

        self.face_pub = {}

        # Throttling for metrics logging
        self._last_metrics_log_time = {}
        self.metrics_log_period = self.config.get("metrics_log_period", 1.0)
        self._face_frame_count = {}
        self._last_loop_log_time = 0

        self.get_logger().info("Head gesture detector started.")

        # Camera for debug and first frame
        self.bridge = CvBridge()
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.latest_image = None
        self.camera_width = None
        self.camera_height = None

        # Subscribe to camera info to get image dimensions
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            qos_profile=qos_profile
        )

        if self.debug_enabled:
            self.subscription = self.create_subscription(
                Image,
                self.camera_topic,
                self.image_callback,
                qos_profile=qos_profile
            )
            self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)

        # Heartbeat publisher
        if ENABLE_HEARTBEAT and self.publish_heartbeat:
            self.heartbeat_pub = self.create_publisher(Header, '/system/health/yesno', 10)
            self.get_logger().info('Heartbeat publishing enabled on /system/health/yesno')
        else:
            self.heartbeat_pub = None

        self._last_log_time = {}
        self.timer = self.create_timer(0.2, self.loop_over_faces)

        if ENABLE_PROCESSING_TIME:
            self.processing_time_pub = self.create_publisher(
                ProcessingTime,
                '/processing_time/yesno',
                10
            )

    def process_time_wall(self, t_in_wall: float):
        """Publish wall-clock processing time measurement."""
        t_out_wall = time.time()
        delta_s = t_out_wall - t_in_wall

        processing_time_msg = ProcessingTime()

        processing_time_msg.id = "yesno"

        processing_time_msg.t_in = Time(sec=int(t_in_wall), nanosec=int((t_in_wall % 1) * 1e9))
        processing_time_msg.t_out = Time(sec=int(t_out_wall), nanosec=int((t_out_wall % 1) * 1e9))

        delta = Duration()
        delta.sec = int(delta_s)
        delta.nanosec = int((delta_s - int(delta_s)) * 1e9)

        processing_time_msg.delta = delta
        self.processing_time_pub.publish(processing_time_msg)

    def load_config(self, path):
        """Load YAML configuration file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def camera_info_callback(self, msg):
        """Store camera dimensions from CameraInfo message"""
        if self.camera_width is None or self.camera_height is None:
            self.camera_width = msg.width
            self.camera_height = msg.height
            self.get_logger().info(f"Camera dimensions received: {self.camera_width}x{self.camera_height}")

    ##################
    # 0. Loop over faces
    ##################
    def log_throttled(self, face_id, gesture, period=1.0):
        """Log gesture detection with rate limiting per face."""
        now = time.time()
        if now - self._last_log_time.get(face_id, 0) > period:
            self.get_logger().info(f"[{face_id}] Gesture: {gesture}")
            self._last_log_time[face_id] = now

    def increment_frame_count(self, face_id):
        """Increment per-face frame counter for metrics logging."""
        self._face_frame_count[face_id] = self._face_frame_count.get(face_id, 0) + 1

    def should_log_metrics(self, face_id):
        """Check if metrics should be logged based on throttle period. Returns (should_log, frame_count)."""
        now = time.time()
        if now - self._last_metrics_log_time.get(face_id, 0) > self.metrics_log_period:
            self._last_metrics_log_time[face_id] = now
            frame_count = self._face_frame_count.get(face_id, 0)
            self._face_frame_count[face_id] = 0
            return True, frame_count
        return False, 0

    def cleanup(self):
        """Remove tracking data for faces no longer detected by HRI."""
        current_face_ids = set(self.hri.faces.keys()) if self.hri.faces else set()
        tracked_face_ids = set(self.face_pub.keys())

        for old_face_id in tracked_face_ids - current_face_ids:
            self.get_logger().info(f"Cleaning up face {old_face_id} - no longer tracked")
            del self.face_pub[old_face_id]

            if old_face_id in self.history_roi:
                del self.history_roi[old_face_id]
            if old_face_id in self.history_span:
                del self.history_span[old_face_id]
            if old_face_id in self._last_log_time:
                del self._last_log_time[old_face_id]
            if old_face_id in self._last_metrics_log_time:
                del self._last_metrics_log_time[old_face_id]
            if old_face_id in self._face_frame_count:
                del self._face_frame_count[old_face_id]

            # Clean persistence / discards
            if old_face_id in self.last_detected_gesture:
                del self.last_detected_gesture[old_face_id]
            if old_face_id in self.timestamp_discard_count:
                del self.timestamp_discard_count[old_face_id]
            if old_face_id in self._last_detect_time:
                del self._last_detect_time[old_face_id]

    def validate_frame_timing(self, history_with_timestamps, face_id):
        """Validate temporal consistency of frame history to reject noisy detections.

        Returns (is_valid, num_valid_frames, max_gap, reason).
        """
        if not self.enable_timestamp_check:
            return True, len(history_with_timestamps), 0.0, "Timestamp check disabled"

        if len(history_with_timestamps) < 2:
            return False, len(history_with_timestamps), 0.0, "Insufficient frames"

        timestamps = [item[1] for item in history_with_timestamps]
        time_gaps = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        max_gap = max(time_gaps) if time_gaps else 0.0

        valid_gaps = sum(1 for gap in time_gaps if gap <= self.max_frame_interval)
        num_valid_frames = valid_gaps + 1

        if num_valid_frames < self.min_valid_frames:
            reason = f"Only {num_valid_frames}/{self.min_valid_frames} valid frames (max gap: {max_gap:.3f}s)"
            return False, num_valid_frames, max_gap, reason

        if max_gap > self.max_frame_interval:
            reason = f"Frame gap too large: {max_gap:.3f}s > {self.max_frame_interval}s"
            return False, num_valid_frames, max_gap, reason

        return True, num_valid_frames, max_gap, "Valid timing"

    def loop_over_faces(self):
        """Main timer callback: iterate over HRI faces and detect yes/no gestures."""
        if ENABLE_PROCESSING_TIME:
            t_in = time.time()
            

        now = time.time()
        if now - self._last_loop_log_time > self.metrics_log_period:
            self.get_logger().info(
                f"loop_over_faces called - faces detected: {len(self.hri.faces) if self.hri.faces else 0}"
            )
            self._last_loop_log_time = now

        self.cleanup()

        if not self.hri.faces:
            return

        if self.camera_width is None or self.camera_height is None:
            self.get_logger().warn("Camera dimensions not yet available from CameraInfo")
            return

        w, h = self.camera_width, self.camera_height
        overlays = []

        for face_id, face in self.hri.faces.items():
            if not face.valid:
                continue

            roi = getattr(face, "roi", None)
            if not roi:
                continue

            # Get timestamp from face transform (preferred) with fallback
            transform = getattr(face, "transform", None)
            if transform and hasattr(transform, 'header'):
                stamp = transform.header.stamp
                current_timestamp = stamp.sec + stamp.nanosec * 1e-9
            else:
                current_timestamp = self.get_clock().now().nanoseconds * 1e-9

            self.increment_frame_count(face_id)

            # ROI tracking
            x, y, rw, rh = roi
            cx, cy = (x + rw / 2.0) * w, (y + rh / 2.0) * h
            self.history_roi[face_id].append(((cx, cy), current_timestamp))

            # Decide backend + compute spans
            backend, spans = self.compute_landmark_spans(face)

            # Push spans into history (each as (value, timestamp))
            for k, v in spans.items():
                if k in self.history_span[face_id]:
                    self.history_span[face_id][k].append((v, current_timestamp))

            # Optional alt method for debug (keep placeholder for overlay compatibility)
            gesture_alt = None

            # Validate timing before detecting gestures
            is_valid_timing, num_valid, max_gap, reason = self.validate_frame_timing(self.history_roi[face_id], face_id)
            if not is_valid_timing:
                if now - self._last_log_time.get(f"{face_id}_timing", 0) > 2.0:
                    self.get_logger().warn(f"[{face_id}] Invalid frame timing: {reason}")
                    self._last_log_time[f"{face_id}_timing"] = now
                self.publish_gesture(face_id, "Normal")
                self.timestamp_discard_count[face_id] += 1
                if self.timestamp_discard_count[face_id] % 10 == 0:
                    self.get_logger().info(
                        f"[{face_id}] {self.timestamp_discard_count[face_id]} frames discarded due to timestamp gaps"
                    )
                continue

            # Decide which detector to use:
            # - If MEDIAPIPE landmarks present: use advanced span method
            # - Else if YUNET: use yunet span method
            # - Else fallback to ROI
            gesture = None
            method_used = "ROI"
            if self._cooldown_ok(face_id, current_timestamp):
                if backend == "MEDIAPIPE":
                    method_used = "MEDIAPIPE_ADV"
                    gesture = self.detect_yes_no_advanced_span_mediapipe(face_id)
                    # For debug overlay, compare with ROI (optional, cheap)
                    if self.debug_enabled:
                        gesture_alt = self.detect_yes_no_roi(face_id)
                elif backend == "YUNET":
                    method_used = "YUNET_SPAN"
                    gesture = self.detect_yes_no_span_yunet(face_id)
                    if self.debug_enabled:
                        gesture_alt = self.detect_yes_no_roi(face_id)
                else:
                    method_used = "ROI"
                    gesture = self.detect_yes_no_roi(face_id)

            if gesture:
                self._last_detect_time[face_id] = current_timestamp
                self.publish_gesture(face_id, gesture)
                self.log_throttled(face_id, gesture)
                self.get_logger().info(f"[{face_id}] Detected gesture: {gesture} (method: {method_used})")

                # Store last detected gesture for debug persistence
                self.last_detected_gesture[face_id] = (gesture, current_timestamp)

                # Clear history optionally
                if self.clear_history_on_detection:
                    self.history_roi[face_id].clear()
                    for k in self.history_span[face_id].keys():
                        self.history_span[face_id][k].clear()
            else:
                self.publish_gesture(face_id, "Normal")

            # Extract points for viz (remove timestamps)
            points_for_viz = [pt[0] for pt in self.history_roi[face_id]]

            # Gesture persistence for viz
            display_gesture = gesture
            if not gesture and face_id in self.last_detected_gesture:
                last_gesture, last_ts = self.last_detected_gesture[face_id]
                if (current_timestamp - last_ts) < self.gesture_display_duration:
                    display_gesture = last_gesture
                else:
                    del self.last_detected_gesture[face_id]

            overlays.append({
                "face_id": face_id,
                "points": points_for_viz,
                "gesture": display_gesture,  # None / Yes / No
                "gesture_alt": gesture_alt,
                "method": method_used
            })

        # Periodic logger: which landmark backend is active
        now_wall = time.time()
        if (now_wall - self._last_backend_log_time) > self.backend_log_period:
            counts = {"MEDIAPIPE": 0, "YUNET": 0, "UNKNOWN": 0}
            for _, f in self.hri.faces.items():
                if not getattr(f, "valid", False):
                    continue
                lm = getattr(f, "landmarks", None)
                b = self._detect_landmark_backend(lm) if lm else "UNKNOWN"
                counts[b] = counts.get(b, 0) + 1
            self.get_logger().info(
                f"Landmark backend usage: MEDIAPIPE={counts['MEDIAPIPE']} YUNET={counts['YUNET']} UNKNOWN={counts['UNKNOWN']}"
            )
            self._last_backend_log_time = now_wall

        if self.debug_enabled and self.latest_image is not None:
            self.publish_debug_image(overlays)

        # Heartbeat publish
        if ENABLE_HEARTBEAT and self.publish_heartbeat and self.heartbeat_pub is not None:
            heartbeat_msg = Header()
            heartbeat_msg.stamp = self.get_clock().now().to_msg()
            heartbeat_msg.frame_id = 'yesno'
            self.heartbeat_pub.publish(heartbeat_msg)

        if ENABLE_PROCESSING_TIME:
            self.process_time_wall(t_in)
            

    ######################################
    # 1. yes no logic - helper methods
    ######################################
    def _detect_landmark_backend(self, landmarks):
        """
        Decide which landmark backend we are receiving.
        - YuNet: 5 keypoints
        - MediaPipe FaceMesh: many landmarks (468-ish)
        """
        if not landmarks:
            return "UNKNOWN"
        n = len(landmarks)
        if n <= 10:
            return "YUNET"
        if n >= 100:
            return "MEDIAPIPE"
        return "UNKNOWN"

    def compute_landmark_spans(self, face):
        """
        Returns:
            backend (str): "YUNET" | "MEDIAPIPE" | "UNKNOWN"
            spans (dict): computed spans (normalized coordinates)
        """
        landmarks = getattr(face, "landmarks", None)
        if not landmarks:
            return "UNKNOWN", {}

        backend = self._detect_landmark_backend(landmarks)

        # Small helper
        def _has_xy(lm):
            return hasattr(lm, "x") and hasattr(lm, "y")

        spans = {}

        try:
            if backend == "MEDIAPIPE":
                # NOTE: these are standard MediaPipe FaceMesh indices.
                # This assumes the hri_face_detect node publishes FaceMesh landmarks in FaceMesh order.
                LEFT_EAR_REGION = 234
                RIGHT_EAR_REGION = 454
                NOSE_TIP = 1
                CHIN = 152

                FOREHEAD = 10
                LEFT_EYE_OUTER = 33
                RIGHT_EYE_OUTER = 263
                NOSE_BRIDGE = 168

                req = [
                    LEFT_EAR_REGION, RIGHT_EAR_REGION, NOSE_TIP, CHIN,
                    FOREHEAD, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, NOSE_BRIDGE
                ]
                if max(req) >= len(landmarks):
                    return "MEDIAPIPE", {}

                left_ear = landmarks[LEFT_EAR_REGION]
                right_ear = landmarks[RIGHT_EAR_REGION]
                nose = landmarks[NOSE_TIP]
                chin = landmarks[CHIN]
                forehead = landmarks[FOREHEAD]
                left_eye_outer = landmarks[LEFT_EYE_OUTER]
                right_eye_outer = landmarks[RIGHT_EYE_OUTER]
                nose_bridge = landmarks[NOSE_BRIDGE]

                if not all(_has_xy(lm) for lm in [
                    left_ear, right_ear, nose, chin, forehead,
                    left_eye_outer, right_eye_outer, nose_bridge
                ]):
                    return "MEDIAPIPE", {}

                # classic spans (ear + nose-chin)
                spans["ear_span"] = abs(right_ear.x - left_ear.x)
                spans["nose_chin"] = abs(nose.y - chin.y)

                # advanced spans (python advanced_landmark_span)
                spans["eye_span"] = abs(right_eye_outer.x - left_eye_outer.x)
                spans["forehead_chin"] = abs(forehead.y - chin.y)
                spans["nose_bridge_chin"] = abs(nose_bridge.y - chin.y)

                return "MEDIAPIPE", spans

            elif backend == "YUNET":
                # Typical 5-point order (common in many detectors): left_eye, right_eye, nose, left_mouth, right_mouth
                # abs() spans makes this robust to left/right swap.
                if len(landmarks) < 5:
                    return "YUNET", {}

                left_eye = landmarks[0]
                right_eye = landmarks[1]
                nose = landmarks[2]
                left_mouth = landmarks[3]
                right_mouth = landmarks[4]

                if not all(_has_xy(lm) for lm in [left_eye, right_eye, nose, left_mouth, right_mouth]):
                    return "YUNET", {}

                spans["eye_span"] = abs(right_eye.x - left_eye.x)
                spans["mouth_span"] = abs(right_mouth.x - left_mouth.x)

                mouth_cy = 0.5 * (left_mouth.y + right_mouth.y)
                spans["nose_mouth"] = abs(nose.y - mouth_cy)

                return "YUNET", spans

            else:
                return "UNKNOWN", {}

        except (IndexError, AttributeError) as e:
            self.get_logger().debug(f"Landmark access error: {e}")
            return backend, {}

    def count_center_crossings(self, values):
        """Count how many times values cross their center line (median or first point)."""
        if len(values) < 3:
            return 0

        if self.center_crossing_method == "first_point":
            center = values[0]
            # compare from the second point onward
            above = values[1] > center
            start_idx = 2
        else:
            center = np.median(values)
            above = values[0] > center
            start_idx = 1

        crossings = 0
        for val in values[start_idx:]:
            current_above = val > center
            if current_above != above:
                crossings += 1
                above = current_above

        return crossings

    def count_oscillations(self, values):
        """Count peaks and valleys (direction changes) in a value sequence."""
        if len(values) < 3:
            return 0

        peaks = 0
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                peaks += 1
            elif values[i] < values[i - 1] and values[i] < values[i + 1]:
                peaks += 1
        return peaks

    def _cooldown_ok(self, face_id, now_ts):
        """Check if enough time has passed since last detection for this face."""
        if self.gesture_cooldown <= 0.0:
            return True
        last = self._last_detect_time.get(face_id, 0.0)
        return (now_ts - last) >= self.gesture_cooldown

    ######################################
    # 2a. ROI-based detection
    ######################################
    def detect_yes_no_roi(self, face_id):
        """Detect yes/no gesture from ROI center point movement over time."""
        history = self.history_roi[face_id]
        if len(history) < self.history_length:
            return None

        # History contains ((cx, cy), timestamp)
        x_vals = [pt[0][0] for pt in history]
        y_vals = [pt[0][1] for pt in history]

        x_range = max(x_vals) - min(x_vals)
        y_range = max(y_vals) - min(y_vals)

        should_log, frame_count = self.should_log_metrics(face_id)
        if should_log:
            self.get_logger().info(
                f"[{face_id}] ROI metrics (frames: {frame_count}): "
                f"x_range={x_range:.1f}, y_range={y_range:.1f}, ratio={y_range/x_range if x_range > 0 else 0:.2f}"
            )

        if y_range > self.ratio_threshold * x_range and y_range > self.yes_threshold_y:
            if self.enable_oscillation_check:
                oscillations = self.count_oscillations(y_vals)
                if oscillations < self.min_oscillations:
                    return None
            return "Yes"

        elif x_range > self.ratio_threshold * y_range and x_range > self.no_threshold_x:
            if self.enable_center_crossing:
                crossings = self.count_center_crossings(x_vals)
                if crossings < self.min_crossings:
                    return None

            if self.enable_oscillation_check:
                oscillations = self.count_oscillations(x_vals)
                if oscillations < self.min_oscillations:
                    return None

            return "No"

        return None

    def detect_yes_no_advanced_span_mediapipe(self, face_id):
        """
        Advanced method (python advanced_landmark_span):
          horizontal = mean(change_ratio(ear_span), change_ratio(eye_span))
          vertical   = mean(change_ratio(forehead_chin), change_ratio(nose_chin), change_ratio(nose_bridge_chin))
        """
        hs = self.history_span[face_id]
        need = self.history_length

        if (len(hs["ear_span"]) < need or len(hs["eye_span"]) < need or
            len(hs["nose_chin"]) < need or len(hs["forehead_chin"]) < need or
            len(hs["nose_bridge_chin"]) < need):
            return None

        ear_vals = [v for (v, _) in hs["ear_span"]]
        eye_vals = [v for (v, _) in hs["eye_span"]]

        nose_chin_vals = [v for (v, _) in hs["nose_chin"]]
        forehead_vals = [v for (v, _) in hs["forehead_chin"]]
        bridge_vals = [v for (v, _) in hs["nose_bridge_chin"]]

        def change_ratio(values):
            m = np.mean(values)
            if m == 0:
                return 0.0
            return (max(values) - min(values)) / m

        horizontal_change = float(np.mean([change_ratio(ear_vals), change_ratio(eye_vals)]))
        vertical_change = float(np.mean([change_ratio(forehead_vals), change_ratio(nose_chin_vals), change_ratio(bridge_vals)]))

        should_log, frame_count = self.should_log_metrics(face_id)
        if should_log:
            self.get_logger().info(
                f"[{face_id}] MEDIAPIPE advanced metrics (frames: {frame_count}): "
                f"h={horizontal_change:.3f}, v={vertical_change:.3f}, v/h={vertical_change/horizontal_change if horizontal_change > 0 else 0:.2f}"
            )

        # YES
        if vertical_change > self.yes_threshold_span_change:
            if vertical_change > self.ratio_threshold * horizontal_change:
                if self.enable_oscillation_check:
                    osc = self.count_oscillations(forehead_vals)
                    if osc < self.min_oscillations:
                        return None
                return "Yes"

        # NO
        if horizontal_change > self.no_threshold_span_change:
            if horizontal_change > self.ratio_threshold * vertical_change:
                if self.enable_center_crossing:
                    crossings = self.count_center_crossings(eye_vals)
                    if crossings < self.min_crossings:
                        return None
                if self.enable_oscillation_check:
                    osc = self.count_oscillations(eye_vals)
                    if osc < self.min_oscillations:
                        return None
                return "No"

        return None

    def detect_yes_no_span_yunet(self, face_id):
        """
        YuNet fallback with 5 keypoints:
          horizontal = mean(change_ratio(eye_span), change_ratio(mouth_span))
          vertical   = change_ratio(nose_mouth)
        """
        hs = self.history_span[face_id]
        need = self.history_length

        if (len(hs["eye_span"]) < need or len(hs["mouth_span"]) < need or len(hs["nose_mouth"]) < need):
            return None

        eye_vals = [v for (v, _) in hs["eye_span"]]
        mouth_vals = [v for (v, _) in hs["mouth_span"]]
        nose_mouth_vals = [v for (v, _) in hs["nose_mouth"]]

        def change_ratio(values):
            m = np.mean(values)
            if m == 0:
                return 0.0
            return (max(values) - min(values)) / m

        horizontal_change = float(np.mean([change_ratio(eye_vals), change_ratio(mouth_vals)]))
        vertical_change = float(change_ratio(nose_mouth_vals))

        should_log, frame_count = self.should_log_metrics(face_id)
        if should_log:
            self.get_logger().info(
                f"[{face_id}] YUNET metrics (frames: {frame_count}): "
                f"h={horizontal_change:.3f}, v={vertical_change:.3f}, v/h={vertical_change/horizontal_change if horizontal_change > 0 else 0:.2f}"
            )

        # YES
        if vertical_change > self.yes_threshold_span_change:
            if vertical_change > self.ratio_threshold * horizontal_change:
                if self.enable_oscillation_check:
                    osc = self.count_oscillations(nose_mouth_vals)
                    if osc < self.min_oscillations:
                        return None
                return "Yes"

        # NO
        if horizontal_change > self.no_threshold_span_change:
            if horizontal_change > self.ratio_threshold * vertical_change:
                if self.enable_center_crossing:
                    crossings = self.count_center_crossings(eye_vals)
                    if crossings < self.min_crossings:
                        return None
                if self.enable_oscillation_check:
                    osc = self.count_oscillations(eye_vals)
                    if osc < self.min_oscillations:
                        return None
                return "No"

        return None

    ######################################
    # 2. Publish gesture message
    ######################################
    def publish_gesture(self, face_id, gesture):
        """Publish gesture string to face-specific topic, creating publisher if needed."""
        if face_id not in self.face_pub:
            topic = f"/humans/faces/{face_id}/head_gesture"
            self.face_pub[face_id] = self.create_publisher(String, topic, 10)
            self.get_logger().info(f"Publishing {topic}")
        self.face_pub[face_id].publish(String(data=gesture))

    ######################################
    # 3) Debug visualization
    ######################################
    def image_callback(self, msg):
        """Buffer latest camera image for debug overlay."""
        if not self.debug_enabled:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = img
        except Exception as e:
            self.get_logger().error(f"[camera] Conversion failed: {e}")

    ######################################
    # 4) Publish debug image
    ######################################
    def publish_debug_image(self, overlays):
        """Draw gesture detection overlay on camera image and publish to debug topic."""
        img = self.latest_image.copy()

        for o in overlays:
            pts = o["points"]
            gesture = o["gesture"]
            gesture_alt = o.get("gesture_alt", None)
            method = o.get("method", "ROI")

            color = (0, 255, 0) if gesture == "Yes" else (0, 0, 255) if gesture == "No" else (255, 255, 0)

            for i in range(1, len(pts)):
                cv2.line(img, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                         (int(pts[i][0]), int(pts[i][1])), color, 2)
            if pts:
                lx, ly = int(pts[-1][0]), int(pts[-1][1])
                cv2.circle(img, (lx, ly), 4, color, -1)

                label_primary = f"{o['face_id']}: {gesture}" if gesture else f"{o['face_id']}: None"
                label_primary = f"[{method}] {label_primary}"

                if gesture_alt is not None:
                    label_alt = f"[ALT ROI] {gesture_alt}" if gesture_alt else "[ALT ROI] None"
                else:
                    label_alt = None

                cv2.putText(img, label_primary, (lx + 5, ly - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if label_alt:
                    color_alt = (200, 200, 200)
                    cv2.putText(img, label_alt, (lx + 5, ly - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_alt, 1)

        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = NodeHRIYesNoRecognition()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
