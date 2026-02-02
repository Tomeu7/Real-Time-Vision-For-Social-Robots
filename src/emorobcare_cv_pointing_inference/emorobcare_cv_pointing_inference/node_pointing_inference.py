import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time, Duration
from emorobcare_cv_msgs.msg import GestureRecognition, GestureRecognitions, ProcessingTime
from geometry_msgs.msg import Point
from emorobcare_cv_msgs.msg import ObjectDetections, PointingDetection
from cv_bridge import CvBridge
import cv2
import os
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import mediapipe as mp
from sensor_msgs.msg import CameraInfo
from collections import deque

import time

ENABLE_PROCESSING_TIME = False
THRESHOLD_EMOROBCARE_IMAGE = 2

class NodePointingInference(Node):
    """
    ROS2 node that detects what a person is pointing at by combining gesture recognition
    (index finger direction) with object detections. Optionally integrates with a knowledge
    base / human radar system and supports both 2D and 3D (depth-based) pointing inference.
    """
    ##################
    # Init methods
    ##################
    def __init__(self):
        """
        Initializes the pointing inference node: loads config, creates subscribers for
        camera images, object detections, and gesture recognitions, sets up publishers,
        and optionally connects to the knowledge base / human radar services.
        """
        super().__init__('pointing_inference_node')

        self.bridge = CvBridge()

        config_path = os.path.join(
            get_package_share_directory('emorobcare_cv_pointing_inference'),
            'config', 'config.yaml'
        )
        self.config = self.load_config(config_path)

        # subscribe to camera, detected objects and gestures
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.camera_type = self.config.get("camera_type", "usb_cam")
        self.use_human_radar = self.config.get("use_human_radar", True)
        self.intersect_with_objects = self.config.get("intersect_with_objects", True)
        self.use_depth = self.config.get("use_depth", False)
        self.draw_image = self.config.get("draw_image", False)
        if self.draw_image:
            self.image_pub = self.create_publisher(Image, "/debug/pointing", 10)
        if self.use_depth:
            assert self.camera_type == "realsense", "depth usage requires realsense camera"
            self.depth_subscription = self.create_subscription(
                Image,
                "/depth_registered/image_rect",
                self.depth_callback,
                qos_profile=sensor_qos
            )
            self.latest_depth = None

        camera_topic = "/camera/image_raw" if self.camera_type == "usb_cam" else "/realsense/color/image_raw"
        self.subscription = self.create_subscription(
            Image, # type of message
            camera_topic,  # topic name
            self.image_callback,
            qos_profile=sensor_qos) # max number of messages to store if subscriber is slower than publisher
        if self.intersect_with_objects:
            self.create_subscription(ObjectDetections, "/detected_objects", self.objects_callback, 10)
        self.latest_objects = []
        self.last_object_time = 0

        self.latest_gestures = None
        self.latest_gestures_time = 0  # BUG FIX: initialize so image_callback won't crash if called before any gesture arrives
        self.gesture_sub = self.create_subscription(
            GestureRecognitions,
            '/gesture_recognition',
            self.gesture_callback,
            10
        )

        # publish to pointed objects
        self.pointing_pub = self.create_publisher(PointingDetection, "/pointing_target", 10)

        # Knowledge base and human radar
        self.kb, self.label_to_rdf_type, self.hr_highlighter, self.hr_highlighter_remover, self.hr_highlighter_request, self.highlight_history = [None] * 6
        if self.use_human_radar:
            # Knowledge base
            from knowledge_core.api import KB
            from my_game_interface.srv import EditObject

            self.kb = KB()
            self.label_to_rdf_type = {"blueberry": "dbr:blueberry", # "dbr" -> DBpedia resource
                                    "corn": "dbr:corn",
                                    "pear": "dbr:pear",
                                    "tomato": "dbr:tomato",
                                    "zucchini": "dbr:zucchini"}

            self.hr_highlighter = self.create_client(EditObject, '/sim_scene/highlight_object')
            while not self.hr_highlighter.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Waiting for /sim_scene/highlight_object service...')
            self.hr_highlighter_remover = self.create_client(EditObject, '/sim_scene/remove_highlight_object')
            while not self.hr_highlighter_remover.wait_for_service(timeout_sec=1.0):  # BUG FIX: was waiting on hr_highlighter instead of hr_highlighter_remover
                self.get_logger().info('Waiting for /sim_scene/remove_highlight_object service...')
            self.hr_highlighter_request = EditObject.Request()

            self.highlighted_object = None
            self.highlight_history = deque(maxlen=self.config.get("N_kb_pointing", 1))
        self.debug = False

        # Camera info, important for realsense
        self.camera_info_received = False
        self.fx = self.fy = self.cx = self.cy = None

        camera_info_topic = "camera/camera_info" if self.camera_type == "usb_cam" else "realsense/color/camera_info"  # BUG FIX: was "use_cam" instead of "usb_cam"
        self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10)
        if ENABLE_PROCESSING_TIME:
            self.processing_time_pub = self.create_publisher(
                    ProcessingTime,
                    '/processing_time/pointing',
                    10)

    def process_time_wall(self, t_in_wall: float):
        """
        Computes wall-clock processing time from t_in_wall to now and publishes
        a ProcessingTime message with input timestamp, output timestamp, and delta.
        """
        t_out_wall = time.time()
        delta_s = t_out_wall - t_in_wall

        processing_time_msg = ProcessingTime()
        processing_time_msg.id = "pointing_inference"

        processing_time_msg.t_in = Time(sec=int(t_in_wall), nanosec=int((t_in_wall % 1)*1e9))
        processing_time_msg.t_out = Time(sec=int(t_out_wall), nanosec=int((t_out_wall % 1)*1e9))

        delta = Duration()
        delta.sec = int(delta_s)
        delta.nanosec = int((delta_s - int(delta_s)) * 1e9)

        processing_time_msg.delta = delta
        self.processing_time_pub.publish(processing_time_msg)

    def load_config(self, path):
        """
        Loads and returns the YAML configuration file at the given path.
        """
        with open(path, 'r') as f:
            return yaml.safe_load(f)


    def camera_info_callback(self, msg):
        """
        Receives camera intrinsic parameters (fx, fy, cx, cy) from CameraInfo messages.
        Only processes the first message, then ignores subsequent ones.
        """
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_info_received = True
            self.get_logger().info(f"Camera info: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    ##################
    # Human radar logic
    ##################
    def send_request_human_radar_highlight(self, type_highlight = "add"):
        """
        Sends an async service request to either highlight or remove highlight from an
        object in the human radar simulation scene. Uses self.highlighted_object as the
        target object name.
        """
        self.hr_highlighter_request.object_name = self.highlighted_object
        if type_highlight == "add":
            future = self.hr_highlighter.call_async(self.hr_highlighter_request)
        elif type_highlight == "remove":
            future = self.hr_highlighter_remover.call_async(self.hr_highlighter_request)
        else:
            raise NotImplementedError

    ##################
    # Pointing detection pipeline
    ##################
    def objects_callback(self, msg):
        """
        Callback for /detected_objects topic. Stores the latest object detections
        as a list of (x1, y1, x2, y2, label) tuples and records the reception time.
        """
        self.latest_objects = [(d.x1, d.y1, d.x2, d.y2, d.label) for d in msg.detections]
        self.last_object_time = time.time()

    def gesture_callback(self, msg):
        """
        Callback for /gesture_recognition topic. Extracts index finger tip and base
        (MCP) landmarks from each recognized gesture, along with bounding box, handedness,
        and gesture label. Stores results in self.latest_gestures.
        """
        gesture_labels = []
        for rec in msg.recognitions:
            bbox = rec.bbox
            handedness = rec.handedness
            gesture = rec.gesture
            landmarks = rec.landmarks  # geometry_msgs/Point[]
            if len(landmarks) > max(mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP): # mediapipe landmark should have 9 points
                index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                index_base = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
                gesture_labels.append((bbox, handedness, gesture, index_tip, index_base))
            else:
                self.get_logger().error('Landmark error in gesture callback')

        self.latest_gestures = gesture_labels
        self.latest_gestures_time = time.time()

    ##################
    # 0. Process frame
    ##################
    def process_frame(self, msg):
        """
        Converts a ROS Image message to a BGR OpenCV frame.
        """
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        return frame  # BUG FIX: removed unused rgb conversion that was discarded

    def depth_callback(self, msg):
        """
        Callback for the depth image topic. Converts the ROS depth Image message
        to an OpenCV array stored in self.latest_depth. Sets it to None on failure.
        """
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")
            self.latest_depth = None

    ##################
    #  1. Prepare message and compute pointing direction
    ##################

    def compute_direction_prepare_message(self, frame, gesture_labels):
        """
        Builds a PointingDetection message from the current frame and gesture data.
        If a hand is detected and recent, computes pointing direction (2D or 3D depending
        on config). Returns (pointing_msg, base_point, direction_vector). base_point and
        direction_vector are None when no hand is detected.
        """
        pointing_msg = PointingDetection()
        pointing_msg.header.stamp = self.get_clock().now().to_msg()
        if gesture_labels and (time.time() - self.latest_gestures_time) < THRESHOLD_EMOROBCARE_IMAGE:
            hand_bbox, handedness, gesture, index_tip, index_base = gesture_labels[0] # NOTE THIS IS AN ASSUMPTION
            img_h, img_w = frame.shape[:2]

            # More from message
            pointing_msg.is_pointing = (gesture == "Pointing")
            pointing_msg.hand = handedness # "Left" or "Right"

            pointing_msg.hand_x1 = float(hand_bbox[0])
            pointing_msg.hand_y1 = float(hand_bbox[1])
            pointing_msg.hand_x2 = float(hand_bbox[2])
            pointing_msg.hand_y2 = float(hand_bbox[3])

            if self.use_depth and self.latest_depth is not None:
                base_point, direction_vector = self.infer_3d_direction(img_h, img_w, index_tip, index_base)

                if base_point is not None and direction_vector is not None:
                    # Fill message
                    pointing_msg.base_x, pointing_msg.base_y, pointing_msg.base_z = map(float, base_point)
                    pointing_msg.direction_x, pointing_msg.direction_y, pointing_msg.direction_z = map(float, direction_vector)
            else:
                base_point, direction_vector = self.infer_direction(img_h, img_w, index_tip, index_base)

                pointing_msg.base_x = float(base_point[0])
                pointing_msg.base_y = float(base_point[1])
                pointing_msg.direction_x = float(direction_vector[0])
                pointing_msg.direction_y = float(direction_vector[1])

                pointing_msg.base_z = 0.0           # Assume hand is at camera plane
                pointing_msg.direction_z = 1.0      # Pointing forward into scene
        else:
            base_point, direction_vector = None, None
            # More from message
            pointing_msg.is_pointing = False
            pointing_msg.hand = ""
            pointing_msg.hand_x1 = pointing_msg.hand_y1 = pointing_msg.hand_x2 = pointing_msg.hand_y2 = 0.0

        return pointing_msg, base_point, direction_vector

    def infer_3d_direction(self, img_h, img_w, index_tip, index_base):
        """
        Computes a 3D pointing direction using depth data and camera intrinsics.
        Converts index finger tip and base pixel coordinates to 3D camera-space points
        via the pinhole projection model, then returns a normalized direction vector.
        Returns (base_point_3d, direction_vector_3d) or (None, None) on failure.
        """
        tip_px = int(index_tip.x * img_w), int(index_tip.y * img_h)
        base_px = int(index_base.x * img_w), int(index_base.y * img_h)

        # Check if pixels are within valid image bounds
        if all(0 <= v < img_w for v in [tip_px[0], base_px[0]]) and all(0 <= v < img_h for v in [tip_px[1], base_px[1]]):
            # Get depth in meters from depth map (convert from mm)
            depth_tip = self.latest_depth[tip_px[1], tip_px[0]] * 0.001
            depth_base = self.latest_depth[base_px[1], base_px[0]] * 0.001
            if depth_tip <= 0 or depth_base <= 0:
                self.get_logger().debug(f"Invalid depth — base: {depth_base:.3f}, tip: {depth_tip:.3f}")
            if depth_tip > 0 and depth_base > 0:
                # Project 2D pixel + depth to 3D point in camera space
                # Using pinhole projection model:
                # X = (u - cx) * Z / fx
                # Y = (v - cy) * Z / fy
                # Z = depth
                Xb = (base_px[0] - self.cx) * depth_base / self.fx
                Yb = (base_px[1] - self.cy) * depth_base / self.fy
                Zb = depth_base

                Xt = (tip_px[0] - self.cx) * depth_tip / self.fx
                Yt = (tip_px[1] - self.cy) * depth_tip / self.fy
                Zt = depth_tip

                # Compute normalized direction vector from base to tip
                dx, dy, dz = Xt - Xb, Yt - Yb, Zt - Zb
                norm = np.linalg.norm([dx, dy, dz])
                if norm > 0:
                    return (Xb, Yb, Zb), (dx / norm, dy / norm, dz / norm)
        return None, None

    def infer_direction(self, img_h, img_w, index_tip, index_base):
        """
        Computes a 2D pointing direction from index finger tip and base landmarks.
        Converts normalized landmark coordinates to pixel coordinates, then returns
        the base point and a unit direction vector in image space.
        """
        tip_x, tip_y = int(index_tip.x * img_w), int(index_tip.y * img_h)
        base_x, base_y = int(index_base.x * img_w), int(index_base.y * img_h)
        direction_vector = np.array([tip_x - base_x, tip_y - base_y])
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            direction_vector = direction_vector / norm
        return (base_x, base_y), direction_vector

    ##################
    # 2. Remove highlight in human radar
    ##################
    def remove_highlight_info_if_object_disappears(self):
        """
        Queries the knowledge base for currently visible objects. If the currently
        highlighted object is no longer visible, clears the highlight state and history.
        NOTE: The comparison on line with class_name.lower() vs self.label_to_rdf_type
        likely has a bug — label_to_rdf_type values are URIs like "dbr:blueberry" which
        won't match plain class names.
        """
        kb_objects = self.kb["myself sees ?obj"]
        if self.highlighted_object is not None:
            found_highlighted_object = False
            for item in kb_objects:
                obj = item["obj"]
                details = self.kb.details(obj)
                label = details["label"]["default"]
                classes = details["attributes"][0]["values"] if details["attributes"] else []
                class_name = classes[0]["label"]["default"] if classes else label
                if class_name.lower() == self.label_to_rdf_type[self.highlighted_object]:
                    found_highlighted_object = True

            if not found_highlighted_object:
                self.highlighted_object = None
                self.highlight_history.clear()

    def remove_highlight_if_not_pointing(self):
        """
        Checks if the user has stopped pointing at the currently highlighted object
        by computing the ratio of recent frames where pointing was detected (from
        highlight_history deque). If below the REMOVE_THRESHOLD_POINTING config
        threshold, removes the highlight via the human radar service.
        """
        if self.highlighted_object is not None:
            presence_ratio = sum(self.highlight_history) / float(self.highlight_history.maxlen)
            if presence_ratio < self.config.get("REMOVE_THRESHOLD_POINTING", 0.05):
                self.send_request_human_radar_highlight("remove")
                self.highlighted_object = None

    ##################
    # 3. Point logic
    ##################

    def point_logic(self, base_point, direction_vector, pointing_msg):
        """
        Core pointing logic: extends a ray from the finger base along the pointing
        direction and checks intersection with each detected object's bounding box.
        Fills pointing_msg with the first intersected target. If human radar is enabled,
        maintains a temporal history and triggers highlight add/remove based on
        configurable thresholds (ADD_THRESHOLD_POINTING).
        """
        if base_point is not None and direction_vector is not None and self.latest_objects and (time.time() - self.last_object_time) < THRESHOLD_EMOROBCARE_IMAGE:
            p1 = base_point
            p2 = self.extend_pointing_line(p1, direction_vector)
            found_target = False
            for x1, y1, x2, y2, label in self.latest_objects:
                box = (x1, y1, x2, y2)
                if self.line_intersects_rect(p1, p2, box):
                    pointing_msg.target_label = label
                    pointing_msg.target_x1 = float(x1)
                    pointing_msg.target_y1 = float(y1)
                    pointing_msg.target_x2 = float(x2)
                    pointing_msg.target_y2 = float(y2)
                    self.get_logger().info(f"Pointing at: {label}")
                    if self.use_human_radar:
                        if label != self.highlighted_object:
                            presence_ratio = sum(self.highlight_history) / float(self.highlight_history.maxlen)
                            if presence_ratio >=  self.config.get("ADD_THRESHOLD_POINTING", 1):
                                self.highlighted_object = label.lower()
                                self.send_request_human_radar_highlight("add")

                        # 3.4) temporal logic again, add 1 if label detected
                        self.highlight_history.append(1)
                    found_target = True
                    break
            if not found_target:
                pointing_msg.target_label = ""
                pointing_msg.target_x1 = pointing_msg.target_y1 = pointing_msg.target_x2 = pointing_msg.target_y2 = 0.0
                if self.use_human_radar:
                    self.highlight_history.append(0)
        else:
            if self.use_human_radar:
                self.highlight_history.append(0)
        return pointing_msg

    def extend_pointing_line(self, base_point, direction_vector, length=1000):
        """
        Extends a pointing ray from base_point along direction_vector by the given
        length (in pixels for 2D). For 3D mode, only uses the x and y components
        of the direction to project onto the 2D image plane.
        """
        if self.use_depth:
            dx, dy, _ = direction_vector
        else:
            dx, dy = direction_vector
        tip_x = int(base_point[0] + dx * length)
        tip_y = int(base_point[1] + dy * length)
        return (tip_x, tip_y)

    def line_intersects_rect(self, p1, p2, box):
        """
        Tests whether a line segment from p1 to p2 intersects an axis-aligned
        rectangle defined by box=(x1, y1, x2, y2). Checks intersection against
        all four edges of the rectangle.
        """
        x1, y1, x2, y2 = box
        rect_edges = [
            ((x1, y1), (x2, y1)),  # top
            ((x2, y1), (x2, y2)),  # right
            ((x2, y2), (x1, y2)),  # bottom
            ((x1, y2), (x1, y1))   # left
        ]
        for edge_start, edge_end in rect_edges:
            if self.do_lines_intersect(p1, p2, edge_start, edge_end):
                return True
        return False

    def do_lines_intersect(self, a1, a2, b1, b2):
        """
        Determines if two line segments (a1-a2) and (b1-b2) intersect using
        the counter-clockwise (CCW) orientation test.
        """
        def ccw(p1, p2, p3):
            return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
        return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)

    ##################
    # 5. Image callback
    ##################

    def image_callback(self, msg):
        """
        Main pipeline callback triggered on each camera frame. Sequentially:
        0) Decodes the image
        1) Computes pointing direction from gesture landmarks
        2) If human radar enabled, checks whether to remove stale highlights
        3) If object intersection enabled, tests pointing ray against detected objects
        4) Publishes the PointingDetection message if a hand was recently detected.
        Optionally publishes a debug image with the pointing arrow overlay.
        """
        # 0) Decode image
        if ENABLE_PROCESSING_TIME:
            t_in = time.time()
        frame = self.process_frame(msg)

        # 1) Compute direction only from the first detected hand, input is the landmarks. Also prepare message.
        pointing_msg, base_point, direction_vector = self.compute_direction_prepare_message(frame, self.latest_gestures)

        if self.use_human_radar:
            # 2.1) remove highlight INFO - object disappears
            self.remove_highlight_info_if_object_disappears()

            # 2.2) remove highlight - not pointing
            self.remove_highlight_if_not_pointing()

        # 3) Intersect with objects, we intersect with first target
        if self.intersect_with_objects:
            pointing_msg = self.point_logic(base_point, direction_vector, pointing_msg)

        # 2D Debug Draw
        if self.draw_image:
            debug_img = frame.copy()
            if base_point is not None and direction_vector is not None:
                p1 = (int(base_point[0]), int(base_point[1]))
                p2 = self.extend_pointing_line(p1, direction_vector, length=200)
                cv2.arrowedLine(debug_img, p1, p2, (0, 255, 0), 3)
            debug_img_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            debug_img_msg.header = msg.header
            self.image_pub.publish(debug_img_msg)

        # 4) Publish
        if self.latest_gestures and (time.time() - self.latest_gestures_time) < THRESHOLD_EMOROBCARE_IMAGE:  # at least one hand detected
            if ENABLE_PROCESSING_TIME:
                self.process_time_wall(t_in)
            self.pointing_pub.publish(pointing_msg)
        else:
            self.get_logger().debug("No hands detected — not publishing pointing message.")

def main(args=None):
    """
    Entry point: initializes ROS2, spins the NodePointingInference node,
    and handles graceful shutdown on KeyboardInterrupt.
    """
    rclpy.init(args=args)
    node = NodePointingInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
