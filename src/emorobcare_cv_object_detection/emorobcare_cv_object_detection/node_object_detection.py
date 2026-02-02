import rclpy
from rclpy.lifecycle import Node, LifecycleState, TransitionCallbackReturn
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time, Duration
from emorobcare_cv_msgs.msg import ObjectDetection, ObjectDetections, ProcessingTime
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time
import yaml
import copy
import threading
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ultralytics import YOLO
from collections import deque
from knowledge_core.api import KB
from my_game_interface.srv import PlaceObject, EditObject
import tf2_ros
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger

ENABLE_PROCESSING_TIME = False

class NodeObjectDetector(Node):
    ##################
    # Init methods
    ##################
    def __init__(self):
        super().__init__('object_detector_node')
        
        config_path = os.path.join(
            get_package_share_directory('emorobcare_cv_object_detection'),
            'config',
            'config.yaml'
        )
        self.config = self.load_config(config_path)
	
        self.publish_debug_image = self.config.get("draw_image", False)
        self.use_human_radar = self.config.get("use_human_radar", False)
        self.use_knowledge_base = self.config.get("use_knowledge_base", False)
        self.processing_rate = self.config.get("processing_rate", 5)
        assert not (self.use_knowledge_base and self.use_human_radar), \
            "Cannot enable both Knowledge Base and Human Radar at the same time."
        self.bridge = CvBridge() # this converts ROS Images to OpenCV Images

        # Locks for timer-based processing (similar to hri_face_detect/hri_body_detect)
        self.image_lock = threading.Lock()
        self.proc_lock = threading.Lock()

        # Image buffer variables
        self.buffered_image = None
        self.buffered_msg = None
        self.new_image = False
        self.skipped_images = 0
        self.start_skipping_ts = self.get_clock().now()

        if self.publish_debug_image:
            self.debug_image_pub = self.create_publisher(Image, '/debug/object_detection', 10)

        # Trigger service configuration
        self.use_trigger_service = self.config.get("use_trigger_service", False)
        self.model_active = False
        self.yolo_model = None

        # Store model path for lazy loading
        model_file = self.config.get("yolo_model_path", None)
        self.yolo_model_path = os.path.join(
            get_package_share_directory('emorobcare_cv_object_detection'),
            'models',
            model_file
        )
        self.yolo_device = self.config.get("yolo_device", 'cpu')

        # Load model immediately if not using trigger service
        if not self.use_trigger_service:
            self.load_yolo_model()
        else:
            # Create trigger services
            self.start_srv = self.create_service(
                Trigger,
                'start_detection',
                self.start_detection_callback
            )
            self.stop_srv = self.create_service(
                Trigger,
                'stop_detection',
                self.stop_detection_callback
            )
            self.get_logger().info("Object detection trigger services created (start_detection, stop_detection)")

        self.last_time = time.time()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.camera_type = self.config.get("camera_type", "usb_cam")
        camera_topic = "/camera/image_raw" if self.camera_type == "usb_cam" else "/realsense/color/image_raw"
        self.subscription = self.create_subscription(
            Image, # type of message
            camera_topic,  # topic name
            self.camera_callback,
            qos_profile=sensor_qos) # max number of messages to store if subscriber is slower than publisher

        self.object_detection_pub = self.create_publisher(
            ObjectDetections,
            '/detected_objects',
            10)

        # Create processing timer
        self.proc_timer = self.create_timer(
            1.0 / self.processing_rate, self.process_image)

        self.get_logger().info(
            f"Object detector node initialized with processing_rate={self.processing_rate} Hz")

        # Knowledge base and human radar
        if self.config.get("use_mediapipe_object_detection", False):
            # TODO, make labels depending on the model
            raise NotImplementedError
        self.fact_list = set()
        if self.use_human_radar or self.use_knowledge_base:
                self.kb = KB()
                self.object_detection_labels = ["blueberry", "corn", "pear", "tomato", "zucchini"]
                self.label_to_rdf_type = {"blueberry": "dbr:blueberry", # "dbr" -> DBpedia resource
                                                "corn": "dbr:corn",
                                                "pear": "dbr:pear",
                                                "tomato": "dbr:tomato",
                                                "zucchini": "dbr:zucchini"}
                if self.use_human_radar:
                    self.human_radar_placer = self.create_client(PlaceObject, '/sim_scene/place_object')
                    while not self.human_radar_placer.wait_for_service(timeout_sec=1.0):
                            self.get_logger().info('Waiting for /sim_scene/place_object service...')
            
                    self.human_radar_place_request = PlaceObject.Request()
                    # for removing object
                    self.human_radar_remover = self.create_client(EditObject, '/sim_scene/remove_object')
                    while not self.human_radar_remover.wait_for_service(timeout_sec=1.0):
                            self.get_logger().info('Waiting for /sim_scene/remove_object service...')
                    self.human_radar_remove_request = EditObject.Request()
                else:
                    self.human_radar_placer, self.human_radar_place_request, self.human_radar_remover, self.human_radar_remove_request = [False] * 4
        
                # Variables 
                self.object_name = None
                self.object_to_remove_name = None
                self.camera_normalized_x = None

                # temporal dimension on object detection
                self.detection_history = {}
                for label in self.object_detection_labels:
                        self.detection_history[label] = deque(maxlen=self.config.get("N_kb_object_detection", 30))

        # depth logic
        self.depth_subscription = self.create_subscription(
            Image,
            '/depth_map',
            self.depth_callback,
            qos_profile=sensor_qos
        )

        self.last_depth_frame = None
        self.epsilon_depth_stability = self.config.get("DEPTH_STABILITY_EPSILON", 0.5)
        self.depth_per_object = {}
        self.object_depth = None
        
        if ENABLE_PROCESSING_TIME:
            self.processing_time_pub = self.create_publisher(
                ProcessingTime,
                '/processing_time/object_detection',
                10)
        
        # TF broadcasting
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_reference_frame = self.config.get("tf_reference_frame", "camera_color_optical_frame")
        self.use_depth = self.config.get("use_depth", False)
        # Store object transforms for persistent publishing
        self.object_transforms = {}  # {label: TransformStamped}
        # Create timer to continuously publish TF (50ms = 20Hz, same as radar)
        self.tf_timer = self.create_timer(0.05, self.publish_all_object_tfs)
        self.get_logger().info(f"TF broadcaster initialized with reference frame: {self.tf_reference_frame}")

    def process_time_wall(self, t_in_wall: float):
        t_out_wall = time.time()
        delta_s = t_out_wall - t_in_wall

        processing_time_msg = ProcessingTime()
        processing_time_msg.id = "object_detection"

        processing_time_msg.t_in = Time(sec=int(t_in_wall), nanosec=int((t_in_wall % 1)*1e9))
        processing_time_msg.t_out = Time(sec=int(t_out_wall), nanosec=int((t_out_wall % 1)*1e9))

        delta = Duration()
        delta.sec = int(delta_s)
        delta.nanosec = int((delta_s - int(delta_s)) * 1e9)

        processing_time_msg.delta = delta
        self.processing_time_pub.publish(processing_time_msg)

    def load_config(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def load_yolo_model(self):
        """Load the YOLO model and activate detection."""
        if self.yolo_model is None:
            self.get_logger().info(f"Loading YOLO model from {self.yolo_model_path}...")
            start_time = time.time()
            self.yolo_model = YOLO(self.yolo_model_path)
            self.yolo_model.to(self.yolo_device)
            load_time = time.time() - start_time
            self.get_logger().info(f"YOLO model loaded successfully on device: {self.yolo_device} (took {load_time:.2f}s)")
        self.model_active = True

    def unload_yolo_model(self):
        """Unload the YOLO model to free up resources."""
        if self.yolo_model is not None:
            self.get_logger().info("Unloading YOLO model...")
            start_time = time.time()
            # Move model to CPU before deletion to free GPU memory
            if 'cuda' in self.yolo_device:
                self.yolo_model.to('cpu')
            del self.yolo_model
            self.yolo_model = None
            # Force garbage collection to free memory
            import gc
            gc.collect()
            if 'cuda' in self.yolo_device:
                import torch
                torch.cuda.empty_cache()
            unload_time = time.time() - start_time
            self.get_logger().info(f"YOLO model unloaded successfully (took {unload_time:.2f}s)")
        self.model_active = False

    def load_object_detector(self, model_path:str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        else:
            print(f"Object detection model exists at {model_path}")

        BaseOptions = mp.tasks.BaseOptions
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        RunningMode = mp.tasks.vision.RunningMode

        base_options = BaseOptions(model_asset_path=model_path)
        options = ObjectDetectorOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            score_threshold=self.config['score_threshold']
        )
        detector = ObjectDetector.create_from_options(options)
        return detector

    ####################################################
    # Depth Callback
    ####################################################

    def depth_callback(self, msg):
        """
        Note: If MiDaS is publishing bgr8 colored depth maps, you should switch it to raw single-channel float or uint8, or adapt this to cv2.COLOR_BGR2GRAY
        """
        # Skip processing if model is not active
        if not self.model_active:
            return
        try:
            self.last_depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert depth image: {e}")

    def get_depth(self, x1, y1, x2, y2):
        depth_value = 0
        if self.last_depth_frame is not None:
            try:
                center_x = int((x1 + x2) / 2.0)
                center_y = int((y1 + y2) / 2.0)
                h, w = self.last_depth_frame.shape[:2]
                if 0 <= center_x < w and 0 <= center_y < h:
                    # Crop small 3x3 region to average depth
                    patch = self.last_depth_frame[max(0, center_y-1):center_y+2,
                                                max(0, center_x-1):center_x+2]
                    if patch.size > 0:
                        depth_value = float(np.nanmean(patch))
            except Exception as e:
                self.get_logger().warn(f"Failed to get depth: {e}")
        return depth_value

    ##################
    # Trigger service callbacks
    ##################
    def start_detection_callback(self, request, response):
        """Service callback to start object detection by loading the model."""
        if not self.model_active:
            try:
                self.load_yolo_model()
                response.success = True
                response.message = "Object detection activated successfully"
                self.get_logger().info("Object detection activated via service call")
            except Exception as e:
                response.success = False
                response.message = f"Failed to activate object detection: {str(e)}"
                self.get_logger().error(f"Failed to activate object detection: {e}")
        else:
            response.success = False
            response.message = "Model is already active"
            self.get_logger().warn("Attempted to start detection but model is already active")
        return response

    def stop_detection_callback(self, request, response):
        """Service callback to stop object detection by unloading the model."""
        if self.model_active:
            try:
                self.unload_yolo_model()
                response.success = True
                response.message = "Object detection deactivated successfully"
                self.get_logger().info("Object detection deactivated via service call")
            except Exception as e:
                response.success = False
                response.message = f"Failed to deactivate object detection: {str(e)}"
                self.get_logger().error(f"Failed to deactivate object detection: {e}")
        else:
            response.success = False
            response.message = "Model is not active"
            self.get_logger().warn("Attempted to stop detection but model is not active")
        return response

    ##################
    # TF publishing logic
    ##################
    def update_object_tf(self, label, frame_width, frame_height, detection, depth_value=None):
        """
        Update TF transform data for a detected object (stores for continuous publishing).

        Args:
            label: Object label (e.g., 'tomato')
            frame_width: Width of the camera frame
            frame_height: Height of the camera frame
            detection: ObjectDetection message with bounding box coordinates
            depth_value: Optional depth value (for use_depth==True in the future)
        """
        transform = TransformStamped()
        transform.header.frame_id = self.tf_reference_frame
        transform.child_frame_id = f"object_{label}"

        # Calculate normalized x and y positions (center of bounding box)
        camera_normalized_x = (detection.x1 + detection.x2) / (2.0 * frame_width)
        camera_normalized_y = (detection.y1 + detection.y2) / (2.0 * frame_height)

        if self.use_depth and depth_value is not None:
            # With depth: use actual 3D position
            # X-axis: depth (forward from camera)
            transform.transform.translation.x = 3.0 - 2.0 * depth_value
            # Y-axis: lateral position (left-right)
            transform.transform.translation.y = camera_normalized_x
            # Z-axis: vertical position (up-down)
            transform.transform.translation.z = camera_normalized_y
        else:
            # Without depth: place object at fixed distance, vary lateral and vertical position
            # X-axis: fixed distance in front of camera
            transform.transform.translation.x = 1.5  # Fixed distance in meters
            # Y-axis: lateral position mapped from normalized x (left-right)
            # Map normalized x (0 to 1) to meters (e.g., -0.5 to 0.5)
            transform.transform.translation.y = (camera_normalized_x - 0.5) * 2.0
            # Z-axis: vertical position mapped from normalized y (up-down)
            # Map normalized y (0 to 1) to meters (e.g., -0.5 to 0.5)
            # Note: invert y because image coordinates have 0 at top
            transform.transform.translation.z = (0.5 - camera_normalized_y) * 2.0

        # No rotation (identity quaternion)
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

        # Store transform for continuous publishing
        self.object_transforms[label] = transform
        self.get_logger().debug(
            f"[TF] Updated transform for {label} at "
            f"x={transform.transform.translation.x:.2f}, "
            f"y={transform.transform.translation.y:.2f}, "
            f"z={transform.transform.translation.z:.2f}"
        )

    def publish_all_object_tfs(self):
        """
        Periodically publish all stored object transforms.
        Called by timer every 50ms.
        """
        for label, transform in self.object_transforms.items():
            # Update timestamp for each publish
            transform.header.stamp = self.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(transform)

    ##################
    # Human radar or KB logic
    ##################
    def update_temporal_history(self, objects):
        seen_labels = [obj[4].lower() for obj in objects]
        for label in self.object_detection_labels:
            self.detection_history[label].append(1 if label in seen_labels else 0)
            
    def handle_object_presence(self, label, depth_value, frame, detection):
        """
        Manage detection history, KB and radar addition, and TF publishing.
        """
        self.detection_history[label].append(1)
        n = self.config.get("N_kb_object_detection", 30)
        if len(self.detection_history[label]) < n:
            return

        presence_ratio = sum(self.detection_history[label]) / n
        if presence_ratio < self.config.get("ADD_THRESHOLD", 1):
            return

        rdf_type = self.label_to_rdf_type.get(label)
        if not rdf_type:
            return

        fact2 = f"myself sees {label}"
        if fact2 not in self.fact_list:
            self.fact_list.add(fact2)
            if self.use_knowledge_base:
                self.kb.add([fact2])
            if self.use_human_radar:
                self.object_name = label
                self.object_depth = self.depth_per_object.get(label, 0.0)
                self.camera_normalized_x = (detection.x1 + detection.x2) / (2.0 * frame.shape[1])
                self.send_request_human_radar()

        # Update TF for knowledge base mode (always update position while detected)
        if self.use_knowledge_base and fact2 in self.fact_list:
            frame_height, frame_width = frame.shape[:2]
            self.update_object_tf(label, frame_width, frame_height, detection, depth_value)

    def send_request_human_radar(self):
        self.human_radar_place_request.object_name = self.object_name
        self.human_radar_place_request.frame_id = ''
        self.human_radar_place_request.position.x = 3.0 - 2.0 * self.object_depth
        self.human_radar_place_request.position.y = self.camera_normalized_x # debug # this is x in camera
        self.human_radar_place_request.position.z = 0.0
        self.get_logger().info(f"[Radar] Setting object X position to {self.human_radar_place_request.position.x:.2f} based on depth {self.object_depth:.2f}")
        
        future = self.human_radar_placer.call_async(self.human_radar_place_request)
    
    def remove_old_objects(self):
        """
        Remove from KB, radar, and TF if no longer visible.
        """
        for label, history in self.detection_history.items():
            presence_ratio = sum(history) / float(self.config.get("N_kb_object_detection", 1))
            if presence_ratio < self.config.get("REMOVE_THRESHOLD", 0.05):
                fact = f"myself sees {label}"
                if fact in self.fact_list:
                    self.object_to_remove_name = label
                    if self.use_human_radar:
                        self.remove_object_human_radar()
                    if self.use_knowledge_base:
                        self.kb.remove([fact])
                        # Remove TF transform
                        if label in self.object_transforms:
                            del self.object_transforms[label]
                            self.get_logger().info(f"[TF] Removed transform for {label}")
                    self.fact_list.remove(fact)
                    self.get_logger().info(f"[Removed] {label}")

    def remove_object_human_radar(self):
        self.human_radar_remove_request.object_name = self.object_to_remove_name
        future = self.human_radar_remover.call_async(self.human_radar_remove_request)
        if future.result() is not None:
            self.get_logger().info(f'Object "{self.object_to_remove_name}" removed successfully.')
        else:
            self.get_logger().error('EditObject service call failed.')
    ##################
    # Debug logic
    ##################

    def draw_bounding_box(self, frame, x1, y1, x2, y2, label, confidence):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Draw bounding box
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def publish_debug_frame(self, frame):
        if self.publish_debug_image:
            debug_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            debug_image_msg.header.stamp = self.get_clock().now().to_msg()
            self.debug_image_pub.publish(debug_image_msg)

    ##################
    # Object detection logic
    ##################
    def detect_objects_yolo(self, bgr_frame):
        results = self.yolo_model.predict(source=bgr_frame, save=False, conf=self.config['score_threshold'], verbose=False)
        objects = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                label = self.yolo_model.names[int(box.cls[0])]
                objects.append((x1, y1, x2, y2, label, conf))
        return objects

    def process_detection(self, obj, frame):
        """
        Process each detection: depth, radar, KB, drawing.
        """
        x1, y1, x2, y2, label, confidence = obj
        detection = ObjectDetection()
        detection.x1, detection.y1, detection.x2, detection.y2 = map(float, [x1, y1, x2, y2])
        detection.label = label
        detection.confidence = float(confidence)

        # depth
        depth_value = self.get_depth(x1, y1, x2, y2)
        if depth_value is not None:
            if self.last_depth_frame is not None and np.nanmax(self.last_depth_frame) > 0:
                depth_norm = depth_value / np.nanmax(self.last_depth_frame)
            else:
                # just in case safeguard
                depth_norm = depth_value
            self.depth_per_object[label.lower()] = depth_norm

        # KB or Radar
        if self.use_human_radar or self.use_knowledge_base:
            self.handle_object_presence(label.lower(), depth_value, frame, detection)

        # draw debug
        if self.publish_debug_image:
            self.draw_bounding_box(frame, x1, y1, x2, y2, label, confidence)

        return detection


    ##################
    # Image callback (buffers image for timer-based processing)
    ##################
    def camera_callback(self, msg):
        """Buffer incoming images for timer-based processing."""
        with self.image_lock:
            # Convert and buffer image
            self.buffered_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.buffered_msg = msg

            # Track skipped images
            if self.new_image:
                self.skipped_images += 1
                if self.skipped_images > 100:
                    now = self.get_clock().now()
                    skip_time = (now - self.start_skipping_ts).nanoseconds / 1e9
                    self.get_logger().warning(
                        "Object detection processing too slow. "
                        f"Skipped 100 new incoming images over the last {skip_time:.1f}sec")
                    self.start_skipping_ts = now
                    self.skipped_images = 0

            self.new_image = True

    ##################
    # Process image (timer callback)
    ##################
    def process_image(self):
        """Timer callback to process buffered images at the configured rate."""
        # Skip if no new image or already processing
        if (not self.new_image) or (not self.proc_lock.acquire(blocking=False)):
            return

        try:
            # Copy image data within the lock, then process outside
            with self.image_lock:
                if ENABLE_PROCESSING_TIME:
                    t_in = time.time()
                frame = copy.deepcopy(self.buffered_image)
                msg = self.buffered_msg
                self.new_image = False

            if frame is None:
                return

            # Skip processing if model is not active
            if not self.model_active:
                return

            # 1) detect with yolo
            objects = self.detect_objects_yolo(frame)

            # 2) logic
            if objects:
                detections_msg = ObjectDetections()
                for obj in objects:
                    # 2.1) process detections, add into KB or human radar, draw debug image
                    detection = self.process_detection(obj, frame)
                    if detection:
                        detections_msg.detections.append(detection)
                if self.use_human_radar or self.use_knowledge_base:
                    # 2.2) temporal logic, add 1 if seen
                    self.update_temporal_history(objects)
                self.object_detection_pub.publish(detections_msg)
            else:
                if self.use_human_radar or self.use_knowledge_base:
                    # 3) temporal logic again, add 0 if object not detected
                    self.update_temporal_history([])

            # 4) remove from knowledge base and human radar
            if self.use_human_radar or self.use_knowledge_base:
                self.remove_old_objects()

            # 5) publish debug image
            if self.publish_debug_image:
                self.publish_debug_frame(frame)

            if ENABLE_PROCESSING_TIME:
                self.process_time_wall(t_in)
        finally:
            self.proc_lock.release()

def main(args=None):
    rclpy.init(args=args)
    node = NodeObjectDetector()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
