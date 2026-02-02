import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time, Duration
from emorobcare_cv_msgs.msg import GestureRecognitions, GestureRecognition, ProcessingTime
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import os
import yaml
import numpy as np
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import tensorflow as tf
import copy
import itertools
from collections import deque
import time
import threading

ENABLE_PROCESSING_TIME = False

# code keypointclassifier based on:
# https://github.com/kinivi/hand-gesture-recognition-mediapipe
class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='models/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index


class NodeGestureDetection(Node):
    ##################
    # Init methods
    ##################
    def __init__(self):
        super().__init__('gesture_detection_node')

        self.bridge = CvBridge()

        config_path = os.path.join(
            get_package_share_directory('emorobcare_cv_gesture_detection'),
            'config', 'config.yaml'
        )
        self.config = self.load_config(config_path)

        self.camera_type = self.config.get("camera_type", "usb_cam")

        # Model: MediaPipe hand model
        self.hands_landmarker = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.get("max_num_hands", 2),
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7
        )
        self.flip_hands = True
        # Model: gesture recognizer
        model_path = os.path.join(
            get_package_share_directory('emorobcare_cv_gesture_detection'),
            'models',
            'keypoint_classifier.tflite'
        )
        self.gesture_recogniser = KeyPointClassifier(model_path=model_path)
        self.gesture_recogniser_labels = ["Open", "Close", "Pointing", "Ok"]

        # Locks for timer-based processing (similar to hri_face_detect/hri_body_detect)
        self.image_lock = threading.Lock()
        self.proc_lock = threading.Lock()

        # Image buffer variables
        self.buffered_image = None
        self.buffered_msg = None
        self.new_image = False
        self.skipped_images = 0
        self.start_skipping_ts = self.get_clock().now()

        # subscribe to camera and to detected objects
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        camera_topic = "/camera/image_raw" if self.camera_type == "usb_cam" else "/realsense/color/image_raw"
        self.subscription = self.create_subscription(
            Image, # type of message
            camera_topic,  # topic name
            self.image_callback,
            qos_profile=sensor_qos) # max number of messages to store if subscriber is slower than publisher

        self.gesture_pub = self.create_publisher(
                                            GestureRecognitions,
                                            '/gesture_recognition',
                                            10
                                            )
        if ENABLE_PROCESSING_TIME:
            self.processing_time_pub = self.create_publisher(
                ProcessingTime,
                '/processing_time/gesture',
                10)

    def process_time_wall(self, t_in_wall: float):
        t_out_wall = time.time()
        delta_s = t_out_wall - t_in_wall

        processing_time_msg = ProcessingTime()
        processing_time_msg.id = "gesture_detection"

        processing_time_msg.t_in = Time(sec=int(t_in_wall), nanosec=int((t_in_wall % 1)*1e9))
        processing_time_msg.t_out = Time(sec=int(t_out_wall), nanosec=int((t_out_wall % 1)*1e9))

        delta = Duration()
        delta.sec = int(delta_s)
        delta.nanosec = int((delta_s - int(delta_s)) * 1e9)

        processing_time_msg.delta = delta
        self.processing_time_pub.publish(processing_time_msg)
            
    def load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    ##################
    # 0. Process frame
    ##################
    def process_frame(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb

    ##################
    # 1. Hand landmarks
    ##################
    def get_hand_landmarks(self, rgb):
        hands_results = self.hands_landmarker.process(rgb)
        # OUTPUT gesture_labels
        # If we use hand landmark
        # a) multi_hand_landmarks: list of landmarks (each 21 points per hand)
        # b) multi_handedness: left/right confidence
        hands_landmarks = hands_results.multi_hand_landmarks if hands_results and hands_results.multi_hand_landmarks else []
        
        if hands_landmarks:
            if self.flip_hands:
                hands_handedness = [self.flip_handedness(h.classification[0].label) for h in hands_results.multi_handedness]
            else:
                hands_handedness = [h.classification[0].label for h in hands_results.multi_handedness]
        else:
            hands_handedness = []

        return hands_landmarks, hands_handedness
    
    def flip_handedness(self,  label):
        return "Left" if label == "Right" else "Right"

    ##################
    # 2. Gest recognition
    ##################
    def recognise_gest(self, rgb, hands_landmarks, hands_handedness):
        """
        Gest recognition based on mediapipe landmarks
        """
        gesture_labels = []
        if len(hands_landmarks) > 0: # only recognise if hand is recognised
            for hand_landmark, handedness in zip(hands_landmarks, hands_handedness):
                if self.config.get("only_detect_right_hand") and handedness.lower() != "right":
                    continue
                brect = self.calc_bounding_box_for_gest(rgb, hand_landmark)
                keypoint_label = self.recognize_gesture(hand_landmark)
                gesture_labels.append((brect, handedness, keypoint_label, hand_landmark))
        return gesture_labels

    def recognize_gesture(self, landmark_list):
        # Preprocessing
        landmark_point = []
        for lm in landmark_list.landmark:
            landmark_point.append([lm.x, lm.y])
        pre_processed_landmark = self.pre_process_landmark(landmark_point)

        # Classify keypoint gesture
        hand_sign_id = self.gesture_recogniser(pre_processed_landmark)

        return self.gesture_recogniser_labels[hand_sign_id]

    def pre_process_landmark(self, landmark_list):
        temp_landmark = copy.deepcopy(landmark_list)
        base_x, base_y = temp_landmark[0]
        for i in range(len(temp_landmark)):
            temp_landmark[i][0] -= base_x
            temp_landmark[i][1] -= base_y
        temp_landmark = list(itertools.chain.from_iterable(temp_landmark))
        max_value = max(list(map(abs, temp_landmark)))
        return [v / max_value for v in temp_landmark]

    def calc_bounding_box_for_gest(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]
    
    ##################
    # 3. Message logic
    ##################
    def create_recognition_msg(self, gesture_labels):
        recognitions_msg = GestureRecognitions()
        recognitions_msg.header.stamp = self.get_clock().now().to_msg()

        for bbox, handedness, gesture, landmark in gesture_labels:
            rec = GestureRecognition()
            rec.header = recognitions_msg.header
            rec.handedness = handedness
            rec.gesture = gesture
            rec.bbox = [float(x) for x in bbox]
            rec.landmarks = [Point(x=lm.x, y=lm.y, z=lm.z) for lm in landmark.landmark]

            recognitions_msg.recognitions.append(rec)
        return recognitions_msg
    
    ##################
    # 4. Image callback (buffers image for timer-based processing)
    ##################
    def image_callback(self, msg):
        with self.image_lock:
            # Track skipped images
            if self.new_image:
                self.skipped_images += 1
                if self.skipped_images > 100:
                    now = self.get_clock().now()
                    skip_time = (now - self.start_skipping_ts).nanoseconds / 1e9
                    self.get_logger().warning(
                        "Gesture detection processing too slow. "
                        f"Skipped 100 new incoming images over the last {skip_time:.1f}sec")
                    self.start_skipping_ts = now
                    self.skipped_images = 0
            
            self.buffered_msg = msg
            self.new_image = True

    ##################
    # 5. Process image (timer callback)
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
                msg = self.buffered_msg
                self.new_image = False

            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if image is None:
                return

            # 0) Convert to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 1) Detect hands
            hands_landmarks, hands_handedness = self.get_hand_landmarks(rgb)

            # 2) Recognize gestures
            gesture_labels = self.recognise_gest(rgb, hands_landmarks, hands_handedness)

            # OUTPUT: a list with (bounding box hand, "left"/"right", "gesture", hand_landmark)
            if gesture_labels:
                # 3) Prepare message
                recognitions_msg = self.create_recognition_msg(gesture_labels)

                # 4) Publish
                self.gesture_pub.publish(recognitions_msg)
                if ENABLE_PROCESSING_TIME:
                    self.process_time_wall(t_in)
        finally:
            self.proc_lock.release()
        
def main(args=None):
    rclpy.init(args=args)
    node = NodeGestureDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
