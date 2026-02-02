#!/usr/bin/env python3
"""
HRI Performance Monitor - Measures topic rates and end-to-end latency.

Usage:
    python3 hri_performance_monitor.py
    python3 hri_performance_monitor.py --csv output.csv
    python3 hri_performance_monitor.py --duration 60
    python3 hri_performance_monitor.py --wait-ready    # Wait for all topics before measuring

Features:
    1. Startup health check - waits for required topics to be active
    2. Measures publish rate (Hz) for all key HRI topics
    3. Measures end-to-end latency from camera to downstream nodes
    4. Prints live stats and optionally logs to CSV
"""

import argparse
import csv
import time
from collections import defaultdict
from datetime import datetime
from threading import Lock

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Message types
from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
from hri_msgs.msg import IdsList

# Try importing custom messages (may not be available)
try:
    from emorobcare_cv_msgs.msg import ObjectDetections, PointingDetection, GestureRecognitions, ProcessingTime
    EMOROBCARE_MSGS_AVAILABLE = True
except ImportError:
    EMOROBCARE_MSGS_AVAILABLE = False
    print("[WARN] emorobcare_cv_msgs not found - some topics won't be monitored")


class TopicStats:
    """Track statistics for a single topic."""

    def __init__(self, name: str):
        self.name = name
        self.message_count = 0
        self.timestamps = []  # Recent message receive times
        self.header_stamps = []  # Recent header timestamps (for latency)
        self.window_size = 50  # Rolling window for Hz calculation
        self.lock = Lock()

    def record_message(self, receive_time: float, header_stamp: float = None):
        with self.lock:
            self.message_count += 1
            self.timestamps.append(receive_time)
            if header_stamp is not None:
                self.header_stamps.append((receive_time, header_stamp))

            # Keep only recent messages
            if len(self.timestamps) > self.window_size:
                self.timestamps = self.timestamps[-self.window_size:]
            if len(self.header_stamps) > self.window_size:
                self.header_stamps = self.header_stamps[-self.window_size:]

    def get_hz(self) -> float:
        with self.lock:
            if len(self.timestamps) < 2:
                return 0.0
            duration = self.timestamps[-1] - self.timestamps[0]
            if duration <= 0:
                return 0.0
            return (len(self.timestamps) - 1) / duration

    def get_latency_ms(self) -> float:
        """Get average latency in milliseconds (receive_time - header_stamp)."""
        with self.lock:
            if not self.header_stamps:
                return -1.0
            latencies = [recv - stamp for recv, stamp in self.header_stamps]
            return (sum(latencies) / len(latencies)) * 1000.0


class HRIPerformanceMonitor(Node):
    """ROS2 node that monitors HRI topic performance."""

    # Topics to monitor: (topic_name, message_type, has_header)
    STATIC_TOPICS = [
        # Input
        ('/camera/image_raw', Image, True),

        # Face detection
        ('/humans/faces/tracked', IdsList, True),

        # Body detection
        ('/humans/bodies/tracked', IdsList, True),

        # Person manager
        ('/humans/persons/tracked', IdsList, True),

        # Face identification
        ('/humans/candidate_matches', 'hri_msgs/IdsMatch', True),
    ]

    # Topics that need emorobcare_cv_msgs
    EMOROBCARE_TOPICS = [
        ('/detected_objects', 'ObjectDetections', True),
        ('/pointing_target', 'PointingDetection', True),
        ('/gesture_recognition', 'GestureRecognitions', True),
        ('/processing_time', 'ProcessingTime', True),
    ]

    def __init__(self, csv_file: str = None, duration: float = None):
        super().__init__('hri_performance_monitor')

        self.csv_file = csv_file
        self.duration = duration
        self.start_time = time.time()

        self.topic_stats: dict[str, TopicStats] = {}
        self.dynamic_subscriptions = {}  # For dynamic topics like /humans/faces/<id>/...

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to static topics
        self._setup_static_subscriptions(sensor_qos)

        # Subscribe to tracked IDs to discover dynamic topics
        self._setup_dynamic_topic_discovery(sensor_qos)

        # Timer for printing stats
        self.print_timer = self.create_timer(2.0, self._print_stats)

        # CSV logging
        self.csv_writer = None
        self.csv_handle = None
        if csv_file:
            self._setup_csv(csv_file)

        self.get_logger().info('HRI Performance Monitor started')
        self.get_logger().info(f'Monitoring {len(self.topic_stats)} topics')

    def _setup_static_subscriptions(self, qos):
        """Set up subscriptions for static topics."""

        # Camera
        self._subscribe_topic('/camera/image_raw', Image, True, qos)

        # HRI tracked lists
        self._subscribe_topic('/humans/faces/tracked', IdsList, True, qos)
        self._subscribe_topic('/humans/bodies/tracked', IdsList, True, qos)
        self._subscribe_topic('/humans/persons/tracked', IdsList, True, qos)

        # Face identification - use generic subscriber
        try:
            from hri_msgs.msg import IdsMatch
            self._subscribe_topic('/humans/candidate_matches', IdsMatch, True, qos)
        except ImportError:
            self.get_logger().warn('hri_msgs/IdsMatch not available')

        # Emorobcare topics
        if EMOROBCARE_MSGS_AVAILABLE:
            self._subscribe_topic('/detected_objects', ObjectDetections, True, qos)
            self._subscribe_topic('/pointing_target', PointingDetection, True, qos)
            self._subscribe_topic('/gesture_recognition', GestureRecognitions, True, qos)
            self._subscribe_topic('/processing_time', ProcessingTime, True, qos)

    def _subscribe_topic(self, topic: str, msg_type, has_header: bool, qos):
        """Subscribe to a topic and track its stats."""
        self.topic_stats[topic] = TopicStats(topic)

        def callback(msg, topic_name=topic, has_hdr=has_header):
            receive_time = time.time()
            header_stamp = None

            if has_hdr and hasattr(msg, 'header'):
                header_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            self.topic_stats[topic_name].record_message(receive_time, header_stamp)

        self.create_subscription(msg_type, topic, callback, qos)
        self.get_logger().debug(f'Subscribed to {topic}')

    def _setup_dynamic_topic_discovery(self, qos):
        """Set up subscriptions to discover dynamic topics (face/body/person IDs)."""

        # Track face IDs to subscribe to engagement and yesno
        def faces_callback(msg: IdsList):
            for face_id in msg.ids:
                topic = f'/humans/faces/{face_id}/head_gesture'
                if topic not in self.topic_stats:
                    self._subscribe_dynamic_topic(topic, String, False, qos)

        def persons_callback(msg: IdsList):
            for person_id in msg.ids:
                # Engagement status
                topic = f'/humans/persons/{person_id}/engagement_status'
                if topic not in self.topic_stats:
                    try:
                        from hri_msgs.msg import EngagementLevel
                        self._subscribe_dynamic_topic(topic, EngagementLevel, True, qos)
                    except ImportError:
                        pass

        # Subscribe to tracked lists for dynamic discovery
        self.create_subscription(IdsList, '/humans/faces/tracked', faces_callback, qos)
        self.create_subscription(IdsList, '/humans/persons/tracked', persons_callback, qos)

    def _subscribe_dynamic_topic(self, topic: str, msg_type, has_header: bool, qos):
        """Subscribe to a dynamically discovered topic."""
        if topic in self.topic_stats:
            return

        self.topic_stats[topic] = TopicStats(topic)

        def callback(msg, topic_name=topic, has_hdr=has_header):
            receive_time = time.time()
            header_stamp = None

            if has_hdr and hasattr(msg, 'header'):
                header_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            self.topic_stats[topic_name].record_message(receive_time, header_stamp)

        sub = self.create_subscription(msg_type, topic, callback, qos)
        self.dynamic_subscriptions[topic] = sub
        self.get_logger().info(f'Discovered dynamic topic: {topic}')

    def _setup_csv(self, filename: str):
        """Set up CSV logging."""
        self.csv_handle = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_handle)
        self.csv_writer.writerow(['timestamp', 'topic', 'hz', 'latency_ms', 'msg_count'])

    def _print_stats(self):
        """Print current statistics."""
        elapsed = time.time() - self.start_time

        # Check duration limit
        if self.duration and elapsed >= self.duration:
            self.get_logger().info(f'Duration {self.duration}s reached, shutting down')
            self._save_final_stats()
            rclpy.shutdown()
            return

        # Build stats table
        print('\n' + '=' * 80)
        print(f'HRI Performance Monitor - Elapsed: {elapsed:.1f}s')
        print('=' * 80)
        print(f'{"Topic":<45} {"Hz":>8} {"Latency":>10} {"Count":>8}')
        print('-' * 80)

        # Sort topics by category
        categories = {
            'Input': ['/camera/image_raw'],
            'Face Detection': [t for t in self.topic_stats if 'faces' in t and 'head_gesture' not in t],
            'Body Detection': [t for t in self.topic_stats if 'bodies' in t],
            'Person Manager': [t for t in self.topic_stats if 'persons' in t and 'engagement' not in t],
            'Face ID': ['/humans/candidate_matches'],
            'Engagement': [t for t in self.topic_stats if 'engagement' in t],
            'Yes/No': [t for t in self.topic_stats if 'head_gesture' in t],
            'Object Detection': ['/detected_objects'],
            'Pointing': ['/pointing_target'],
            'Gesture': ['/gesture_recognition', '/processing_time'],
        }

        timestamp = datetime.now().isoformat()

        for category, topics in categories.items():
            existing_topics = [t for t in topics if t in self.topic_stats]
            if not existing_topics:
                continue

            print(f'\n[{category}]')
            for topic in existing_topics:
                stats = self.topic_stats[topic]
                hz = stats.get_hz()
                latency = stats.get_latency_ms()
                count = stats.message_count

                latency_str = f'{latency:.1f}ms' if latency >= 0 else 'N/A'

                # Truncate topic name for display
                display_topic = topic if len(topic) <= 44 else '...' + topic[-41:]
                print(f'{display_topic:<45} {hz:>7.1f}  {latency_str:>10} {count:>8}')

                # Log to CSV
                if self.csv_writer:
                    self.csv_writer.writerow([timestamp, topic, f'{hz:.2f}', f'{latency:.2f}', count])

        print('=' * 80)

        # Flush CSV
        if self.csv_handle:
            self.csv_handle.flush()

    def _save_final_stats(self):
        """Save final statistics before shutdown."""
        if self.csv_handle:
            self.csv_handle.close()
            self.get_logger().info(f'CSV saved to {self.csv_file}')

    def destroy_node(self):
        if self.csv_handle:
            self.csv_handle.close()
        super().destroy_node()


def wait_for_topics_ready(timeout: float = 60.0) -> bool:
    """
    Wait for all required topics to be publishing before starting measurements.
    Returns True if all topics are ready, False if timeout.
    """
    # Required topics that must be active
    REQUIRED_TOPICS = [
        '/camera/image_raw',
        '/humans/faces/tracked',
        '/humans/bodies/tracked',
        '/humans/persons/tracked',
    ]

    # Optional topics (nice to have, but don't block)
    OPTIONAL_TOPICS = [
        '/humans/candidate_matches',
        '/detected_objects',
        '/pointing_target',
        '/gesture_recognition',
    ]

    print('\n' + '=' * 60)
    print('STARTUP HEALTH CHECK')
    print('=' * 60)
    print(f'Waiting up to {timeout}s for required topics...\n')

    rclpy.init()
    node = rclpy.create_node('topic_checker')

    start_time = time.time()
    topics_ready = {t: False for t in REQUIRED_TOPICS}
    optional_ready = {t: False for t in OPTIONAL_TOPICS}

    while time.time() - start_time < timeout:
        # Get current topic list
        topic_list = node.get_topic_names_and_types()
        active_topics = [name for name, _ in topic_list]

        # Check required topics
        all_required_ready = True
        print('\r' + ' ' * 80 + '\r', end='')  # Clear line

        status_parts = []
        for topic in REQUIRED_TOPICS:
            if topic in active_topics:
                topics_ready[topic] = True
                status_parts.append(f'[OK] {topic.split("/")[-1]}')
            else:
                all_required_ready = False
                status_parts.append(f'[..] {topic.split("/")[-1]}')

        print(f'  {" | ".join(status_parts)}', end='', flush=True)

        # Check optional topics
        for topic in OPTIONAL_TOPICS:
            if topic in active_topics:
                optional_ready[topic] = True

        if all_required_ready:
            print('\n')
            break

        time.sleep(0.5)

    node.destroy_node()
    rclpy.shutdown()

    # Print summary
    elapsed = time.time() - start_time
    print('-' * 60)
    print('REQUIRED TOPICS:')
    for topic, ready in topics_ready.items():
        status = '\033[92m[READY]\033[0m' if ready else '\033[91m[MISSING]\033[0m'
        print(f'  {status} {topic}')

    print('\nOPTIONAL TOPICS:')
    for topic, ready in optional_ready.items():
        status = '\033[92m[READY]\033[0m' if ready else '\033[93m[NOT FOUND]\033[0m'
        print(f'  {status} {topic}')

    print('-' * 60)

    all_required = all(topics_ready.values())
    if all_required:
        print(f'\033[92mAll required topics ready in {elapsed:.1f}s\033[0m')
        print('=' * 60 + '\n')
        return True
    else:
        missing = [t for t, r in topics_ready.items() if not r]
        print(f'\033[91mTIMEOUT: Missing required topics: {missing}\033[0m')
        print('=' * 60 + '\n')
        return False


def main():
    parser = argparse.ArgumentParser(description='HRI Performance Monitor')
    parser.add_argument('--csv', type=str, help='Output CSV file path')
    parser.add_argument('--duration', type=float, help='Duration in seconds (then exit)')
    parser.add_argument('--wait-ready', action='store_true',
                        help='Wait for all required topics before starting')
    parser.add_argument('--wait-timeout', type=float, default=60.0,
                        help='Timeout for --wait-ready in seconds (default: 60)')
    args = parser.parse_args()

    # Step 2: Health check if requested
    if args.wait_ready:
        if not wait_for_topics_ready(args.wait_timeout):
            print('Aborting: Not all required topics are available.')
            print('Make sure the launcher and rosbag are running.')
            return 1

    # Step 3: Start monitoring
    rclpy.init()
    node = HRIPerformanceMonitor(csv_file=args.csv, duration=args.duration)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    return 0


if __name__ == '__main__':
    exit(main())
