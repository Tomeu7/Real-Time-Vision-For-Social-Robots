#!/usr/bin/env python3
"""
### MODIFIED FOR TIMING MEASUREMENT LBR HRI ###

HRI Timing Monitor - Monitors processing time from all HRI nodes.

Usage:
    python3 hri_timing_monitor.py
    python3 hri_timing_monitor.py --csv timing_results.csv
    python3 hri_timing_monitor.py --duration 60
    python3 hri_timing_monitor.py --output data.json  # Save all measurements for plotting

Monitors these topics:
    /processing_time/engagement
    /processing_time/pointing
    /processing_time/object_detection
    /processing_time/gesture
    /processing_time/yesno
    /processing_time/body_detect
    /processing_time/face_detect
    /processing_time/face_id

Output JSON format (for plotting):
{
    "start_time": "2024-01-15T10:30:00",
    "duration_sec": 60.5,
    "nodes": {
        "gesture": {
            "timestamps": [0.1, 0.2, ...],  # seconds from start
            "values_ms": [8.2, 7.9, ...]    # processing times in ms
        },
        ...
    }
}
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from threading import Lock
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from emorobcare_cv_msgs.msg import ProcessingTime


class TimingStats:
    """Track timing statistics for a single node."""

    def __init__(self, name: str, start_time: float):
        self.name = name
        self.start_time = start_time  # Reference start time
        self.lock = Lock()
        self.count = 0
        self.total_ms = 0.0
        self.min_ms = float('inf')
        self.max_ms = 0.0
        self.recent_times = []  # Last N measurements for rolling average
        self.window_size = 50

        # Store ALL measurements for plotting
        self.all_timestamps = []  # Seconds from start
        self.all_values_ms = []   # Processing time in ms

    def record(self, delta_ms: float):
        with self.lock:
            self.count += 1
            self.total_ms += delta_ms
            self.min_ms = min(self.min_ms, delta_ms)
            self.max_ms = max(self.max_ms, delta_ms)
            self.recent_times.append(delta_ms)
            if len(self.recent_times) > self.window_size:
                self.recent_times = self.recent_times[-self.window_size:]

            # Store for plotting
            timestamp = time.time() - self.start_time
            self.all_timestamps.append(timestamp)
            self.all_values_ms.append(delta_ms)

    def get_stats(self):
        with self.lock:
            if self.count == 0:
                return None

            avg_ms = self.total_ms / self.count

            if self.recent_times:
                recent_avg = sum(self.recent_times) / len(self.recent_times)
                # Rolling std (population std)
                var = sum((x - recent_avg) ** 2 for x in self.recent_times) / len(self.recent_times)
                recent_std = math.sqrt(var)
            else:
                recent_avg = 0.0
                recent_std = 0.0

            return {
                'count': self.count,
                'avg_ms': avg_ms,
                'min_ms': self.min_ms if self.min_ms != float('inf') else 0,
                'max_ms': self.max_ms,
                'recent_avg_ms': recent_avg,
                'recent_std_ms': recent_std,
                'hz': len(self.recent_times) / (self.recent_times[-1] - self.recent_times[0]) if len(self.recent_times) > 1 else 0
            }



    def get_all_data(self):
        """Get all recorded data for plotting."""
        with self.lock:
            return {
                'timestamps': self.all_timestamps.copy(),
                'values_ms': self.all_values_ms.copy()
            }


class HRITimingMonitor(Node):
    """ROS2 node that monitors processing times from all HRI nodes."""

    # Processing time topics to monitor
    TIMING_TOPICS = [
        '/processing_time/engagement',
        '/processing_time/multiuser_engagement',
        '/processing_time/pointing',
        '/processing_time/object_detection',
        '/processing_time/gesture',
        '/processing_time/yesno',
        '/processing_time/body_detect'
    ]

    def __init__(self, csv_file: str = None, duration: float = None, output_file: str = None):
        super().__init__('hri_timing_monitor')

        self.csv_file = csv_file
        self.output_file = output_file  # JSON output for plotting
        self.duration = duration
        self.start_time = time.time()
        self.start_datetime = datetime.now().isoformat()

        # Stats for each node
        self.stats: dict[str, TimingStats] = {}

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to all timing topics
        for topic in self.TIMING_TOPICS:
            node_name = topic.split('/')[-1]
            self.stats[node_name] = TimingStats(node_name, self.start_time)
            self.create_subscription(
                ProcessingTime,
                topic,
                lambda msg, name=node_name: self._timing_callback(msg, name),
                qos
            )
            self.get_logger().info(f'Subscribed to {topic}')

        # Timer for printing stats
        self.print_timer = self.create_timer(2.0, self._print_stats)

        # CSV logging
        self.csv_writer = None
        self.csv_handle = None
        if csv_file:
            self._setup_csv(csv_file)

        self.get_logger().info('HRI Timing Monitor started')

    def _timing_callback(self, msg: ProcessingTime, node_name: str):
        """Handle incoming ProcessingTime message."""
        # Convert delta to milliseconds
        delta_ms = msg.delta.sec * 1000.0 + msg.delta.nanosec / 1_000_000.0
        self.stats[node_name].record(delta_ms)

    def _setup_csv(self, filename: str):
        """Set up CSV logging."""
        self.csv_handle = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_handle)
        self.csv_writer.writerow([
            'timestamp', 'node', 'count', 'avg_ms', 'recent_avg_ms', 'min_ms', 'max_ms'
        ])

    def _print_stats(self):
        """Print current timing statistics."""
        elapsed = time.time() - self.start_time

        # Check duration limit
        if self.duration and elapsed >= self.duration:
            self.get_logger().info(f'Duration {self.duration}s reached, shutting down')
            self._save_final_summary()
            rclpy.shutdown()
            return

        # Clear screen and print header
        print('\033[2J\033[H', end='')  # Clear screen
        print('=' * 80)
        print(f'HRI TIMING MONITOR - Elapsed: {elapsed:.1f}s')
        print('=' * 80)
        print(f'{"Node":<20} {"Count":>8} {"Avg(ms)":>10} {"Recent":>10} {"Std":>9} {"Min":>10} {"Max":>10}')
        print('-' * 80)

        timestamp = datetime.now().isoformat()
        total_time = 0.0
        active_nodes = 0

        for node_name in ['body_detect', 'face_detect', 'face_id', 'gesture', 'pointing', 'object_detection', 'engagement', 'multiuser_engagement', 'yesno']:
            if node_name not in self.stats:
                continue

            stats = self.stats[node_name].get_stats()
            if stats is None:
                print(f'{node_name:<20} {"--":>8} {"--":>10} {"--":>10} {"--":>10} {"--":>10}')
                continue

            active_nodes += 1
            total_time += stats['recent_avg_ms']

            # Color code based on processing time
            avg = stats['recent_avg_ms']
            if avg < 20:
                color = '\033[92m'  # Green
            elif avg < 50:
                color = '\033[93m'  # Yellow
            else:
                color = '\033[91m'  # Red
            reset = '\033[0m'

            print(f'{node_name:<20} {stats["count"]:>8} {stats["avg_ms"]:>9.1f}ms '
                f'{color}{stats["recent_avg_ms"]:>9.1f}ms{reset} '
                f'{stats["recent_std_ms"]:>8.2f} '
                f'{stats["min_ms"]:>9.1f}ms {stats["max_ms"]:>9.1f}ms')


            # Log to CSV
            if self.csv_writer:
                self.csv_writer.writerow([
                    timestamp, node_name, stats['count'],
                    f'{stats["avg_ms"]:.2f}', f'{stats["recent_avg_ms"]:.2f}',
                    f'{stats["min_ms"]:.2f}', f'{stats["max_ms"]:.2f}'
                ])

        print('-' * 80)
        if active_nodes > 0:
            print(f'{"TOTAL (sum)":<20} {"":>8} {"":>10} {total_time:>9.1f}ms')
        print('=' * 80)
        print('\nColor: \033[92mGreen\033[0m < 20ms | \033[93mYellow\033[0m < 50ms | \033[91mRed\033[0m >= 50ms')
        print('Press Ctrl+C to stop')

        # Flush CSV
        if self.csv_handle:
            self.csv_handle.flush()

    def _save_final_summary(self):
        """Save final summary before shutdown."""
        elapsed = time.time() - self.start_time

        print('\n' + '=' * 80)
        print('FINAL SUMMARY')
        print('=' * 80)

        for node_name, timing_stats in self.stats.items():
            stats = timing_stats.get_stats()
            if stats:
                print(f'{node_name}: {stats["count"]} calls, '
                      f'avg={stats["avg_ms"]:.1f}ms, '
                      f'min={stats["min_ms"]:.1f}ms, '
                      f'max={stats["max_ms"]:.1f}ms')

        if self.csv_handle:
            self.csv_handle.close()
            print(f'\nCSV saved to {self.csv_file}')

        # Save JSON with all data for plotting
        if self.output_file:
            output_data = {
                'start_time': self.start_datetime,
                'duration_sec': elapsed,
                'nodes': {}
            }
            for node_name, timing_stats in self.stats.items():
                data = timing_stats.get_all_data()
                if data['timestamps']:  # Only include if we have data
                    stats = timing_stats.get_stats()
                    output_data['nodes'][node_name] = {
                        'timestamps': data['timestamps'],
                        'values_ms': data['values_ms'],
                        'summary': {
                            'count': stats['count'],
                            'avg_ms': stats['avg_ms'],
                            'min_ms': stats['min_ms'],
                            'max_ms': stats['max_ms']
                        }
                    }

            with open(self.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f'JSON data saved to {self.output_file}')
            print(f'Total data points: {sum(len(d["timestamps"]) for d in output_data["nodes"].values())}')

    def destroy_node(self):
        if self.csv_handle:
            self.csv_handle.close()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description='HRI Timing Monitor')
    parser.add_argument('--csv', type=str, help='Output CSV file path (periodic stats)')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file for plotting (all data points)')
    parser.add_argument('--duration', type=float, help='Duration in seconds (then exit)')
    args = parser.parse_args()

    rclpy.init()
    node = HRITimingMonitor(csv_file=args.csv, duration=args.duration, output_file=args.output)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nShutting down...')
        node._save_final_summary()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    return 0


if __name__ == '__main__':
    exit(main())
