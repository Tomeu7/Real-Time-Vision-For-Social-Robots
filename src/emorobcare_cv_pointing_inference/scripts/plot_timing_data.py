#!/usr/bin/env python3
"""
### MODIFIED FOR TIMING MEASUREMENT LBR HRI ###

Plot timing data from hri_timing_monitor.py JSON output.

Usage:
    python3 plot_timing_data.py timing_data.json
    python3 plot_timing_data.py timing_data.json --save plot.png
    python3 plot_timing_data.py timing_data.json --rolling 20  # Rolling average window
"""

import argparse
import json
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required for plotting.")
    print("Install with: pip install matplotlib numpy")
    sys.exit(1)


def load_data(filepath: str) -> dict:
    """Load timing data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def rolling_average(values: list, window: int) -> list:
    """Compute rolling average."""
    if window <= 1:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def plot_timing_data(data: dict, rolling_window: int = 1, save_path: str = None):
    """Create plots from timing data."""
    nodes = data['nodes']
    duration = data['duration_sec']

    if not nodes:
        print("No data to plot!")
        return

    # Create figure with subplots
    n_nodes = len(nodes)
    fig, axes = plt.subplots(n_nodes + 1, 1, figsize=(12, 3 * (n_nodes + 1)), sharex=True)
    if n_nodes == 0:
        return

    # Colors for each node
    colors = {
        'gesture': '#2ecc71',       # Green
        'pointing': '#3498db',      # Blue
        'object_detection': '#e74c3c',  # Red
        'engagement': '#9b59b6',    # Purple
        'yesno': '#f39c12'          # Orange
    }

    # Plot each node
    for idx, (node_name, node_data) in enumerate(nodes.items()):
        ax = axes[idx]
        timestamps = node_data['timestamps']
        values = node_data['values_ms']
        summary = node_data['summary']

        color = colors.get(node_name, '#95a5a6')

        # Plot raw data (light)
        ax.plot(timestamps, values, color=color, alpha=0.3, linewidth=0.5, label='Raw')

        # Plot rolling average (dark)
        if rolling_window > 1:
            smoothed = rolling_average(values, rolling_window)
            ax.plot(timestamps, smoothed, color=color, linewidth=1.5, label=f'Rolling avg ({rolling_window})')

        # Add horizontal line for average
        ax.axhline(y=summary['avg_ms'], color=color, linestyle='--', alpha=0.7, label=f'Avg: {summary["avg_ms"]:.1f}ms')

        ax.set_ylabel(f'{node_name}\n(ms)')
        ax.set_title(f'{node_name}: avg={summary["avg_ms"]:.1f}ms, min={summary["min_ms"]:.1f}ms, max={summary["max_ms"]:.1f}ms, n={summary["count"]}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Combined plot (all nodes together)
    ax_combined = axes[-1]
    for node_name, node_data in nodes.items():
        color = colors.get(node_name, '#95a5a6')
        timestamps = node_data['timestamps']
        values = node_data['values_ms']

        if rolling_window > 1:
            values = rolling_average(values, rolling_window)

        ax_combined.plot(timestamps, values, color=color, linewidth=1, alpha=0.8, label=node_name)

    ax_combined.set_xlabel('Time (seconds)')
    ax_combined.set_ylabel('Processing time (ms)')
    ax_combined.set_title('All nodes combined')
    ax_combined.legend(loc='upper right', fontsize=8)
    ax_combined.grid(True, alpha=0.3)
    ax_combined.set_ylim(bottom=0)

    # Overall title
    fig.suptitle(f'HRI Timing Analysis - Duration: {duration:.1f}s - Started: {data["start_time"]}',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    else:
        plt.show()


def print_summary(data: dict):
    """Print summary statistics."""
    print('\n' + '=' * 60)
    print('TIMING SUMMARY')
    print('=' * 60)
    print(f'Start time: {data["start_time"]}')
    print(f'Duration: {data["duration_sec"]:.1f} seconds')
    print('-' * 60)

    total_avg = 0
    for node_name, node_data in data['nodes'].items():
        s = node_data['summary']
        print(f'{node_name:20} | avg: {s["avg_ms"]:6.1f}ms | min: {s["min_ms"]:6.1f}ms | max: {s["max_ms"]:6.1f}ms | n: {s["count"]}')
        total_avg += s['avg_ms']

    print('-' * 60)
    print(f'{"TOTAL (sum of avg)":<20} | {total_avg:6.1f}ms')
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='Plot HRI timing data')
    parser.add_argument('input', type=str, help='Input JSON file from hri_timing_monitor.py')
    parser.add_argument('--save', '-s', type=str, help='Save plot to file (e.g., plot.png)')
    parser.add_argument('--rolling', '-r', type=int, default=10,
                        help='Rolling average window size (default: 10)')
    parser.add_argument('--no-plot', action='store_true', help='Only print summary, no plot')
    args = parser.parse_args()

    # Load data
    try:
        data = load_data(args.input)
    except FileNotFoundError:
        print(f'Error: File not found: {args.input}')
        return 1
    except json.JSONDecodeError as e:
        print(f'Error: Invalid JSON file: {e}')
        return 1

    # Print summary
    print_summary(data)

    # Plot
    if not args.no_plot:
        plot_timing_data(data, rolling_window=args.rolling, save_path=args.save)

    return 0


if __name__ == '__main__':
    sys.exit(main())
