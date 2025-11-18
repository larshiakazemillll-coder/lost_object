import re
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path

def clear_file(file_path):
    """Safely clear the contents of a file (keep the file itself)."""
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"File does not exist: {file_path}")
            return False
        path.write_text("")
        print(f"Successfully cleared: {file_path}")
        return True
    except PermissionError:
        print(f"Permission denied: Cannot clear {file_path}")
    except Exception as e:
        print(f"Error clearing {file_path}: {e}")
    return False

def decide_tracker_movement(log_path):
    """
    Decide movement state for each tracker based on log entries.
    Rules:
    - If log says "lateral movement detected" → moving
    - If log says "false positive" → stationary, and ignore later depth movement
    - If log says "depth movement detected" and not preceded by false positive → moving
    - Else:
        * Slice directions into 2 halves of 20 samples
        * If mean direction in both halves is same → moving
        * If less than 50% movement in any half → undecided
        * If majority stationary but first != last half → moving
        * Otherwise stationary
    """

    pattern_main = re.compile(
        r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s+-\s+Track ID (?P<id>\d+).*Main direction\s*=\s*(?P<dir>\w+)"
    )
    pattern_lateral = re.compile(r"Track ID (\d+) is moving \(lateral movement detected\)")
    pattern_depth = re.compile(r"Track ID (\d+) is moving \(depth movement detected\)")
    pattern_false = re.compile(r"Track ID (\d+) was a false positive")

    track_data = {}
    movement_state = {}
    false_positive_ids = set()

    with open(log_path, "r") as f:
        for line in f:
            # False positive
            m_false = pattern_false.search(line)
            if m_false:
                tid = int(m_false.group(1))
                movement_state[tid] = "stationary"
                false_positive_ids.add(tid)
                continue

            # Lateral movement → moving
            m_lat = pattern_lateral.search(line)
            if m_lat:
                tid = int(m_lat.group(1))
                movement_state[tid] = "moving"
                continue

            # Depth movement (only if not false positive before)
            m_dep = pattern_depth.search(line)
            if m_dep:
                tid = int(m_dep.group(1))
                if tid not in false_positive_ids:
                    movement_state[tid] = "moving"
                continue

            # Main direction log
            m_main = pattern_main.search(line)
            if m_main:
                tid = int(m_main.group("id"))
                t = datetime.strptime(m_main.group("time"), "%Y-%m-%d %H:%M:%S,%f")
                direction = m_main.group("dir").lower()
                track_data.setdefault(tid, []).append((t, direction))

    # For trackers without direct movement log → decide from main directions
    for tid, records in track_data.items():
        if tid in movement_state:
            continue  # Already decided from logs

        if len(records) < 20:
            movement_state[tid] = "undecided"
            continue

        # Sort by time
        records.sort(key=lambda x: x[0])
        directions = [d for _, d in records]

        half_size = len(directions) // 2
        first_half = directions[:half_size]
        second_half = directions[half_size:]

        def direction_mean(part):
            return sum(d != "stationary" for d in part) / len(part) * 100

        mean_first = direction_mean(first_half)
        mean_second = direction_mean(second_half)

        if mean_first >= 50 and mean_second >= 50:
            if all(d != "stationary" for d in first_half) and all(d != "stationary" for d in second_half):
                movement_state[tid] = "moving"
            else:
                if first_half[0] != second_half[-1]:
                    movement_state[tid] = "moving"
                else:
                    movement_state[tid] = "stationary"
        else:
            movement_state[tid] = "undecided"

    return movement_state, track_data


def plot_tracker_movements(track_data, movement_state, save_path=None):
    color_map = {
        "stationary": "gray",
        "up": "blue",
        "down": "red",
        "left": "green",
        "right": "orange"
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for tid in sorted(track_data.keys()):
        times = [t for t, _ in track_data[tid]]
        directions = [d for _, d in track_data[tid]]

        ax.plot(times, [tid] * len(times), color="black", linewidth=1, alpha=0.3)

        for t, d in zip(times, directions):
            ax.scatter(t, tid, color=color_map.get(d, "black"))

        ax.text(times[-1], tid + 0.1, movement_state.get(tid, ""), fontsize=9, color="purple")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)
    ax.set_yticks(sorted(track_data.keys()))
    ax.set_xlabel("Time")
    ax.set_ylabel("Tracker ID")
    ax.set_title("Tracker Movements Over Time")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    
    # parser.add_argument("--save-plot", help="./")
    # args = parser.parse_args()

    movement_state, track_data = decide_tracker_movement('/home/arshia/Documents/Git hub/lost_object/unattended_items_at_airport_update.log')
    for tid, state in movement_state.items():
        print(f"Tracker {tid}: {state}")

    plot_tracker_movements(track_data, movement_state, save_path='./1')
