"""
goal_detector.py — Map match goals to video timestamps

Loads match config, converts match minutes to video seconds using
detected kickoff offsets, and refines each goal timestamp using
audio peak detection.
"""

import json
from highlight_audio import find_peak_in_window


def load_match_config(config_path):
    """Load match configuration JSON."""
    with open(config_path) as f:
        return json.load(f)


def match_minute_to_video_sec(minute, kickoff_1st, kickoff_2nd):
    """
    Convert a match minute to a video timestamp in seconds.

    Args:
        minute: match minute (e.g. 6, 69, 91 for 90+1')
        kickoff_1st: video timestamp of 1st half kickoff (seconds)
        kickoff_2nd: video timestamp of 2nd half kickoff (seconds)

    Returns:
        estimated video timestamp in seconds
    """
    if minute <= 45:
        return kickoff_1st + minute * 60
    else:
        # Second half: minute 46 = 0:00 of second half
        second_half_minute = minute - 45
        return kickoff_2nd + second_half_minute * 60


def find_goal_peak(video_path, expected_sec, window_sec=120):
    """
    Find the exact moment of a goal by looking for the loudest audio
    peak in a window around the expected timestamp.

    Args:
        video_path: path to match video
        expected_sec: estimated goal timestamp
        window_sec: search +-window_sec around expected time

    Returns:
        peak_sec: refined timestamp of the goal moment
    """
    return find_peak_in_window(video_path, expected_sec, window_sec)


def get_all_goal_windows(video_path, config, kickoff_1st, kickoff_2nd):
    """
    For each goal in the config, compute the clip window:
    15s before the audio peak through 90s after.

    Args:
        video_path: path to match video
        config: parsed match config dict
        kickoff_1st: 1st half kickoff video timestamp
        kickoff_2nd: 2nd half kickoff video timestamp

    Returns:
        list of dicts with keys:
            goal_num, scorer, score_before, score_after, team,
            peak_sec, clip_start, clip_end
    """
    goals = config["goals"]
    match_info = config["match"]

    # Compute score_before for each goal
    prev_score = "0-0"
    goal_windows = []

    for i, goal in enumerate(goals):
        minute = goal["minute"]
        expected_sec = match_minute_to_video_sec(minute, kickoff_1st, kickoff_2nd)

        print(f"  [goal] Goal {i+1}: {goal['scorer']} ({minute}') "
              f"— expected at {expected_sec:.0f}s, searching audio peak...")

        peak_sec = find_goal_peak(video_path, expected_sec, window_sec=120)

        print(f"         Audio peak found at {peak_sec:.0f}s "
              f"({int(peak_sec)//60}:{int(peak_sec)%60:02d})")

        clip_start = max(0, peak_sec - 15)
        clip_end = peak_sec + 90

        goal_windows.append({
            "goal_num": i + 1,
            "scorer": goal["scorer"],
            "team": goal["team"],
            "score_before": prev_score,
            "score_after": goal["score_after"],
            "minute": minute,
            "peak_sec": peak_sec,
            "clip_start": clip_start,
            "clip_end": clip_end,
        })

        prev_score = goal["score_after"]

    return goal_windows


if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else "match.mp4"
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "match_config.json"

    config = load_match_config(cfg_path)
    # Use dummy kickoff values for testing
    print("Using dummy kickoff times (0s, 2700s) for testing.")
    print("Run via pipeline.py --mode goals for auto-detection.\n")

    windows = get_all_goal_windows(video, config, 0, 2700)
    for w in windows:
        print(f"  Goal {w['goal_num']}: {w['scorer']} ({w['minute']}') "
              f"  {w['score_before']} -> {w['score_after']}")
        print(f"    clip: {w['clip_start']:.0f}s - {w['clip_end']:.0f}s "
              f"(peak at {w['peak_sec']:.0f}s)")
