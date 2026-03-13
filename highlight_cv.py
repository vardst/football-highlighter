"""
highlight_cv.py — YOLO-based event detection

Uses YOLOv8 to detect players and the ball, then infers highlight-worthy
events: ball near goal, fast ball movement, player clustering (celebrations),
and goalkeeper diving near the ball (goal signal).
"""

import cv2
import numpy as np

from soccer_detector import SoccerDetector


def detect_events_yolo(video_path, sample_rate=5, model_path=None,
                       use_sahi=False):
    """
    Detect potential highlight events using YOLO object detection.

    Looks for:
    - Ball near the goal area (left/right edges of frame)
    - Rapid ball movement between frames
    - Player clustering (celebrations, fouls)
    - Goalkeeper near ball in goal area (strong goal signal)

    Args:
        video_path: path to match video
        sample_rate: process every Nth frame (5 ≈ 6fps for 30fps video)
        model_path: path to YOLO .pt weights (None = auto-detect best model)
        use_sahi: enable SAHI sliced inference

    Returns:
        list of (timestamp_sec, event_type, confidence)
    """
    detector = SoccerDetector(model_path=model_path, use_sahi=use_sahi)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    events = []
    prev_ball_pos = None
    frame_idx = 0

    print(f"  [cv] Scanning {total_frames} frames (every {sample_rate}th)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps

        result = detector.detect(frame)

        frame_w = frame.shape[1]
        frame_h = frame.shape[0]

        # Collect person positions (players + goalkeepers for clustering)
        persons = []
        for px in result.players_x:
            persons.append((px * frame_w, frame_h * 0.5))  # approx center y
        for gx in result.goalkeepers_x:
            persons.append((gx * frame_w, frame_h * 0.5))

        ball = None
        if result.ball_x is not None:
            ball = (result.ball_x * frame_w, result.ball_y * frame_h)

        # --- Event: Ball near goal area (left/right 15% of frame) ---
        if ball:
            if ball[0] < frame_w * 0.15 or ball[0] > frame_w * 0.85:
                events.append((timestamp, "ball_near_goal", 0.7))

            # Fast ball movement between sampled frames
            if prev_ball_pos:
                dist = np.sqrt(
                    (ball[0] - prev_ball_pos[0]) ** 2
                    + (ball[1] - prev_ball_pos[1]) ** 2
                )
                if dist > frame_w * 0.15:
                    events.append((timestamp, "fast_ball", 0.6))
            prev_ball_pos = ball

        # --- Event: Goalkeeper near ball in goal area (goal signal) ---
        if ball and result.goalkeepers_x:
            ball_in_goal_area = (ball[0] < frame_w * 0.2 or
                                 ball[0] > frame_w * 0.8)
            if ball_in_goal_area:
                for gx in result.goalkeepers_x:
                    gk_px = gx * frame_w
                    dist_to_ball = abs(gk_px - ball[0])
                    if dist_to_ball < frame_w * 0.1:
                        events.append((timestamp, "goalkeeper_save", 0.85))
                        break

        # --- Event: Player clustering (celebration / foul) ---
        if len(persons) >= 4:
            centroid = np.mean(persons, axis=0)
            distances = [
                np.sqrt((p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
                for p in persons
            ]
            avg_dist = np.mean(distances)
            if avg_dist < frame_w * 0.08:
                events.append((timestamp, "player_cluster", 0.8))

        frame_idx += 1

        # Progress
        if frame_idx % (sample_rate * 500) == 0:
            pct = frame_idx / total_frames * 100
            print(f"  [cv] {pct:.0f}% done...")

    cap.release()
    print(f"  [cv] Found {len(events)} raw CV events.")
    return events


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "match.mp4"
    model = sys.argv[2] if len(sys.argv) > 2 else None
    events = detect_events_yolo(path, model_path=model)
    for ts, etype, conf in events[:20]:
        m, s = divmod(int(ts), 60)
        print(f"  {m:02d}:{s:02d}  {etype:20s}  conf={conf:.2f}")
    if len(events) > 20:
        print(f"  ... and {len(events) - 20} more events")
