"""
highlight_combined.py — Multi-signal highlight scoring

Fuses audio energy, computer-vision events, and (optionally) external
match-event data into a single per-second score, then extracts the
top-scoring windows as highlight segments.
"""

import numpy as np


def combine_signals(audio_events, cv_events, match_data=None,
                    video_duration_sec=5400):
    """
    Score every second of the match based on multiple signals.

    Args:
        audio_events: list of (start, end) from audio analysis
        cv_events: list of (timestamp, event_type, confidence) from CV
        match_data: optional dict with {"events": [{"minute": N, "type": "Goal"}, ...]}
        video_duration_sec: total video length in seconds

    Returns:
        list of (start_sec, end_sec, score) — highlight windows, ranked by score
    """
    scores = np.zeros(video_duration_sec)

    # --- Audio signal (weight: 1.0) ---
    for start, end in audio_events:
        s, e = int(start), min(int(end), video_duration_sec)
        scores[s:e] += 1.0

    # --- CV signal (weight by event type) ---
    cv_weights = {
        "ball_near_goal": 1.5,
        "fast_ball": 0.8,
        "player_cluster": 2.0,  # celebrations are a strong signal
    }
    for timestamp, event_type, confidence in cv_events:
        t = int(timestamp)
        w = cv_weights.get(event_type, 0.5) * confidence
        s = max(0, t - 10)
        e = min(video_duration_sec, t + 10)
        scores[s:e] += w

    # --- External data signal (if available) ---
    if match_data:
        for event in match_data.get("events", []):
            t = event["minute"] * 60  # approximate second
            event_type = event["type"]
            if event_type in ("Goal", "Penalty"):
                w = 5.0
            elif event_type in ("Red Card", "Yellow Card"):
                w = 3.0
            else:
                w = 1.0
            s = max(0, int(t) - 30)
            e = min(video_duration_sec, int(t) + 30)
            scores[s:e] += w

    # --- Extract top-scoring windows ---
    if np.any(scores > 0):
        threshold = np.percentile(scores[scores > 0], 70)
    else:
        threshold = 0

    highlights = _extract_windows(scores, threshold, min_gap=45, padding=10)
    return highlights


def _extract_windows(scores, threshold, min_gap=45, padding=10):
    """Extract contiguous windows above threshold, merge nearby ones."""
    above = scores >= threshold
    windows = []
    in_window = False
    start = 0

    for i in range(len(above)):
        if above[i] and not in_window:
            start = max(0, i - padding)
            in_window = True
        elif not above[i] and in_window:
            end = min(len(scores), i + padding)
            windows.append((start, end, float(np.max(scores[start:end]))))
            in_window = False

    if in_window:
        windows.append((start, len(scores), float(np.max(scores[start:]))))

    # Merge windows closer than min_gap
    merged = []
    for w in windows:
        if merged and w[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], w[1], max(merged[-1][2], w[2]))
        else:
            merged.append(list(w))

    # Sort by score descending
    merged.sort(key=lambda x: x[2], reverse=True)
    return [(int(s), int(e), sc) for s, e, sc in merged]
