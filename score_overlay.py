"""
score_overlay.py — Generate ffmpeg drawtext filter for score overlay

Creates a scoreboard overlay that shows the score before the goal
during the buildup, then updates to the new score at the peak moment.
"""

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def build_score_filter(goal_info, clip_duration):
    """
    Build an ffmpeg drawtext filter chain for a single goal clip.

    The clip timeline:
    - 0s to peak_offset: show score_before (buildup)
    - peak_offset to end: show score_after (goal scored)

    A small background box + white text in top-center of the 9:16 frame.

    Args:
        goal_info: dict with keys score_before, score_after, peak_sec,
                   clip_start, clip_end, plus match home/away names
        clip_duration: total clip length in seconds

    Returns:
        ffmpeg filter string (drawtext chain)
    """
    score_before = goal_info["score_before"]
    score_after = goal_info["score_after"]
    home = goal_info.get("home", "HOME")
    away = goal_info.get("away", "AWAY")

    # Peak offset relative to clip start
    peak_offset = goal_info["peak_sec"] - goal_info["clip_start"]

    # Text content
    text_before = f"{home}  {score_before}  {away}"
    text_after = f"{home}  {score_after}  {away}"

    filters = []

    # Score before the goal (buildup phase)
    filters.append(
        f"drawtext=fontfile='{FONT_PATH}'"
        f":text='{_escape(text_before)}'"
        f":fontcolor=white:fontsize=42"
        f":box=1:boxcolor=black@0.6:boxborderw=12"
        f":x=(w-text_w)/2:y=60"
        f":enable='between(t,0,{peak_offset:.2f})'"
    )

    # Score after the goal
    filters.append(
        f"drawtext=fontfile='{FONT_PATH}'"
        f":text='{_escape(text_after)}'"
        f":fontcolor=white:fontsize=42"
        f":box=1:boxcolor=black@0.6:boxborderw=12"
        f":x=(w-text_w)/2:y=60"
        f":enable='between(t,{peak_offset:.2f},{clip_duration:.2f})'"
    )

    # Scorer name flash (shown for 5s after the goal)
    scorer_text = f"GOAL! {goal_info['scorer']}"
    flash_end = min(peak_offset + 5, clip_duration)
    filters.append(
        f"drawtext=fontfile='{FONT_PATH}'"
        f":text='{_escape(scorer_text)}'"
        f":fontcolor=yellow:fontsize=36"
        f":box=1:boxcolor=black@0.5:boxborderw=8"
        f":x=(w-text_w)/2:y=120"
        f":enable='between(t,{peak_offset:.2f},{flash_end:.2f})'"
    )

    return ",".join(filters)


def _escape(text):
    """Escape special characters for ffmpeg drawtext."""
    return (text
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace(":", "\\:")
            .replace("%", "%%"))


if __name__ == "__main__":
    # Example usage
    goal = {
        "score_before": "0-0",
        "score_after": "0-1",
        "scorer": "Christensen",
        "home": "RMA",
        "away": "BAR",
        "peak_sec": 415,
        "clip_start": 400,
        "clip_end": 505,
    }
    filt = build_score_filter(goal, clip_duration=105)
    print("Filter chain:")
    print(filt)
