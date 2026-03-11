"""
highlight_audio.py — Audio-based highlight detection

Detects highlights by finding moments where crowd noise / commentator
voice spike above the average energy level of the match.
"""

import numpy as np
from moviepy import VideoFileClip
from scipy.signal import medfilt


def detect_highlights_audio(video_path, top_percent=10, window_sec=10, merge_gap=60):
    """
    Detect highlight timestamps based on audio energy peaks.

    Args:
        video_path: path to full match .mp4
        top_percent: percent of loudest windows to keep (10 = top 10%)
        window_sec: averaging window in seconds
        merge_gap: merge events within this many seconds

    Returns:
        list of (start_sec, end_sec) tuples
    """
    clip = VideoFileClip(video_path)

    audio = clip.audio
    if audio is None:
        clip.close()
        print("  [audio] No audio track found.")
        return []

    fps = audio.fps  # typically 44100

    # Extract audio as numpy array
    audio_array = audio.to_soundarray()  # shape: (n_samples, 2)
    mono = np.mean(audio_array, axis=1)  # average channels to mono

    samples_per_sec = fps
    n_seconds = len(mono) // samples_per_sec

    if n_seconds == 0:
        clip.close()
        print("  [audio] Audio too short to analyse.")
        return []

    # RMS energy for each second
    energy = np.array([
        np.sqrt(np.mean(mono[i * samples_per_sec:(i + 1) * samples_per_sec] ** 2))
        for i in range(n_seconds)
    ])

    # Smooth with a median-filter rolling window
    kernel_size = window_sec if window_sec % 2 == 1 else window_sec + 1
    smoothed = medfilt(energy, kernel_size=kernel_size)

    # Threshold: keep top N%
    threshold = np.percentile(smoothed, 100 - top_percent)

    # Timestamps above threshold
    peak_times = np.where(smoothed >= threshold)[0]  # in seconds

    # Group nearby peaks (merge within merge_gap seconds)
    events = []
    if len(peak_times) > 0:
        current_start = peak_times[0]
        current_end = peak_times[0]

        for t in peak_times[1:]:
            if t - current_end <= merge_gap:
                current_end = t
            else:
                events.append((max(0, current_start - 5), current_end + 10))
                current_start = t
                current_end = t
        events.append((max(0, current_start - 5), min(current_end + 10, n_seconds)))

    clip.close()
    return events


def find_peak_in_window(video_path, center_sec, window_sec=120):
    """
    Find the loudest audio moment within a narrow time window.

    Used by goal_detector to refine the expected goal timestamp
    by finding the actual crowd/commentator audio peak.

    Args:
        video_path: path to match video
        center_sec: center of the search window (seconds)
        window_sec: search +-window_sec around center

    Returns:
        peak_sec: timestamp of the loudest moment (seconds)
    """
    clip = VideoFileClip(video_path)
    audio = clip.audio

    if audio is None:
        clip.close()
        return center_sec

    fps = audio.fps
    duration = clip.duration

    start = max(0, center_sec - window_sec)
    end = min(duration, center_sec + window_sec)

    # Extract just the audio in our window
    window_audio = audio.subclipped(start, end)
    audio_array = window_audio.to_soundarray()
    mono = np.mean(audio_array, axis=1)

    samples_per_sec = fps
    n_seconds = len(mono) // samples_per_sec

    if n_seconds == 0:
        clip.close()
        return center_sec

    # RMS energy per second within the window
    energy = np.array([
        np.sqrt(np.mean(mono[i * samples_per_sec:(i + 1) * samples_per_sec] ** 2))
        for i in range(n_seconds)
    ])

    # Smooth slightly to avoid catching a random pop
    kernel_size = min(5, len(energy))
    if kernel_size % 2 == 0:
        kernel_size = max(1, kernel_size - 1)
    if kernel_size >= 3:
        energy = medfilt(energy, kernel_size=kernel_size)

    peak_offset = int(np.argmax(energy))
    peak_sec = start + peak_offset

    clip.close()
    return peak_sec


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "match.mp4"
    events = detect_highlights_audio(path, top_percent=10)
    print(f"Found {len(events)} highlight events:")
    for i, (start, end) in enumerate(events):
        m_s, s_s = divmod(int(start), 60)
        m_e, s_e = divmod(int(end), 60)
        print(f"  [{i + 1}] {m_s:02d}:{s_s:02d} -> {m_e:02d}:{s_e:02d}  "
              f"(duration: {end - start:.0f}s)")
