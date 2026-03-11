"""
kickoff_detector.py — Auto-detect match kickoff via broadcast clock OCR

Reads the broadcast scoreboard clock from sampled frames using EasyOCR,
then computes the video-to-match time offset (kickoff timestamp).

Falls back to kickoff_offsets in match_config.json if OCR fails.
"""

import cv2
import numpy as np
import re


def detect_kickoff(video_path, config=None):
    """
    Auto-detect first-half and second-half kickoff timestamps by
    reading the broadcast match clock via OCR.

    Samples frames during gameplay, reads the match clock from the
    broadcast scoreboard, and computes the video-to-match time offset.

    Falls back to config['kickoff_offsets'] if not enough OCR readings.

    Returns:
        (first_half_start_sec, second_half_start_sec)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    # Try OCR-based detection
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)

        first_half = _ocr_calibrate(
            cap, reader, fps, video_duration,
            scan_start=360, scan_end=min(2800, video_duration * 0.45),
            scan_step=60, half=1, label="1st half",
        )

        second_half = _ocr_calibrate(
            cap, reader, fps, video_duration,
            scan_start=max(3300, video_duration * 0.5),
            scan_end=min(5500, video_duration * 0.85),
            scan_step=60, half=2, label="2nd half",
        )
    except ImportError:
        print("  [kickoff] easyocr not installed, skipping OCR detection")
        first_half = None
        second_half = None

    cap.release()

    # Fall back to config if OCR didn't produce results
    fallback = (config or {}).get("kickoff_offsets", {})

    if first_half is None:
        first_half = fallback.get("first_half")
        if first_half is not None:
            print(f"  [kickoff] 1st half: using config value {first_half}s")
        else:
            print("  [kickoff] WARNING: could not detect 1st half kickoff")
            first_half = 0

    if second_half is None:
        second_half = fallback.get("second_half")
        if second_half is not None:
            print(f"  [kickoff] 2nd half: using config value {second_half}s")
        else:
            # Estimate: 1st half + ~55 min (45 play + ~10 halftime)
            second_half = first_half + 55 * 60
            print(f"  [kickoff] 2nd half: estimated at {second_half}s")

    _print_kickoff(first_half, "1st half")
    _print_kickoff(second_half, "2nd half")

    return first_half, second_half


def _ocr_calibrate(cap, reader, fps, video_duration,
                   scan_start, scan_end, scan_step, half, label):
    """
    Scan frames in a time range, OCR the scoreboard clock,
    and compute the kickoff offset from multiple readings.
    """
    offsets = []

    print(f"  [kickoff] Scanning {label} ({int(scan_start)}-{int(scan_end)}s, "
          f"every {scan_step}s)...")

    for ts in range(int(scan_start), int(scan_end), scan_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ret, frame = cap.read()
        if not ret:
            continue

        clock_reading = _read_clock(frame, reader)
        if clock_reading is None:
            continue

        mins, secs = clock_reading
        match_sec = mins * 60 + secs

        if half == 1 and 0 <= mins <= 50:
            offset = ts - match_sec
            offsets.append(offset)
        elif half == 2 and 46 <= mins <= 99:
            second_half_sec = match_sec - 45 * 60
            offset = ts - second_half_sec
            offsets.append(offset)

    if len(offsets) < 2:
        print(f"  [kickoff] {label}: only {len(offsets)} OCR readings, insufficient")
        return None

    # Use median for robustness against outliers
    median_offset = float(np.median(offsets))
    print(f"  [kickoff] {label}: {len(offsets)} OCR readings, "
          f"median offset = {median_offset:.0f}s")
    return median_offset


def _read_clock(frame, reader):
    """
    Try to read the match clock from the broadcast scoreboard.
    Returns (minutes, seconds) or None if not found.
    """
    h, w = frame.shape[:2]

    # Crop top-left scoreboard region (works for most broadcast formats)
    region = frame[int(h * 0.01):int(h * 0.14), int(w * 0.01):int(w * 0.17)]

    # Upscale for better OCR
    big = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    results = reader.readtext(big)

    for _, text, conf in results:
        text = text.strip()
        if conf < 0.1:
            continue

        parsed = _parse_clock_text(text)
        if parsed:
            return parsed

    return None


def _parse_clock_text(text):
    """
    Parse a match clock reading from OCR text.
    Handles common OCR artifacts: ':' read as '.', \"'\", '*', '4', etc.
    """
    # Direct MM:SS with standard separators
    m = re.match(r'^(\d{1,2})[:\.\'\*,;](\d{2})$', text)
    if m:
        mins, secs = int(m.group(1)), int(m.group(2))
        if 0 <= secs <= 59:
            return (mins, secs)

    # 5-digit pattern: MM[colon-as-digit]SS (e.g., "13411" = 13:11)
    m = re.match(r'^(\d{2})\d(\d{2})$', text)
    if m:
        mins, secs = int(m.group(1)), int(m.group(2))
        if 0 <= secs <= 59:
            return (mins, secs)

    return None


def _print_kickoff(offset_sec, label):
    m, s = divmod(int(offset_sec), 60)
    print(f"  [kickoff] {label} kickoff: {offset_sec:.0f}s ({m}:{s:02d})")


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "match.mp4"
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else None
    config = None
    if cfg_path:
        with open(cfg_path) as f:
            config = json.load(f)

    h1, h2 = detect_kickoff(path, config=config)
    print(f"\n1st half kickoff: {h1:.1f}s")
    print(f"2nd half kickoff: {h2:.1f}s")
