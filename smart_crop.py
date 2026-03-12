"""
smart_crop.py — Professional 9:16 vertical reframe for football

Ball-first tracking with player fallback, camera-cut detection,
velocity-clamped smoothing, and per-scene interpolation.
Based on techniques from SmartCrop (MMM 2024), Google AutoFlip,
and AWS Elemental Inference.
"""

import cv2
import numpy as np
import subprocess

from soccer_detector import SoccerDetector


# ── Camera cut detection ─────────────────────────────────────────────────

def _detect_cuts(cap, fps, total_frames, threshold=0.6):
    """
    Detect camera/shot cuts via histogram difference.
    Returns a set of frame indices where cuts occur.
    """
    cuts = set()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    if not ret:
        return cuts

    prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None,
                             [8, 8, 8], [0, 256, 0, 256, 0, 256])
    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()

    # Sample every 2 frames for speed
    step = 2
    for fi in range(step, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        hist = cv2.calcHist([frame], [0, 1, 2], None,
                            [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        diff = 1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
        if diff > threshold:
            cuts.add(fi)

        prev_hist = hist

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cuts


# ── Analysis ─────────────────────────────────────────────────────────────

def analyze_action_positions(video_path, sample_every=2, model_path=None,
                             use_sahi=False):
    """
    Run YOLO on sampled frames. Track the ball primarily, fall back to
    player centroid when ball is not detected.

    Args:
        video_path: path to video file
        sample_every: process every Nth frame
        model_path: path to YOLO .pt weights (None = auto-detect best model)
        use_sahi: enable SAHI sliced inference for better ball detection

    Returns:
        raw_positions: list of (frame_idx, center_x_norm, source)
            source = 'ball' | 'players' | 'interp'
        cuts: set of frame indices where camera cuts occur
        fps, width, height
    """
    detector = SoccerDetector(model_path=model_path, use_sahi=use_sahi)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  [crop] Detecting camera cuts...")
    cuts = _detect_cuts(cap, fps, total_frames)
    print(f"  [crop] Found {len(cuts)} camera cuts")

    print(f"  [crop] Analysing action positions ({total_frames} frames, "
          f"every {sample_every}th)...")

    raw_positions = []  # (frame_idx, x_norm, source)
    last_ball_x = None
    last_ball_frame = -999
    frames_since_ball = 0

    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        # Reset ball tracking at camera cuts
        is_cut = any(abs(frame_idx - c) <= sample_every for c in cuts)
        if is_cut:
            last_ball_x = None
            frames_since_ball = 999

        ball_x, players_x = detector.detect_frame_compat(frame, width)

        if ball_x is not None:
            # Ball detected — use it
            raw_positions.append((frame_idx, ball_x, 'ball'))
            last_ball_x = ball_x
            last_ball_frame = frame_idx
            frames_since_ball = 0
        elif frames_since_ball < 15 and last_ball_x is not None:
            # Ball recently lost — hold last known position
            raw_positions.append((frame_idx, last_ball_x, 'interp'))
            frames_since_ball += 1
        elif players_x:
            # Ball lost — fall back to player centroid
            centroid = float(np.median(players_x))
            raw_positions.append((frame_idx, centroid, 'players'))
            frames_since_ball += 1
        else:
            # Nothing detected — hold center
            raw_positions.append((frame_idx, 0.5, 'interp'))
            frames_since_ball += 1

        frame_idx += 1

    cap.release()

    ball_count = sum(1 for _, _, s in raw_positions if s == 'ball')
    total = len(raw_positions)
    print(f"  [crop] Ball detected in {ball_count}/{total} sampled frames "
          f"({100*ball_count/max(total,1):.0f}%)")

    return raw_positions, cuts, fps, width, height


# ── Smoothing ────────────────────────────────────────────────────────────

def smooth_positions(raw_positions, cuts, total_frames, fps,
                     max_velocity=0.08, alpha=0.12):
    """
    Build a smooth per-frame crop center path.

    1. Interpolate raw detections to every frame
    2. Split into scenes at camera cuts
    3. Smooth each scene independently (EMA forward + backward)
    4. Clamp velocity between frames

    Args:
        raw_positions: from analyze_action_positions
        cuts: set of cut frame indices
        total_frames: total frame count
        fps: video fps
        max_velocity: max crop movement per frame (as fraction of frame width)
        alpha: EMA smoothing factor (higher = more responsive)
    """
    # Step 1: interpolate to every frame
    frame_to_x = {}
    for fi, cx, src in raw_positions:
        frame_to_x[fi] = cx

    all_x = np.full(total_frames, 0.5)
    last_known = 0.5
    for i in range(total_frames):
        if i in frame_to_x:
            last_known = frame_to_x[i]
        all_x[i] = last_known

    # Step 2: identify scene boundaries
    sorted_cuts = sorted(cuts)
    boundaries = [0] + sorted_cuts + [total_frames]

    # Step 3: smooth each scene with bidirectional EMA
    smoothed = np.copy(all_x)
    for s_idx in range(len(boundaries) - 1):
        start = boundaries[s_idx]
        end = boundaries[s_idx + 1]
        if end - start < 2:
            continue

        segment = all_x[start:end].copy()

        # Forward EMA
        fwd = np.copy(segment)
        for i in range(1, len(fwd)):
            fwd[i] = alpha * segment[i] + (1 - alpha) * fwd[i - 1]

        # Backward EMA
        bwd = np.copy(segment)
        for i in range(len(bwd) - 2, -1, -1):
            bwd[i] = alpha * segment[i] + (1 - alpha) * bwd[i + 1]

        # Average forward and backward for centered smoothing
        smoothed[start:end] = (fwd + bwd) / 2.0

    # Step 4: clamp velocity — no more than max_velocity per frame
    for i in range(1, total_frames):
        # Allow instant jump at cuts
        if i in cuts:
            continue
        delta = smoothed[i] - smoothed[i - 1]
        if abs(delta) > max_velocity:
            smoothed[i] = smoothed[i - 1] + np.sign(delta) * max_velocity

    return smoothed.tolist()


# ── Render ───────────────────────────────────────────────────────────────

def render_smart_crop(input_path, output_path, smoothed_centers, aspect="9:16"):
    """
    Render the video with per-frame crop piped through ffmpeg.

    Args:
        aspect: "9:16" for vertical or "1:1" for square
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if aspect == "1:1":
        crop_w = h  # square: crop width = frame height
        out_w, out_h = 1080, 1080
    else:
        crop_w = int(h * 9 / 16)
        out_w, out_h = 1080, 1920

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-i", input_path,         # for audio
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(smoothed_centers):
            center_ratio = smoothed_centers[frame_idx]
        else:
            center_ratio = 0.5

        cx = int(center_ratio * w)
        x_start = max(0, min(cx - crop_w // 2, w - crop_w))

        cropped = frame[:, x_start:x_start + crop_w]
        resized = cv2.resize(cropped, (out_w, out_h))

        proc.stdin.write(resized.tobytes())
        frame_idx += 1

    proc.stdin.close()
    proc.wait()
    cap.release()
    print(f"  [crop] Smart crop rendered -> {output_path}")


# ── Public API ───────────────────────────────────────────────────────────

def smart_crop_video(input_path, output_path, sample_every=2, alpha=0.12,
                     model_path=None, use_sahi=False, aspect="9:16"):
    """
    Full pipeline: detect cuts → analyse ball/player positions →
    smooth with velocity clamping → render.

    Args:
        aspect: "9:16" for vertical or "1:1" for square
    """
    raw_positions, cuts, fps, w, h = analyze_action_positions(
        input_path, sample_every, model_path=model_path, use_sahi=use_sahi
    )
    total_frames = int(cv2.VideoCapture(input_path).get(cv2.CAP_PROP_FRAME_COUNT))
    smoothed = smooth_positions(raw_positions, cuts, total_frames, fps, alpha=alpha)
    render_smart_crop(input_path, output_path, smoothed, aspect=aspect)
