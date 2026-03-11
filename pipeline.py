"""
pipeline.py — End-to-end soccer highlight generator

Usage:
    python pipeline.py --input match.mp4
    python pipeline.py --input match.mp4 --output highlights.mp4 --crop smart --grade cinematic
    python pipeline.py --input match.mp4 --strategy audio      # audio-only (fastest)
    python pipeline.py --input match.mp4 --strategy combined   # audio + CV (best quality)
    python pipeline.py --input match.mp4 --mode goals --config match_config.json

Requirements:
    pip install -r requirements.txt
    ffmpeg must be installed (brew install ffmpeg)
"""

import argparse
import os
import shutil
import subprocess

from highlight_audio import detect_highlights_audio

# Auto-detect best available model
_BASE = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_MODEL = os.path.join(_BASE, "models", "soccer_yolov8s.pt")
_SOCCANA_MODEL = os.path.join(_BASE, "models", "soccana_yolov11n.pt")


def _default_model():
    """Return the best available model path, or None for auto-detect."""
    for p in [_CUSTOM_MODEL, _SOCCANA_MODEL]:
        if os.path.isfile(p):
            return p
    return None


GRADE_FILTERS = {
    "cinematic": "eq=contrast=1.1:brightness=0.02:saturation=1.2",
    "dramatic": "eq=contrast=1.3:saturation=0.7",
    "vibrant": "eq=contrast=1.05:saturation=1.5",
    "none": "null",
}


def run_pipeline(
    input_path,
    output_path="highlights_9x16.mp4",
    strategy="audio",
    top_percent=10,
    crop_mode="center",
    color_grade="cinematic",
    max_duration_min=3,
    model_path=None,
    use_sahi=False,
):
    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}")
        return

    tmp_dir = "/tmp/highlight_clips"
    os.makedirs(tmp_dir, exist_ok=True)

    # ── 1. Detect highlights ───────────────────────────────────────────
    print(f"[1/5] Detecting highlights (strategy={strategy})...")

    audio_events = detect_highlights_audio(input_path, top_percent=top_percent)
    print(f"       Audio found {len(audio_events)} events")

    if strategy == "combined":
        from highlight_cv import detect_events_yolo
        from highlight_combined import combine_signals

        cv_events = detect_events_yolo(input_path, model_path=model_path,
                                       use_sahi=use_sahi)

        # Get video duration
        import cv2
        cap = cv2.VideoCapture(input_path)
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        combined = combine_signals(audio_events, cv_events,
                                   video_duration_sec=duration)
        # combined returns (start, end, score) — use as our event list
        events = [(s, e) for s, e, _ in combined]
        print(f"       Combined scoring produced {len(events)} highlight windows")
    else:
        events = audio_events

    if not events:
        print("No highlights detected. Try lowering --top-percent.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # ── 2. Trim to max duration ────────────────────────────────────────
    max_dur = max_duration_min * 60
    total_dur = sum(e - s for s, e in events)
    if total_dur > max_dur:
        trimmed = []
        running = 0
        for s, e in events:
            dur = e - s
            if running + dur <= max_dur:
                trimmed.append((s, e))
                running += dur
        events = trimmed
        print(f"       Trimmed to {len(events)} events ({running:.0f}s total)")
    else:
        print(f"       Total duration: {total_dur:.0f}s")

    # ── 3. Cut individual clips with ffmpeg ────────────────────────────
    print(f"[2/5] Cutting {len(events)} clips...")
    clip_paths = []
    for i, (start, end) in enumerate(events):
        out = os.path.join(tmp_dir, f"clip_{i:03d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-i", input_path,
            "-t", str(end - start),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            out,
        ]
        subprocess.run(cmd, capture_output=True)
        clip_paths.append(out)

    # ── 4. Concatenate clips ───────────────────────────────────────────
    print(f"[3/5] Concatenating {len(clip_paths)} clips...")
    concat_path = os.path.join(tmp_dir, "concat.mp4")
    list_file = os.path.join(tmp_dir, "list.txt")
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", list_file, "-c", "copy", concat_path],
        capture_output=True,
    )

    # ── 5. Crop to 9:16 ───────────────────────────────────────────────
    print(f"[4/5] Cropping to 9:16 ({crop_mode})...")
    cropped_path = os.path.join(tmp_dir, "cropped.mp4")

    if crop_mode == "smart":
        from smart_crop import smart_crop_video
        smart_crop_video(concat_path, cropped_path, model_path=model_path,
                         use_sahi=use_sahi)
    else:
        # Center crop
        vf = "crop=in_h*9/16:in_h,scale=1080:1920"
        subprocess.run(
            ["ffmpeg", "-y", "-i", concat_path,
             "-vf", vf,
             "-c:v", "libx264", "-crf", "20", "-preset", "medium",
             "-c:a", "aac", "-b:a", "192k",
             cropped_path],
            capture_output=True,
        )

    # ── 6. Color grade ─────────────────────────────────────────────────
    print(f"[5/5] Applying color grade ({color_grade})...")
    vf = GRADE_FILTERS.get(color_grade, "null")

    subprocess.run(
        ["ffmpeg", "-y", "-i", cropped_path,
         "-vf", vf,
         "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         "-c:a", "copy",
         output_path],
        capture_output=True,
    )

    # ── Cleanup ────────────────────────────────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if os.path.isfile(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nDone! Output: {output_path}")
        print(f"File size: {size_mb:.1f} MB")
    else:
        print("\nError: output file was not created. Check ffmpeg logs.")


def run_goals_pipeline(
    input_path,
    config_path,
    crop_mode="center",
    color_grade="cinematic",
    output_dir=None,
    model_path=None,
    use_sahi=False,
):
    """
    Goal-focused pipeline: extracts each goal as a separate clip
    with score overlay, 9:16 crop, and color grading.
    """
    from kickoff_detector import detect_kickoff
    from goal_detector import load_match_config, get_all_goal_windows
    from score_overlay import build_score_filter

    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}")
        return
    if not os.path.isfile(config_path):
        print(f"Error: config file not found: {config_path}")
        return

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path)) or "."

    tmp_dir = "/tmp/goal_clips"
    os.makedirs(tmp_dir, exist_ok=True)

    # ── 1. Load config & detect kickoff times ───────────────────────
    config = load_match_config(config_path)
    match_info = config["match"]

    print("[1/4] Detecting kickoff times (broadcast clock OCR)...")
    kickoff_1st, kickoff_2nd = detect_kickoff(input_path, config=config)

    # ── 2. Compute goal windows ─────────────────────────────────────
    print("[2/4] Computing goal windows from match config...")
    goal_windows = get_all_goal_windows(
        input_path, config, kickoff_1st, kickoff_2nd
    )

    print(f"       Found {len(goal_windows)} goals to extract")

    # ── 3. Process each goal ─────────────────────────────────────────
    print(f"[3/4] Extracting and processing {len(goal_windows)} goal clips...")
    output_files = []

    for gw in goal_windows:
        goal_num = gw["goal_num"]
        scorer = gw["scorer"].lower()
        score_after = gw["score_after"]
        clip_start = gw["clip_start"]
        clip_end = gw["clip_end"]
        clip_duration = clip_end - clip_start

        filename = f"goal_{goal_num}_{score_after}_{scorer}.mp4"
        final_output = os.path.join(output_dir, filename)

        print(f"\n  --- Goal {goal_num}: {gw['scorer']} ({gw['minute']}') "
              f"{gw['score_before']} -> {score_after} ---")

        # Step A: Cut raw clip from source video
        raw_clip = os.path.join(tmp_dir, f"raw_{goal_num}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip_start), "-i", input_path,
            "-t", str(clip_duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            raw_clip,
        ]
        subprocess.run(cmd, capture_output=True)
        print(f"    Cut clip: {clip_start:.0f}s - {clip_end:.0f}s "
              f"({clip_duration:.0f}s)")

        # Step B: Crop to 9:16
        cropped_clip = os.path.join(tmp_dir, f"cropped_{goal_num}.mp4")
        if crop_mode == "smart":
            from smart_crop import smart_crop_video
            smart_crop_video(raw_clip, cropped_clip, model_path=model_path,
                             use_sahi=use_sahi)
        else:
            vf = "crop=in_h*9/16:in_h,scale=1080:1920"
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_clip,
                 "-vf", vf,
                 "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                 "-c:a", "aac", "-b:a", "192k",
                 cropped_clip],
                capture_output=True,
            )
        print(f"    Cropped to 9:16")

        # Step C: Apply score overlay + color grade in one pass
        gw_with_match = {
            **gw,
            "home": match_info["home"],
            "away": match_info["away"],
        }
        score_filter = build_score_filter(gw_with_match, clip_duration)

        grade_vf = GRADE_FILTERS.get(color_grade, "null")
        if grade_vf != "null":
            full_vf = f"{score_filter},{grade_vf}"
        else:
            full_vf = score_filter

        subprocess.run(
            ["ffmpeg", "-y", "-i", cropped_clip,
             "-vf", full_vf,
             "-c:v", "libx264", "-crf", "18", "-preset", "medium",
             "-c:a", "copy",
             final_output],
            capture_output=True,
        )

        if os.path.isfile(final_output):
            size_mb = os.path.getsize(final_output) / (1024 * 1024)
            print(f"    Saved: {filename} ({size_mb:.1f} MB)")
            output_files.append(final_output)
        else:
            print(f"    ERROR: Failed to create {filename}")

    # ── 4. Summary ───────────────────────────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n[4/4] Done! Created {len(output_files)} goal clips:")
    for f in output_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {os.path.basename(f):40s} {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Soccer Highlight Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python pipeline.py --input match.mp4
  python pipeline.py --input match.mp4 --strategy combined --crop smart
  python pipeline.py --input match.mp4 --top-percent 15 --max-duration 5
  python pipeline.py --input match.mp4 --mode goals --config match_config.json
        """,
    )
    parser.add_argument("--input", required=True, help="Path to full match .mp4")
    parser.add_argument("--output", default="highlights_9x16.mp4",
                        help="Output file path (default: highlights_9x16.mp4)")
    parser.add_argument("--mode", choices=["highlights", "goals"], default="highlights",
                        help="Mode: highlights (generic reel) or goals (individual goal clips)")
    parser.add_argument("--config", default=None,
                        help="Path to match_config.json (required for --mode goals)")
    parser.add_argument("--strategy", choices=["audio", "combined"], default="audio",
                        help="Detection strategy: audio (fast) or combined (audio+CV)")
    parser.add_argument("--top-percent", type=int, default=10,
                        help="Top N%% loudest moments to keep (default: 10)")
    parser.add_argument("--crop", choices=["center", "smart"], default="center",
                        help="Crop mode: center (fast) or smart (YOLO-tracked)")
    parser.add_argument("--grade", default="cinematic",
                        choices=["cinematic", "dramatic", "vibrant", "none"],
                        help="Color grading style (default: cinematic)")
    parser.add_argument("--max-duration", type=float, default=3,
                        help="Max highlight duration in minutes (default: 3)")
    parser.add_argument("--model", default=None,
                        help="Path to YOLO model weights "
                        "(default: auto-detect models/soccer_yolov8s.pt)")
    parser.add_argument("--sahi", action="store_true",
                        help="Enable SAHI sliced inference for better ball detection "
                        "(slower but more accurate)")

    args = parser.parse_args()

    model = args.model or _default_model()

    if args.mode == "goals":
        if not args.config:
            parser.error("--config is required when using --mode goals")
        run_goals_pipeline(
            args.input,
            args.config,
            crop_mode=args.crop,
            color_grade=args.grade,
            model_path=model,
            use_sahi=args.sahi,
        )
    else:
        run_pipeline(
            args.input,
            args.output,
            strategy=args.strategy,
            top_percent=args.top_percent,
            crop_mode=args.crop,
            color_grade=args.grade,
            max_duration_min=args.max_duration,
            model_path=model,
            use_sahi=args.sahi,
        )
