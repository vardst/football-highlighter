#!/usr/bin/env python3
"""
fh.py — Football Highlighter CLI

Unified entry point for stream-first football highlight detection.

Usage:
    fh.py browse                          # Browse IPTV sports streams
    fh.py browse --search "bein"          # Search streams
    fh.py browse --auto-watch             # Select → immediately watch

    fh.py watch <url>                     # Live highlight detection
    fh.py watch <url> --strategy combined # With CV detection

    fh.py record <url> -o match.mp4       # Record stream to file
    fh.py record <url> -d 7200            # Record for 2 hours

    fh.py highlights match.mp4            # Extract highlights from file
    fh.py goals match.mp4 -c config.json  # Extract goal clips

"""

import argparse
import os
import signal
import sys


# Auto-detect best available model
_BASE = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_MODEL = os.path.join(_BASE, "models", "soccer_yolov8s.pt")
_SOCCANA_MODEL = os.path.join(_BASE, "models", "soccana_yolov11n.pt")


def _default_model():
    for p in [_CUSTOM_MODEL, _SOCCANA_MODEL]:
        if os.path.isfile(p):
            return p
    return None


def _add_detection_args(parser):
    """Add shared detection arguments to a subparser."""
    parser.add_argument(
        "--strategy", choices=["audio", "combined"], default="audio",
        help="Detection strategy (default: audio)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to YOLO model weights (default: auto-detect)",
    )
    parser.add_argument(
        "--sahi", action="store_true",
        help="Enable SAHI sliced inference",
    )


def _add_watch_args(parser):
    """Add shared watch/live arguments to a subparser."""
    parser.add_argument(
        "--segment-duration", type=int, default=30,
        help="Segment duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--clip-pre", type=int, default=15,
        help="Seconds before highlight to include (default: 15)",
    )
    parser.add_argument(
        "--clip-post", type=int, default=90,
        help="Seconds after highlight to include (default: 90)",
    )
    parser.add_argument(
        "--output-dir", default="live_highlights",
        help="Output directory for highlight clips (default: live_highlights)",
    )
    parser.add_argument(
        "--aspect", choices=["9:16", "1:1"], default="9:16",
        help="Output aspect ratio (default: 9:16)",
    )


# ── Subcommand handlers ──────────────────────────────────────────────


def cmd_browse(args):
    """Browse IPTV sports streams."""
    from stream_discovery import fetch_all_streams
    from stream_browser import run_browser

    print("Fetching IPTV stream lists...")
    streams = fetch_all_streams()
    print(f"Found {len(streams)} sports streams")

    if not streams:
        print("No streams found. Check your internet connection.")
        return

    selected_url = run_browser(streams, keyword=args.search)

    if selected_url:
        if args.auto_watch:
            # Build a fake args namespace for cmd_watch
            watch_args = argparse.Namespace(
                url=selected_url,
                strategy=getattr(args, "strategy", "audio"),
                model=getattr(args, "model", None),
                sahi=getattr(args, "sahi", False),
                segment_duration=30,
                clip_pre=15,
                clip_post=90,
                output_dir="live_highlights",
                aspect="9:16",
            )
            cmd_watch(watch_args)
        else:
            print(f"\nSelected: {selected_url}")
            print(f"Start watching with:\n  python fh.py watch \"{selected_url}\"")


def cmd_watch(args):
    """Live highlight detection on a stream."""
    from stream_capture import segment_stream, normalize_stream_url
    from live_detector import LiveDetector
    from live_monitor import LiveMonitor

    url = normalize_stream_url(args.url)
    model = args.model or _default_model()

    detector = LiveDetector(
        strategy=args.strategy,
        model_path=model,
        use_sahi=args.sahi,
        clip_pre_sec=args.clip_pre,
        clip_post_sec=args.clip_post,
        output_dir=args.output_dir,
        aspect=args.aspect,
        verbose=False,
    )

    monitor = LiveMonitor()
    monitor.update(strategy=args.strategy)

    def on_segment(segment_path, segment_index):
        """Segment callback: detect + update dashboard."""
        metrics = detector.on_segment_ready(segment_path, segment_index)
        if metrics:
            monitor.update(
                peak_energy=metrics["peak_energy"],
                threshold=metrics["threshold"],
                rolling_mean=metrics["rolling_mean"],
                rolling_var=metrics["rolling_var"],
                is_highlight=metrics["is_highlight"],
                segments_processed=detector.segments_processed,
                warmup_remaining=metrics["warmup_remaining"],
                highlight_info=metrics["highlight_info"],
            )

    def on_status(status):
        """Stream status callback."""
        monitor.update(stream_status=status)

    # Graceful shutdown on Ctrl+C
    def sigint_handler(sig, frame):
        monitor.stop()
        print("\nStopping...")
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    monitor.start()

    try:
        segment_stream(
            url,
            segment_dir="/tmp/live_segments",
            segment_duration=args.segment_duration,
            on_segment_ready=on_segment,
            on_status=on_status,
        )
    finally:
        monitor.stop()


def cmd_record(args):
    """Record a stream to file."""
    from stream_capture import record_stream

    recorded = record_stream(args.url, args.output, args.duration)
    print(f"\nProcess with:\n  python fh.py highlights {recorded}")


def cmd_highlights(args):
    """Extract highlights from a local video file."""
    from pipeline import run_pipeline

    model = args.model or _default_model()
    run_pipeline(
        args.input,
        output_path=args.output,
        strategy=args.strategy,
        top_percent=args.top_percent,
        crop_mode=args.crop,
        color_grade=args.grade,
        max_duration_min=args.max_duration,
        model_path=model,
        use_sahi=args.sahi,
        aspect=args.aspect,
    )


def cmd_goals(args):
    """Extract individual goal clips with score overlay."""
    from pipeline import run_goals_pipeline

    model = args.model or _default_model()
    run_goals_pipeline(
        args.input,
        args.config,
        crop_mode=args.crop,
        color_grade=args.grade,
        output_dir=args.output_dir,
        model_path=model,
        use_sahi=args.sahi,
        aspect=args.aspect,
    )


# ── Argument parser ───────────────────────────────────────────────────


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fh",
        description="Football Highlighter — stream-first highlight detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── browse ──
    p_browse = sub.add_parser("browse", help="Browse IPTV sports streams")
    p_browse.add_argument("--search", "-s", default=None,
                          help="Initial search keyword")
    p_browse.add_argument("--auto-watch", "-w", action="store_true",
                          help="Immediately start watching after selection")
    _add_detection_args(p_browse)

    # ── watch ──
    p_watch = sub.add_parser("watch", help="Live highlight detection on a stream")
    p_watch.add_argument("url", help="Stream URL (http, acestream://, etc.)")
    _add_detection_args(p_watch)
    _add_watch_args(p_watch)

    # ── record ──
    p_record = sub.add_parser("record", help="Record a stream to file")
    p_record.add_argument("url", help="Stream URL")
    p_record.add_argument("-o", "--output", default="recording.mp4",
                          help="Output file path (default: recording.mp4)")
    p_record.add_argument("-d", "--duration", type=int, default=None,
                          help="Max recording duration in seconds")

    # ── highlights ──
    p_hl = sub.add_parser("highlights", help="Extract highlights from a video file")
    p_hl.add_argument("input", help="Path to video file")
    p_hl.add_argument("-o", "--output", default="highlights_9x16.mp4",
                      help="Output file path")
    _add_detection_args(p_hl)
    p_hl.add_argument("--top-percent", type=int, default=10,
                      help="Top N%% loudest moments (default: 10)")
    p_hl.add_argument("--crop", choices=["center", "smart"], default="center",
                      help="Crop mode (default: center)")
    p_hl.add_argument("--grade", default="cinematic",
                      choices=["cinematic", "dramatic", "vibrant", "none"],
                      help="Color grade (default: cinematic)")
    p_hl.add_argument("--max-duration", type=float, default=3,
                      help="Max highlight duration in minutes (default: 3)")
    p_hl.add_argument("--aspect", choices=["9:16", "1:1"], default="9:16",
                      help="Output aspect ratio (default: 9:16)")

    # ── goals ──
    p_goals = sub.add_parser("goals", help="Extract goal clips with score overlay")
    p_goals.add_argument("input", help="Path to video file")
    p_goals.add_argument("-c", "--config", required=True,
                         help="Path to match_config.json")
    _add_detection_args(p_goals)
    p_goals.add_argument("--crop", choices=["center", "smart"], default="center",
                         help="Crop mode (default: center)")
    p_goals.add_argument("--grade", default="cinematic",
                         choices=["cinematic", "dramatic", "vibrant", "none"],
                         help="Color grade (default: cinematic)")
    p_goals.add_argument("--output-dir", default=None,
                         help="Output directory for goal clips")
    p_goals.add_argument("--aspect", choices=["9:16", "1:1"], default="9:16",
                         help="Output aspect ratio (default: 9:16)")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "browse": cmd_browse,
        "watch": cmd_watch,
        "record": cmd_record,
        "highlights": cmd_highlights,
        "goals": cmd_goals,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
