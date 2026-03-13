#!/usr/bin/env python3
"""
dashboard.py — Web dashboard for Football Highlighter

Flask backend serving a browser UI for stream browsing, recording,
live highlight detection, and clip downloading.

Usage:
    python dashboard.py                     # http://localhost:5555
    python dashboard.py --port 8080         # custom port
    python dashboard.py --host 0.0.0.0      # expose on LAN
"""

import atexit
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict
from datetime import datetime

from flask import Flask, jsonify, request, send_file, send_from_directory

from stream_discovery import fetch_all_streams, probe_streams_batch
from stream_capture import normalize_stream_url
from live_detector import LiveDetector

# ── Auto-detect model (same logic as fh.py) ──────────────────────────

_BASE = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_MODEL = os.path.join(_BASE, "models", "soccer_yolov8s.pt")
_SOCCANA_MODEL = os.path.join(_BASE, "models", "soccana_yolov11n.pt")


def _default_model():
    for p in [_CUSTOM_MODEL, _SOCCANA_MODEL]:
        if os.path.isfile(p):
            return p
    return None


# ── Global State ──────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.streams = []
        self.streams_loading = False
        self.probing = False

        # Recording state
        self.recording = None  # {url, output_path, proc, thread, start_time, status, filename}

        # Watch/live detection state
        self.watching = None  # {url, proc, thread, start_time, status, strategy, stop_event, detector, metrics}

        # File lists
        self.highlights = []  # [{name, path, size_mb, timestamp}]
        self.recordings = []  # [{name, path, size_mb, timestamp}]


state = AppState()

# ── Flask App ─────────────────────────────────────────────────────────

app = Flask(__name__)

RECORDINGS_DIR = os.path.join(_BASE, "recordings")
HIGHLIGHTS_DIR = os.path.join(_BASE, "live_highlights")
SEGMENT_DIR = "/tmp/live_segments_dashboard"


@app.route("/")
def index():
    return send_from_directory(_BASE, "dashboard.html")


# ── Streams ───────────────────────────────────────────────────────────

@app.route("/api/streams")
def api_streams():
    q = request.args.get("q", "").strip().lower()
    refresh = request.args.get("refresh", "").lower() == "true"

    if not state.streams or refresh:
        if not state.streams_loading:
            state.streams_loading = True
            try:
                state.streams = fetch_all_streams()
            finally:
                state.streams_loading = False

    streams = state.streams
    if q:
        streams = [
            s for s in streams
            if q in s.name.lower() or q in s.group.lower() or q in s.tvg_country.lower()
        ]

    # Cap at 200
    capped = streams[:200]
    return jsonify({
        "streams": [
            {
                "name": s.name,
                "url": s.url,
                "group": s.group,
                "country": s.tvg_country,
                "quality": s.quality,
                "status": s.status,
            }
            for s in capped
        ],
        "total": len(streams),
        "shown": len(capped),
        "loading": state.streams_loading,
    })


@app.route("/api/streams/probe", methods=["POST"])
def api_probe():
    if state.probing:
        return jsonify({"error": "Probe already in progress"}), 409

    def _probe_bg():
        state.probing = True
        try:
            probe_streams_batch(state.streams, max_workers=20, timeout=5)
        finally:
            state.probing = False

    threading.Thread(target=_probe_bg, daemon=True).start()
    return jsonify({"status": "probing", "count": len(state.streams)})


# ── Recording ─────────────────────────────────────────────────────────

def _recording_thread(url, output_path):
    """Background recording via ffmpeg."""
    norm_url = normalize_stream_url(url)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "30",
        "-i", norm_url,
        "-c", "copy",
        output_path,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with state.lock:
        if state.recording:
            state.recording["proc"] = proc
            state.recording["status"] = "recording"

    proc.wait()

    with state.lock:
        if state.recording:
            state.recording["status"] = "stopped"
            state.recording["proc"] = None

    # Add to recordings list
    if os.path.isfile(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        name = os.path.basename(output_path)
        with state.lock:
            state.recordings.append({
                "name": name,
                "path": output_path,
                "size_mb": round(size_mb, 1),
                "timestamp": datetime.now().isoformat(),
            })


@app.route("/api/record/start", methods=["POST"])
def api_record_start():
    with state.lock:
        if state.recording and state.recording["status"] == "recording":
            return jsonify({"error": "Already recording"}), 409

    data = request.json or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL required"}), 400

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    filename = data.get("filename") or f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = os.path.join(RECORDINGS_DIR, filename)

    rec = {
        "url": url,
        "output_path": output_path,
        "proc": None,
        "thread": None,
        "start_time": time.time(),
        "status": "starting",
        "filename": filename,
    }

    t = threading.Thread(target=_recording_thread, args=(url, output_path), daemon=True)
    rec["thread"] = t

    with state.lock:
        state.recording = rec

    t.start()
    return jsonify({"status": "started", "filename": filename})


@app.route("/api/record/stop", methods=["POST"])
def api_record_stop():
    with state.lock:
        rec = state.recording
        if not rec or rec["status"] != "recording":
            return jsonify({"error": "Not recording"}), 409
        proc = rec.get("proc")

    if proc and proc.poll() is None:
        try:
            proc.stdin.write(b"q")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            proc.terminate()

    return jsonify({"status": "stopping"})


# ── Watch / Live Detection ────────────────────────────────────────────

def _segment_monitor(segment_dir, stop_event, detector):
    """Watch for new segments and run detection."""
    last_index = -1
    max_age_sec = 600

    while not stop_event.is_set():
        try:
            files = sorted(
                f for f in os.listdir(segment_dir)
                if f.startswith("seg_") and f.endswith(".mp4")
            )
        except OSError:
            time.sleep(1)
            continue

        if len(files) >= 2:
            for seg_file in files[:-1]:
                try:
                    idx = int(seg_file.replace("seg_", "").replace(".mp4", ""))
                except ValueError:
                    continue

                if idx > last_index:
                    last_index = idx
                    seg_path = os.path.join(segment_dir, seg_file)
                    try:
                        metrics = detector.on_segment_ready(seg_path, idx)
                        with state.lock:
                            if state.watching:
                                state.watching["metrics"] = metrics
                                state.watching["segments_processed"] = detector.segments_processed

                                # Check if a highlight was extracted
                                if metrics and metrics.get("highlight_info"):
                                    info = metrics["highlight_info"]
                                    state.highlights.append({
                                        "name": info["clip_name"],
                                        "path": info["clip_path"],
                                        "size_mb": round(info["size_mb"], 1),
                                        "timestamp": datetime.now().isoformat(),
                                    })
                    except Exception as e:
                        print(f"[dashboard] Detection error on seg {idx}: {e}")

        # Cleanup old segments
        now = time.time()
        try:
            for seg_file in files[:-2]:
                seg_path = os.path.join(segment_dir, seg_file)
                if now - os.path.getmtime(seg_path) > max_age_sec:
                    os.remove(seg_path)
        except (OSError, UnboundLocalError):
            pass

        time.sleep(2)


def _watch_thread(url, strategy, stop_event, detector):
    """Background ffmpeg segmentation + monitoring."""
    norm_url = normalize_stream_url(url)

    # Clean and recreate segment dir
    if os.path.isdir(SEGMENT_DIR):
        shutil.rmtree(SEGMENT_DIR, ignore_errors=True)
    os.makedirs(SEGMENT_DIR, exist_ok=True)

    segment_pattern = os.path.join(SEGMENT_DIR, "seg_%04d.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "30",
        "-i", norm_url,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", "30",
        "-segment_format", "mp4",
        "-reset_timestamps", "1",
        segment_pattern,
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with state.lock:
        if state.watching:
            state.watching["proc"] = proc
            state.watching["status"] = "connected"

    # Start segment monitor
    monitor_thread = threading.Thread(
        target=_segment_monitor,
        args=(SEGMENT_DIR, stop_event, detector),
        daemon=True,
    )
    monitor_thread.start()

    proc.wait()

    stop_event.set()
    monitor_thread.join(timeout=5)

    with state.lock:
        if state.watching:
            state.watching["status"] = "stopped"
            state.watching["proc"] = None


@app.route("/api/watch/start", methods=["POST"])
def api_watch_start():
    with state.lock:
        if state.watching and state.watching["status"] in ("connecting", "connected"):
            return jsonify({"error": "Already watching"}), 409

    data = request.json or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL required"}), 400

    strategy = data.get("strategy", "audio")
    model = _default_model()

    os.makedirs(HIGHLIGHTS_DIR, exist_ok=True)

    detector = LiveDetector(
        strategy=strategy,
        model_path=model,
        output_dir=HIGHLIGHTS_DIR,
        verbose=True,
    )

    stop_event = threading.Event()

    watch = {
        "url": url,
        "proc": None,
        "thread": None,
        "start_time": time.time(),
        "status": "connecting",
        "strategy": strategy,
        "stop_event": stop_event,
        "detector": detector,
        "metrics": None,
        "segments_processed": 0,
    }

    t = threading.Thread(target=_watch_thread, args=(url, strategy, stop_event, detector), daemon=True)
    watch["thread"] = t

    with state.lock:
        state.watching = watch

    t.start()
    return jsonify({"status": "started", "strategy": strategy})


@app.route("/api/watch/stop", methods=["POST"])
def api_watch_stop():
    with state.lock:
        watch = state.watching
        if not watch or watch["status"] not in ("connecting", "connected"):
            return jsonify({"error": "Not watching"}), 409
        proc = watch.get("proc")
        stop_event = watch.get("stop_event")

    if stop_event:
        stop_event.set()

    if proc and proc.poll() is None:
        try:
            proc.stdin.write(b"q")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            proc.terminate()

    return jsonify({"status": "stopping"})


# ── Status (unified polling endpoint) ─────────────────────────────────

@app.route("/api/status")
def api_status():
    with state.lock:
        rec_info = None
        if state.recording:
            r = state.recording
            elapsed = time.time() - r["start_time"] if r["start_time"] else 0
            rec_info = {
                "url": r["url"],
                "filename": r["filename"],
                "status": r["status"],
                "elapsed": round(elapsed),
            }

        watch_info = None
        if state.watching:
            w = state.watching
            elapsed = time.time() - w["start_time"] if w["start_time"] else 0
            metrics = w.get("metrics") or {}
            watch_info = {
                "url": w["url"],
                "status": w["status"],
                "strategy": w["strategy"],
                "elapsed": round(elapsed),
                "segments_processed": w.get("segments_processed", 0),
                "peak_energy": metrics.get("peak_energy", 0),
                "threshold": metrics.get("threshold", 0),
                "rolling_mean": metrics.get("rolling_mean", 0),
                "warmup_remaining": metrics.get("warmup_remaining", 0),
                "is_highlight": metrics.get("is_highlight", False),
            }

        return jsonify({
            "recording": rec_info,
            "watching": watch_info,
            "highlights": list(state.highlights),
            "recordings": list(state.recordings),
            "probing": state.probing,
        })


# ── File serving ──────────────────────────────────────────────────────

@app.route("/api/highlights/<name>")
def api_highlight_download(name):
    path = os.path.join(HIGHLIGHTS_DIR, name)
    if not os.path.isfile(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, as_attachment=True)


@app.route("/api/recordings/<name>")
def api_recording_download(name):
    path = os.path.join(RECORDINGS_DIR, name)
    if not os.path.isfile(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, as_attachment=True)


# ── Scan existing files on startup ────────────────────────────────────

def _scan_existing_files():
    """Populate highlights and recordings lists from existing files on disk."""
    for directory, target_list in [(HIGHLIGHTS_DIR, state.highlights), (RECORDINGS_DIR, state.recordings)]:
        if os.path.isdir(directory):
            for f in sorted(os.listdir(directory)):
                if f.endswith(".mp4"):
                    fpath = os.path.join(directory, f)
                    size_mb = os.path.getsize(fpath) / (1024 * 1024)
                    target_list.append({
                        "name": f,
                        "path": fpath,
                        "size_mb": round(size_mb, 1),
                        "timestamp": datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat(),
                    })


# ── Cleanup ───────────────────────────────────────────────────────────

def _cleanup():
    """Terminate any running ffmpeg processes on exit."""
    if state.recording and state.recording.get("proc"):
        proc = state.recording["proc"]
        if proc.poll() is None:
            proc.terminate()
    if state.watching and state.watching.get("proc"):
        proc = state.watching["proc"]
        if proc.poll() is None:
            proc.terminate()
        stop = state.watching.get("stop_event")
        if stop:
            stop.set()


atexit.register(_cleanup)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Football Highlighter Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5555, help="Port (default: 5555)")
    args = parser.parse_args()

    _scan_existing_files()

    print(f"Football Highlighter Dashboard: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
