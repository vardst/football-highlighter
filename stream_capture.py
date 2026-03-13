"""
stream_capture.py — Stream recording and segmentation via ffmpeg

Handles HTTP/HLS/RTMP stream URLs (e.g., Ace Stream, IPTV, m3u8).
Two modes:
  - record_stream(): record full stream to a single MP4 file
  - segment_stream(): chop stream into 30s segments for live processing
"""

import os
import subprocess
import threading
import time


def is_stream_url(path):
    """Return True if path looks like a stream URL rather than a local file."""
    lower = path.lower()
    return (
        lower.startswith("http://")
        or lower.startswith("https://")
        or lower.startswith("rtmp://")
        or lower.startswith("rtsp://")
        or lower.startswith("acestream://")
        or ".m3u8" in lower
    )


def normalize_stream_url(url):
    """
    Normalize stream URLs — converts acestream:// to local HTTP API URL.

    Args:
        url: raw stream URL (may be acestream://, http://, etc.)

    Returns:
        normalized HTTP URL ready for ffmpeg
    """
    if url.startswith("acestream://"):
        content_id = url.replace("acestream://", "")
        return f"http://127.0.0.1:6878/ace/getstream?id={content_id}"
    return url


def record_stream(url, output_path, max_duration_sec=None):
    """
    Record a live stream to a single MP4 file.

    Blocks until the stream ends, max_duration is reached, or Ctrl+C.
    On KeyboardInterrupt, sends 'q' to ffmpeg for clean shutdown.

    Args:
        url: stream URL (HTTP/HLS/RTMP/acestream)
        output_path: output .mp4 path
        max_duration_sec: optional max recording duration in seconds

    Returns:
        path to the recorded file
    """
    url = normalize_stream_url(url)
    cmd = [
        "ffmpeg", "-y",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "30",
        "-i", url,
        "-c", "copy",
    ]
    if max_duration_sec:
        cmd += ["-t", str(max_duration_sec)]
    cmd.append(output_path)

    print(f"[record] Recording stream to {output_path}")
    print(f"[record] Press Ctrl+C to stop recording")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[record] Stopping recording (clean shutdown)...")
        proc.stdin.write(b"q")
        proc.stdin.flush()
        proc.wait(timeout=10)

    if os.path.isfile(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[record] Saved: {output_path} ({size_mb:.1f} MB)")
    else:
        print("[record] Error: output file was not created")

    return output_path


def segment_stream(url, segment_dir="/tmp/live_segments",
                   segment_duration=30, on_segment_ready=None,
                   on_status=None):
    """
    Segment a live stream into fixed-duration MP4 chunks.

    Runs ffmpeg with the segment muxer. A monitor thread watches for new
    segments and calls on_segment_ready(path, index) when each is complete.

    Args:
        url: stream URL (HTTP/HLS/RTMP/acestream)
        segment_dir: directory for segment files
        segment_duration: seconds per segment (default 30)
        on_segment_ready: callback(segment_path, segment_index)
        on_status: optional callback(status_str) for stream status updates
                   status_str is one of: CONNECTING, CONNECTED, ERROR
    """
    url = normalize_stream_url(url)
    os.makedirs(segment_dir, exist_ok=True)

    segment_pattern = os.path.join(segment_dir, "seg_%04d.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "30",
        "-i", url,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-segment_format", "mp4",
        "-reset_timestamps", "1",
        segment_pattern,
    ]

    stop_event = threading.Event()

    def _monitor():
        """Watch segment_dir for new complete segments."""
        last_index = -1
        max_age_sec = 600  # delete segments older than 10 minutes

        while not stop_event.is_set():
            try:
                files = sorted(f for f in os.listdir(segment_dir)
                               if f.startswith("seg_") and f.endswith(".mp4"))
            except OSError:
                time.sleep(1)
                continue

            # When segment N+1 exists, segment N is complete
            if len(files) >= 2:
                for seg_file in files[:-1]:  # all except the latest (still writing)
                    # Parse index from seg_NNNN.mp4
                    try:
                        idx = int(seg_file.replace("seg_", "").replace(".mp4", ""))
                    except ValueError:
                        continue

                    if idx > last_index:
                        last_index = idx
                        seg_path = os.path.join(segment_dir, seg_file)
                        if on_segment_ready:
                            try:
                                on_segment_ready(seg_path, idx)
                            except Exception as e:
                                print(f"[segment] Callback error on seg {idx}: {e}")

            # Cleanup old segments
            now = time.time()
            for seg_file in files[:-2]:  # keep at least the 2 most recent
                seg_path = os.path.join(segment_dir, seg_file)
                try:
                    if now - os.path.getmtime(seg_path) > max_age_sec:
                        os.remove(seg_path)
                except OSError:
                    pass

            time.sleep(2)

    monitor_thread = threading.Thread(target=_monitor, daemon=True)

    print(f"[segment] Segmenting stream into {segment_duration}s chunks")
    print(f"[segment] Segments directory: {segment_dir}")
    print(f"[segment] Press Ctrl+C to stop")

    if on_status:
        on_status("CONNECTING")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    monitor_thread.start()

    if on_status:
        on_status("CONNECTED")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[segment] Stopping stream capture...")
        proc.stdin.write(b"q")
        proc.stdin.flush()
        proc.wait(timeout=10)
    finally:
        stop_event.set()
        monitor_thread.join(timeout=5)

    # Process any remaining complete segments
    try:
        files = sorted(f for f in os.listdir(segment_dir)
                       if f.startswith("seg_") and f.endswith(".mp4"))
        if files and on_segment_ready:
            for seg_file in files:
                try:
                    idx = int(seg_file.replace("seg_", "").replace(".mp4", ""))
                except ValueError:
                    continue
                seg_path = os.path.join(segment_dir, seg_file)
                # Only process if file is non-trivially small (likely incomplete)
                if os.path.getsize(seg_path) > 10000:
                    on_segment_ready(seg_path, idx)
    except OSError:
        pass

    print("[segment] Stream capture ended")
