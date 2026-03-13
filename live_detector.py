"""
live_detector.py — Real-time highlight detection for live streams

Processes 30s video segments as they arrive, detects crowd noise spikes
(and optionally CV events), and extracts highlight clips automatically.

Uses adaptive EMA thresholds instead of percentile-based (which needs
the full file). Adapts to stream audio characteristics over time.
"""

import math
import os
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np
from moviepy import VideoFileClip


class LiveDetector:
    """
    Real-time highlight detector for live stream segments.

    Call on_segment_ready(segment_path, segment_index) for each new segment.
    When a highlight is detected, a clip is extracted from surrounding segments.
    """

    def __init__(
        self,
        strategy="audio",
        model_path=None,
        use_sahi=False,
        clip_pre_sec=15,
        clip_post_sec=90,
        output_dir="live_highlights",
        aspect="9:16",
        verbose=True,
    ):
        self.strategy = strategy
        self.model_path = model_path
        self.use_sahi = use_sahi
        self.clip_pre_sec = clip_pre_sec
        self.clip_post_sec = clip_post_sec
        self.output_dir = output_dir
        self.aspect = aspect
        self.verbose = verbose

        # Adaptive threshold state (EMA)
        self.rolling_mean = 0.0
        self.rolling_var = 0.0
        self.alpha = 0.1
        self.warmup_segments = 5
        self.threshold_k = 2.5

        # Tracking
        self.segments_processed = 0
        self.segment_paths = {}  # index -> path
        self.highlight_count = 0
        self.last_highlight_time = -999  # cooldown tracking
        self.cooldown_sec = 90

        # Lazy-loaded detector for combined strategy
        self._detector = None

        os.makedirs(output_dir, exist_ok=True)

    def _get_detector(self):
        if self._detector is None:
            from soccer_detector import SoccerDetector
            self._detector = SoccerDetector(
                model_path=self.model_path, use_sahi=self.use_sahi
            )
        return self._detector

    def on_segment_ready(self, segment_path, segment_index):
        """
        Process a newly completed segment.

        Args:
            segment_path: path to the segment .mp4 file
            segment_index: sequential segment number

        Returns:
            dict with detection metrics:
                peak_energy, threshold, is_highlight, segment_index,
                rolling_mean, rolling_var, warmup_remaining, highlight_info
        """
        self.segment_paths[segment_index] = segment_path

        # Compute audio energy
        audio_peak, audio_mean, audio_var = self._analyze_audio(segment_path)

        # Compute CV score (if combined strategy)
        cv_score = 0.0
        if self.strategy == "combined":
            cv_score = self._analyze_cv(segment_path)

        # Fused score
        peak_score = audio_peak + cv_score

        # Update adaptive threshold
        seg_mean = audio_mean + cv_score
        seg_var = audio_var
        self.rolling_mean = (
            self.alpha * seg_mean + (1 - self.alpha) * self.rolling_mean
        )
        self.rolling_var = (
            self.alpha * seg_var + (1 - self.alpha) * self.rolling_var
        )

        self.segments_processed += 1

        # Compute threshold
        threshold_val = 0.0
        if self.segments_processed < self.warmup_segments:
            is_highlight = False
        else:
            threshold_val = self.rolling_mean + self.threshold_k * math.sqrt(
                max(0, self.rolling_var)
            )
            is_highlight = peak_score > threshold_val

        # Cooldown check
        elapsed = segment_index * 30  # approximate stream time
        if is_highlight and (elapsed - self.last_highlight_time) < self.cooldown_sec:
            is_highlight = False

        warmup_remaining = max(0, self.warmup_segments - self.segments_processed)

        # Log (only if verbose)
        if self.verbose:
            threshold_str = f"{threshold_val:.4f}" if not warmup_remaining else "-- (warming up)"
            tag = " *** HIGHLIGHT DETECTED ***" if is_highlight else ""
            print(
                f"[live] Segment {segment_index} processed: "
                f"peak_energy={peak_score:.4f}, threshold={threshold_str}{tag}"
            )

        # Extract clip if highlight
        highlight_info = None
        if is_highlight:
            self.last_highlight_time = elapsed
            highlight_info = self._extract_clip(segment_index)

        return {
            "peak_energy": peak_score,
            "threshold": threshold_val,
            "is_highlight": is_highlight,
            "segment_index": segment_index,
            "rolling_mean": self.rolling_mean,
            "rolling_var": self.rolling_var,
            "warmup_remaining": warmup_remaining,
            "highlight_info": highlight_info,
        }

    def _analyze_audio(self, segment_path):
        """
        Compute per-second RMS energy for a segment.

        Returns:
            (peak_rms, mean_rms, var_rms)
        """
        try:
            clip = VideoFileClip(segment_path)
            audio = clip.audio
            if audio is None:
                clip.close()
                return 0.0, 0.0, 0.0

            fps = audio.fps
            audio_array = audio.to_soundarray()
            mono = np.mean(audio_array, axis=1)

            samples_per_sec = fps
            n_seconds = len(mono) // samples_per_sec

            if n_seconds == 0:
                clip.close()
                return 0.0, 0.0, 0.0

            energy = np.array([
                np.sqrt(
                    np.mean(
                        mono[i * samples_per_sec : (i + 1) * samples_per_sec] ** 2
                    )
                )
                for i in range(n_seconds)
            ])

            clip.close()

            peak = float(np.max(energy))
            mean = float(np.mean(energy))
            var = float(np.var(energy))
            return peak, mean, var

        except Exception as e:
            print(f"[live] Audio analysis error: {e}")
            return 0.0, 0.0, 0.0

    def _analyze_cv(self, segment_path):
        """
        Run YOLO detection on sampled frames, return a CV event score.

        Samples every 5th frame, checks for ball_near_goal, fast_ball,
        goalkeeper_save, and player_cluster events. Returns a weighted sum.
        """
        detector = self._get_detector()
        cap = cv2.VideoCapture(segment_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        cv_weights = {
            "ball_near_goal": 1.5,
            "fast_ball": 0.8,
            "player_cluster": 2.0,
            "goalkeeper_save": 1.5,
        }

        events = []
        prev_ball_pos = None
        frame_idx = 0
        sample_rate = 5

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            result = detector.detect(frame)
            frame_w = frame.shape[1]
            frame_h = frame.shape[0]

            # Ball position
            ball = None
            if result.ball_x is not None:
                ball = (result.ball_x * frame_w, result.ball_y * frame_h)

            # Ball near goal
            if ball:
                if ball[0] < frame_w * 0.15 or ball[0] > frame_w * 0.85:
                    events.append(("ball_near_goal", 0.7))

                # Fast ball
                if prev_ball_pos:
                    dist = math.sqrt(
                        (ball[0] - prev_ball_pos[0]) ** 2
                        + (ball[1] - prev_ball_pos[1]) ** 2
                    )
                    if dist > frame_w * 0.15:
                        events.append(("fast_ball", 0.6))
                prev_ball_pos = ball

            # Goalkeeper near ball in goal area
            if ball and result.goalkeepers_x:
                ball_in_goal = ball[0] < frame_w * 0.2 or ball[0] > frame_w * 0.8
                if ball_in_goal:
                    for gx in result.goalkeepers_x:
                        gk_px = gx * frame_w
                        if abs(gk_px - ball[0]) < frame_w * 0.1:
                            events.append(("goalkeeper_save", 0.85))
                            break

            # Player clustering
            persons = []
            for px in result.players_x:
                persons.append((px * frame_w, frame_h * 0.5))
            for gx in result.goalkeepers_x:
                persons.append((gx * frame_w, frame_h * 0.5))

            if len(persons) >= 4:
                centroid = np.mean(persons, axis=0)
                distances = [
                    math.sqrt(
                        (p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2
                    )
                    for p in persons
                ]
                if np.mean(distances) < frame_w * 0.08:
                    events.append(("player_cluster", 0.8))

            frame_idx += 1

        cap.release()

        # Compute weighted score
        total_score = 0.0
        for event_type, confidence in events:
            total_score += cv_weights.get(event_type, 0.5) * confidence

        # Normalize by number of sampled frames to get per-segment score
        n_sampled = max(1, frame_idx // sample_rate)
        return total_score / n_sampled

    def _extract_clip(self, trigger_index):
        """
        Extract a highlight clip from segments surrounding the trigger.

        Collects segments covering clip_pre_sec before and clip_post_sec after
        the trigger, concatenates them, and saves as a raw (uncropped) clip.

        Returns:
            dict with clip_name, clip_path, size_mb, segment_index, segments
            or None on failure
        """
        segment_dur = 30  # seconds per segment

        # How many segments before/after to collect
        pre_segments = max(1, self.clip_pre_sec // segment_dur)
        post_segments = max(1, self.clip_post_sec // segment_dur)

        start_idx = max(0, trigger_index - pre_segments)
        end_idx = trigger_index + post_segments

        # Wait for post-trigger segments (up to 3 minutes)
        timeout = 180
        waited = 0
        while waited < timeout:
            available = all(
                idx in self.segment_paths for idx in range(start_idx, end_idx + 1)
            )
            if available:
                break
            time.sleep(2)
            waited += 2

        # Collect available segments
        segments_to_concat = []
        for idx in range(start_idx, end_idx + 1):
            if idx in self.segment_paths and os.path.isfile(self.segment_paths[idx]):
                segments_to_concat.append(self.segment_paths[idx])

        if not segments_to_concat:
            if self.verbose:
                print(f"[live] No segments available for clip extraction")
            return None

        # Concatenate segments
        self.highlight_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_name = f"highlight_{self.highlight_count:03d}_{timestamp}.mp4"
        clip_path = os.path.join(self.output_dir, clip_name)

        list_file = os.path.join(self.output_dir, f"_concat_{self.highlight_count}.txt")
        with open(list_file, "w") as f:
            for seg in segments_to_concat:
                f.write(f"file '{seg}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            clip_path,
        ]
        subprocess.run(cmd, capture_output=True)

        # Cleanup temp list file
        try:
            os.remove(list_file)
        except OSError:
            pass

        if os.path.isfile(clip_path):
            size_mb = os.path.getsize(clip_path) / (1024 * 1024)
            if self.verbose:
                print(
                    f"[live] Extracted clip: segments {start_idx}-{end_idx} "
                    f"-> {clip_path} ({size_mb:.1f} MB)"
                )
            return {
                "clip_name": clip_name,
                "clip_path": clip_path,
                "size_mb": size_mb,
                "segment_index": trigger_index,
                "segments": f"{start_idx}-{end_idx}",
            }
        else:
            if self.verbose:
                print(f"[live] Error: failed to create {clip_path}")
            return None
