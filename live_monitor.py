"""
live_monitor.py — Rich Live dashboard for real-time highlight detection

Displays stream status, detection metrics, and highlight log in a
continuously updating terminal UI.
"""

import threading
import time
from datetime import timedelta

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class LiveMonitor:
    """
    Real-time Rich dashboard for live highlight detection.

    Usage:
        monitor = LiveMonitor()
        monitor.start()
        # ... on each segment:
        monitor.update(energy=0.12, threshold=0.08, ...)
        # ... when done:
        monitor.stop()
    """

    def __init__(self, refresh_rate=1.0):
        self.refresh_rate = refresh_rate
        self._console = Console()
        self._live = None
        self._thread = None
        self._stop_event = threading.Event()

        # State
        self._stream_status = "CONNECTING"
        self._elapsed_sec = 0
        self._start_time = None
        self._segments_processed = 0
        self._highlight_count = 0
        self._strategy = "audio"
        self._warmup_remaining = 5

        # Detection metrics
        self._peak_energy = 0.0
        self._threshold = 0.0
        self._rolling_mean = 0.0
        self._rolling_var = 0.0
        self._is_warming_up = True

        # Highlight log (last 10)
        self._highlights = []
        self._max_log = 10

        self._lock = threading.Lock()

    def start(self):
        """Start the live dashboard."""
        self._start_time = time.time()
        self._stop_event.clear()

        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=int(1 / self.refresh_rate),
            screen=False,
        )
        self._live.start()

        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the dashboard and restore terminal."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self._live:
            self._live.stop()

    def update(
        self,
        peak_energy=None,
        threshold=None,
        rolling_mean=None,
        rolling_var=None,
        is_highlight=False,
        stream_status=None,
        segments_processed=None,
        warmup_remaining=None,
        highlight_info=None,
        strategy=None,
    ):
        """
        Update dashboard state. Called from segment callback.

        Args:
            peak_energy: current segment peak energy
            threshold: current adaptive threshold
            rolling_mean: EMA rolling mean
            rolling_var: EMA rolling variance
            is_highlight: True if this segment triggered a highlight
            stream_status: one of CONNECTED, BUFFERING, ERROR
            segments_processed: total segments processed so far
            warmup_remaining: segments until warmup completes
            highlight_info: dict with clip_name, clip_path, etc.
            strategy: detection strategy name
        """
        with self._lock:
            if peak_energy is not None:
                self._peak_energy = peak_energy
            if threshold is not None:
                self._threshold = threshold
                self._is_warming_up = False
            if rolling_mean is not None:
                self._rolling_mean = rolling_mean
            if rolling_var is not None:
                self._rolling_var = rolling_var
            if stream_status is not None:
                self._stream_status = stream_status
            if segments_processed is not None:
                self._segments_processed = segments_processed
            if warmup_remaining is not None:
                self._warmup_remaining = warmup_remaining
                self._is_warming_up = warmup_remaining > 0
            if strategy is not None:
                self._strategy = strategy

            if is_highlight:
                self._highlight_count += 1

            if highlight_info:
                self._highlights.append(highlight_info)
                if len(self._highlights) > self._max_log:
                    self._highlights = self._highlights[-self._max_log:]

    def _refresh_loop(self):
        """Background thread that refreshes the layout."""
        while not self._stop_event.is_set():
            if self._start_time:
                self._elapsed_sec = time.time() - self._start_time
            try:
                self._live.update(self._build_layout())
            except Exception:
                pass
            time.sleep(self.refresh_rate)

    def _build_layout(self):
        """Build the full dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
        )
        layout["body"].split_row(
            Layout(name="detection", ratio=1),
            Layout(name="highlights", ratio=1),
        )

        layout["header"].update(self._build_header())
        layout["detection"].update(self._build_detection_panel())
        layout["highlights"].update(self._build_highlights_panel())

        return layout

    def _build_header(self):
        """Stream status header bar."""
        with self._lock:
            status = self._stream_status
            elapsed = str(timedelta(seconds=int(self._elapsed_sec)))
            segs = self._segments_processed
            hl_count = self._highlight_count

        # Status color
        if status == "CONNECTED":
            status_text = Text(f" {status} ", style="bold white on green")
        elif status == "BUFFERING":
            status_text = Text(f" {status} ", style="bold black on yellow")
        else:
            status_text = Text(f" {status} ", style="bold white on red")

        header = Text.assemble(
            ("Football Highlighter ", "bold cyan"),
            status_text,
            (f"  Elapsed: {elapsed}", ""),
            (f"  Segments: {segs}", "dim"),
            (f"  Highlights: {hl_count}", "bold yellow" if hl_count else "dim"),
        )
        return Panel(header, style="bold")

    def _build_detection_panel(self):
        """Left panel: detection metrics."""
        with self._lock:
            strategy = self._strategy
            is_warming = self._is_warming_up
            warmup_rem = self._warmup_remaining
            peak = self._peak_energy
            threshold = self._threshold
            mean = self._rolling_mean
            var = self._rolling_var

        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column("Key", style="dim", width=16)
        table.add_column("Value")

        table.add_row("Strategy", Text(strategy, style="cyan"))

        if is_warming:
            table.add_row(
                "Warmup",
                Text(f"{warmup_rem} segments remaining", style="yellow"),
            )
        else:
            table.add_row("Warmup", Text("Complete", style="green"))

        # Energy bar visualization
        bar_width = 30
        energy_pct = min(1.0, peak / max(threshold, 0.001)) if threshold > 0 else 0
        filled = int(energy_pct * bar_width)
        bar = "[green]" + "|" * min(filled, bar_width)
        if filled > bar_width:
            bar = "[red]" + "|" * bar_width
        bar += "[dim]" + "." * max(0, bar_width - filled)

        table.add_row("Peak Energy", Text(f"{peak:.4f}"))
        table.add_row("Threshold", Text(f"{threshold:.4f}" if not is_warming else "--"))
        table.add_row("Energy Bar", Text.from_markup(bar))
        table.add_row("Rolling Mean", Text(f"{mean:.4f}"))
        table.add_row("Rolling Var", Text(f"{var:.6f}"))

        return Panel(table, title="Detection", border_style="blue")

    def _build_highlights_panel(self):
        """Right panel: highlight clip log."""
        with self._lock:
            highlights = list(self._highlights)

        if not highlights:
            content = Text("No highlights detected yet...", style="dim italic")
            return Panel(content, title="Highlights", border_style="yellow")

        table = Table(show_header=True, box=None, pad_edge=False)
        table.add_column("#", style="dim", width=3)
        table.add_column("Time", width=8)
        table.add_column("Clip", no_wrap=True)
        table.add_column("Size", width=8, justify="right")

        for i, hl in enumerate(highlights, 1):
            clip_name = hl.get("clip_name", "?")
            size_mb = hl.get("size_mb", 0)
            seg_idx = hl.get("segment_index", 0)
            approx_time = str(timedelta(seconds=seg_idx * 30))

            table.add_row(
                str(i),
                approx_time,
                clip_name,
                f"{size_mb:.1f}MB" if size_mb else "-",
            )

        return Panel(table, title="Highlights", border_style="yellow")
