# Football Highlighter

Stream-first football highlight pipeline: discover streams, watch live with real-time detection, or process recorded matches into polished 9:16 vertical clips.

## Architecture

```
Live Stream / File → Detect Highlights → Cut Clips → Crop 9:16 → Score Overlay + Color Grade → Final .mp4
```

### Core Modules

| File | Purpose |
|------|---------|
| `fh.py` | Unified CLI entry point — browse, watch, record, highlights, goals |
| `stream_discovery.py` | Fetch & parse IPTV M3U playlists, filter sports, cache, probe |
| `stream_browser.py` | Rich interactive stream browser with search/filter/select |
| `live_monitor.py` | Rich Live dashboard for real-time detection status |
| `live_detector.py` | Real-time highlight detection for live stream segments |
| `stream_capture.py` | Stream recording and segmentation via ffmpeg |
| `pipeline.py` | File-based pipeline, orchestrates all stages |
| `highlight_audio.py` | Audio energy analysis — detects crowd noise spikes |
| `highlight_cv.py` | YOLO-based event detection — ball near goal, fast ball, player clustering, goalkeeper saves |
| `highlight_combined.py` | Multi-signal scoring — merges audio + CV signals |
| `smart_crop.py` | YOLO-tracked 9:16 reframe — follows ball with velocity-clamped smoothing |
| `soccer_detector.py` | Unified inference wrapper with SAHI support, auto-detects best model |
| `kickoff_detector.py` | OCR-based broadcast clock reading to find kickoff timestamps |
| `goal_detector.py` | Computes goal windows from match_config.json + kickoff offsets |
| `score_overlay.py` | ffmpeg drawtext score overlay with `enable='between(t,S,E)'` |

### Detection Strategies

- **Audio-only** (`--strategy audio`): Fastest, ~80% accuracy. Detects crowd noise peaks.
- **Combined** (`--strategy combined`): Audio + YOLO CV. Better but slower.
- **Goal mode** (`goals` subcommand): Extracts individual goal clips with score overlay using known match data.

### Custom Soccer Model

- Unified classes: `0=ball, 1=player, 2=goalkeeper, 3=referee`
- `SoccerDetector` auto-selects: `soccer_yolov8s.pt` > `soccana_yolov11n.pt`
- SAHI sliced inference (`--sahi`) for better small ball detection
- Model stored at `models/soccer_yolov8s.pt`

## Usage

```bash
# Browse IPTV sports streams
python fh.py browse
python fh.py browse --search "bein" --auto-watch

# Live highlight detection (primary use case)
python fh.py watch "http://stream.url/live.m3u8"
python fh.py watch "acestream://ffbf8c..." --strategy combined

# Record stream for later processing
python fh.py record "http://stream.url" -o match.mp4 -d 7200

# Generic highlights from file (audio-based, fast)
python fh.py highlights match.mp4
python fh.py highlights match.mp4 --strategy combined --crop smart

# Goal clips with score overlay
python fh.py goals match.mp4 -c match_config.json

# Legacy pipeline entry point (still works)
python pipeline.py --input match.mp4
```

## Key Technical Details

### Kickoff Detection
- YOLO center-circle detection does NOT work for broadcast video (lineup graphics confuse it)
- OCR-based approach works: read broadcast match clock via EasyOCR, compute video-to-match offset
- Clock position for LaLiga 2023-24: top-left scoreboard, ~y:60-100, x:10-220 in 1280x720
- EasyOCR reads ":" as "4", ".", "'", "*" — parser handles these artifacts
- Need 3x upscale of scoreboard crop for reliable OCR

### Smart Crop
- Ball-first tracking with player centroid fallback
- Camera-cut detection via histogram difference, resets tracking at cuts
- Bidirectional EMA smoothing per scene, velocity-clamped to prevent jarring pans
- Render via OpenCV frame crop piped to ffmpeg

### Stream Discovery
- Fetches public IPTV M3U playlists from iptv-org and Free-TV
- File-based caching in `.stream_cache/` with 6h TTL
- Sports keyword filtering, concurrent probe for availability
- Acestream URLs auto-converted to local HTTP API

### ffmpeg
- `drawtext` filter requires `homebrew-ffmpeg/ffmpeg` tap (default brew ffmpeg lacks freetype)
- Install: `brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-fdk-aac`

## Directory Structure

```
.
├── fh.py                    # Unified CLI entry point
├── stream_discovery.py      # IPTV stream fetching & parsing
├── stream_browser.py        # Rich interactive stream browser
├── live_monitor.py          # Rich Live detection dashboard
├── live_detector.py         # Real-time highlight detection
├── stream_capture.py        # Stream recording & segmentation
├── pipeline.py              # File-based pipeline
├── highlight_audio.py       # Audio energy detection
├── highlight_cv.py          # YOLO event detection
├── highlight_combined.py    # Multi-signal fusion
├── smart_crop.py            # Ball-tracking 9:16 crop
├── soccer_detector.py       # Unified YOLO wrapper + SAHI
├── kickoff_detector.py      # OCR clock reading
├── goal_detector.py         # Goal window computation
├── score_overlay.py         # Score text overlay
├── match_config.json        # Match metadata (goals, teams, etc.)
├── requirements.txt         # Python dependencies
├── models/                  # Trained weights (.gitignored)
├── web/                     # Browser version (vanilla JS + ffmpeg.wasm)
└── venv/                    # Python virtual environment
```

## Dependencies

Python: `moviepy numpy scipy opencv-python ultralytics requests sahi rich easyocr`
System: `ffmpeg` (with freetype/drawtext support)
