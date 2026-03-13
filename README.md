# Football Highlighter

Stream-first soccer highlight pipeline: discover live streams, watch with real-time detection, or process recorded matches into polished vertical clips.

```
Live Stream / File ──► Detect Highlights ──► Cut Clips ──► Crop 9:16 ──► Score Overlay ──► Final .mp4
                       (audio / CV)          (ffmpeg)      (YOLO/center)  (+ color grade)
```

## Demo

**[Try it in the browser](https://vardst.github.io/football-highlighter/)** — runs entirely client-side, no uploads or installs needed.

## Features

- **Stream discovery** — browse IPTV sports streams with Rich interactive UI
- **Live detection** — real-time highlight detection with Rich dashboard while watching streams
- **Acestream support** — `acestream://` URLs auto-converted to local HTTP API
- **Audio-based detection** — crowd noise spike analysis with median filtering and percentile thresholds
- **YOLO computer vision** — ball tracking, player clustering, goalkeeper saves, fast ball detection
- **Combined scoring** — multi-signal fusion of audio + CV for best accuracy
- **Goal mode** — extract individual goal clips from match config with automatic score overlay
- **Smart crop** — ball-tracking 9:16/1:1 reframe with camera-cut detection and velocity-clamped smoothing
- **OCR kickoff detection** — reads broadcast match clock via EasyOCR to sync match time to video time
- **Color grading** — cinematic, dramatic, vibrant presets via ffmpeg
- **Custom YOLO model** — trained on SoccerNet + Soccana datasets (44K images) for 60-80% ball detection
- **Browser version** — runs entirely client-side with Web Audio API + ffmpeg.wasm ([try it](https://vardst.github.io/football-highlighter/))

## Quick Start

```bash
# Install
git clone https://github.com/vardst/football-highlighter.git
cd football-highlighter
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Browse & Watch Streams (primary use case)

```bash
# Browse IPTV sports streams
python fh.py browse
python fh.py browse --search "bein"
python fh.py browse --auto-watch              # select → immediately start watching

# Live highlight detection
python fh.py watch "http://stream.url/live.m3u8"
python fh.py watch "acestream://ffbf8c687c..." --strategy combined

# Record stream for later processing
python fh.py record "http://stream.url" -o match.mp4 -d 7200
```

### Highlights Mode

Detects the most exciting moments by audio energy and outputs a single compiled highlight reel.

```bash
python fh.py highlights match.mp4
python fh.py highlights match.mp4 --strategy combined --crop smart
```

| Flag | Default | Description |
|------|---------|-------------|
| `--strategy` | `audio` | `audio` (fast) or `combined` (audio + YOLO CV) |
| `--top-percent` | `10` | Top N% loudest moments to keep |
| `--crop` | `center` | `center` (fast) or `smart` (YOLO ball tracking) |
| `--grade` | `cinematic` | `cinematic`, `dramatic`, `vibrant`, `none` |
| `--aspect` | `9:16` | `9:16` (vertical) or `1:1` (square) |
| `--max-duration` | `3` | Max output duration in minutes |
| `--model` | auto | Path to YOLO weights |
| `--sahi` | off | Enable SAHI sliced inference |

### Goals Mode

Extracts each goal as a separate clip with score overlay, using a match config file.

```bash
python fh.py goals match.mp4 -c match_config.json
```

#### Match Config Format

```json
{
  "match": {
    "home": "Real Madrid",
    "away": "FC Barcelona",
    "competition": "La Liga",
    "date": "2024-04-21"
  },
  "kickoff_offsets": {
    "first_half": 354,
    "second_half": 3589
  },
  "goals": [
    {
      "minute": 6,
      "scorer": "Christensen",
      "team": "away",
      "score_after": "0-1"
    }
  ]
}
```

The pipeline will:
1. Detect kickoff times via OCR (or use `kickoff_offsets` fallback)
2. Convert match minutes to video timestamps
3. Refine each goal time using audio peak detection (+-120s window)
4. Cut 15s before to 90s after the peak
5. Crop to 9:16 or 1:1
6. Apply score overlay (updates at the goal moment)
7. Color grade

### Web Dashboard

A browser-based UI for stream browsing, recording, live watching with highlight detection, and clip downloading.

```bash
# Run dashboard (http://localhost:5555)
pip install -r requirements.txt && python dashboard.py

# Run dashboard exposed on LAN
python dashboard.py --host 0.0.0.0

# Run dashboard with ngrok (public URL)
pip install -r requirements.txt && python dashboard.py & ngrok http 5555
```

## Browser Version

A fully client-side web app that runs in the browser — no server, no uploads, no installs.

```bash
cd web
python -m http.server 8080
# Open http://localhost:8080
```

**What works in the browser:**
- Audio-based highlight detection (Web Audio API)
- Goal mode with match config
- Video cutting, cropping, score overlay, color grading (ffmpeg.wasm)
- 9:16 and 1:1 aspect ratios

**What requires the Python pipeline:**
- YOLO smart crop (browser uses center crop)
- OCR kickoff detection (browser uses manual input)
- Combined audio+CV detection strategy
- Live stream detection

## Architecture

```
fh.py                        Unified CLI entry point
dashboard.py                 Web dashboard (Flask backend + REST API)
stream_discovery.py          IPTV M3U playlist fetching & parsing
stream_browser.py            Rich interactive stream browser
live_monitor.py              Rich Live detection dashboard
live_detector.py             Real-time highlight detection for streams
stream_capture.py            Stream recording & segmentation
pipeline.py                  File-based pipeline (legacy entry point)
highlight_audio.py           Audio energy analysis, peak detection
highlight_cv.py              YOLO-based event detection
highlight_combined.py        Multi-signal scoring (audio + CV)
smart_crop.py                Ball-tracking 9:16/1:1 reframe
soccer_detector.py           Unified YOLO wrapper with SAHI support
kickoff_detector.py          OCR broadcast clock reading
goal_detector.py             Goal window computation from match config
score_overlay.py             ffmpeg drawtext score overlay
web/                         Browser version (vanilla JS + ffmpeg.wasm)
```

## Dependencies

**Python:** `moviepy numpy scipy opencv-python ultralytics requests sahi rich easyocr`

**System:** `ffmpeg` with freetype/drawtext support
```bash
# macOS (requires tap for drawtext support)
brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-fdk-aac
```

## License

MIT
