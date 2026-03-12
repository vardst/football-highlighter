# Football Highlighter

Automated soccer highlight pipeline: full match video in, polished vertical clips out.

```
Full Match .mp4 ──► Detect Highlights ──► Cut Clips ──► Crop 9:16 ──► Score Overlay ──► Final .mp4
                    (audio / CV)          (ffmpeg)      (YOLO/center)  (+ color grade)
```

## Demo

**[Try it in the browser](https://vardst.github.io/football-highlighter/)** — runs entirely client-side, no uploads or installs needed.

## Features

- **Audio-based detection** — crowd noise spike analysis with median filtering and percentile thresholds
- **YOLO computer vision** — ball tracking, player clustering, goalkeeper saves, fast ball detection
- **Combined scoring** — multi-signal fusion of audio + CV for best accuracy
- **Goal mode** — extract individual goal clips from match config with automatic score overlay
- **Smart crop** — ball-tracking 9:16/1:1 reframe with camera-cut detection and velocity-clamped smoothing
- **OCR kickoff detection** — reads broadcast match clock via EasyOCR to sync match time to video time
- **Color grading** — cinematic, dramatic, vibrant presets via ffmpeg
- **Custom YOLO model** — trained on SoccerNet + Soccana datasets (44K images) for 60-80% ball detection vs 14-27% with COCO
- **Browser version** — runs entirely client-side with Web Audio API + ffmpeg.wasm ([try it](https://vardst.github.io/football-highlighter/))

## Quick Start

```bash
# Install
git clone https://github.com/vardst/football-highlighter.git
cd football-highlighter
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generic highlights (audio-based, fast)
python pipeline.py --input match.mp4

# Goal clips with score overlay
python pipeline.py --input match.mp4 --mode goals --config match_config.json --crop smart

# With custom model and SAHI
python pipeline.py --input match.mp4 --mode goals --config match_config.json \
    --crop smart --model models/soccer_yolov8s.pt --sahi
```

## Usage

### Highlights Mode (default)

Detects the most exciting moments by audio energy and outputs a single compiled highlight reel.

```bash
python pipeline.py --input match.mp4 --strategy audio --top-percent 10
python pipeline.py --input match.mp4 --strategy combined --crop smart  # audio + CV
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
python pipeline.py --input match.mp4 --mode goals --config match_config.json
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

### Training a Custom Model

```bash
python train_model.py                  # full: download datasets + merge + train
python train_model.py --skip-download  # reuse downloaded data
python train_model.py --resume         # resume from checkpoint
python train_model.py --test-soccana   # quick test with pre-trained model
python dashboard.py                    # training monitor at localhost:8501
```

Datasets: SoccerNet_v3_H250 (19K images) + Soccana (25K images), unified to 4 classes: ball, player, goalkeeper, referee.

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

## Architecture

```
pipeline.py                  Main entry point, orchestrates all stages
highlight_audio.py           Audio energy analysis, peak detection
highlight_cv.py              YOLO-based event detection
highlight_combined.py        Multi-signal scoring (audio + CV)
smart_crop.py                Ball-tracking 9:16/1:1 reframe
soccer_detector.py           Unified YOLO wrapper with SAHI support
kickoff_detector.py          OCR broadcast clock reading
goal_detector.py             Goal window computation from match config
score_overlay.py             ffmpeg drawtext score overlay
train_model.py               Dataset download + model training
dashboard.py                 Training monitor web UI
web/                         Browser version (vanilla JS + ffmpeg.wasm)
```

## Dependencies

**Python:** `moviepy numpy scipy opencv-python ultralytics yt-dlp requests sahi huggingface_hub`

**System:** `ffmpeg` with freetype/drawtext support
```bash
# macOS (requires tap for drawtext support)
brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-fdk-aac
```

## License

MIT
