# Automated Soccer Highlight Pipeline — Build Guide

## The Pipeline

```
Full Match .mp4
  ├─ Audio energy analysis (crowd noise spikes)
  ├─ YOLO object detection (ball, players, goalkeeper, referee)
  └─ Match config (known goal times, teams, scores)
       │
       ▼
  Detect Highlights / Goal Windows
       │
       ▼
  Cut Clips → Smart Crop 9:16 → Score Overlay + Color Grade → Final .mp4
```

Two modes:
- **Highlights mode**: auto-detect exciting moments, concatenate into a reel
- **Goals mode**: extract each goal as a separate clip with score overlay

---

## Project Structure

```
football highlighter/
├── pipeline.py              # Main CLI — orchestrates everything
├── highlight_audio.py       # Audio energy peak detection
├── highlight_cv.py          # YOLO event detection (ball near goal, clustering, GK saves)
├── highlight_combined.py    # Fuses audio + CV signals into scored windows
├── smart_crop.py            # Ball-tracking 9:16 vertical reframe
├── soccer_detector.py       # Unified YOLO wrapper with SAHI support
├── kickoff_detector.py      # Broadcast clock OCR → kickoff timestamp
├── goal_detector.py         # Computes goal clip windows from match config
├── score_overlay.py         # ffmpeg drawtext score bug
├── train_model.py           # Downloads datasets + trains custom YOLOv8s
├── dashboard.py             # Training monitor web UI (localhost:8501)
├── match_config.json        # Match metadata: teams, goals, minutes
├── requirements.txt         # Python dependencies
├── models/                  # Trained model weights
├── data/                    # Downloaded datasets (SoccerNet + Soccana)
└── runs/                    # Training artifacts
```

---

## Phase 1 — Acquiring Footage

### Free / Open Sources

| Source | What You Get | How to Access |
|--------|-------------|---------------|
| YouTube full match channels | Full 90-min broadcasts | `yt-dlp` |
| SoccerFullMatch.com | PL, La Liga, Serie A, Bundesliga, CL replays | `yt-dlp` on page URL |
| r/footballhighlights | Community-posted match links | `yt-dlp` |
| SoccerNet Dataset | 500+ matches with temporal annotations | Academic NDA at soccernet.org |

### Downloading with yt-dlp

```bash
# Best quality MP4
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" --merge-output-format mp4 \
       -o "%(title)s.%(ext)s" "URL_HERE"

# From subscription services — use browser cookies
yt-dlp --cookies-from-browser chrome "URL_HERE"
```

For DRM-protected services (DAZN, Peacock), screen-record with OBS Studio instead.

---

## Phase 2 — Highlight Detection

### Strategy A: Audio Energy (Simplest, ~80% accuracy)

**File: `highlight_audio.py`**

Core insight: crowd noise and commentator voice spike during goals, near-misses, red cards. Computes RMS energy per second, applies median filter, takes top N% loudest windows, merges nearby peaks.

```bash
python pipeline.py --input match.mp4 --strategy audio --top-percent 10
```

### Strategy B: YOLO Computer Vision

**File: `highlight_cv.py`** using **`soccer_detector.py`**

Detects events via object detection:
- **Ball near goal area** (left/right 15% of frame)
- **Fast ball movement** between sampled frames
- **Player clustering** (celebrations, fouls)
- **Goalkeeper near ball in goal area** (strong goal signal)

Uses `SoccerDetector` which auto-selects the best available model:
1. `models/soccer_yolov8s.pt` — custom trained (best)
2. `models/soccana_yolov11n.pt` — pre-trained Soccana (good)
3. `yolov8n.pt` — COCO generic (fallback, 14-27% ball detection)

### Strategy C: Combined (Audio + Vision)

**File: `highlight_combined.py`**

Scores every second of the match by fusing audio energy + CV events with weighted contributions. Player clustering (celebrations) gets highest weight.

```bash
python pipeline.py --input match.mp4 --strategy combined --crop smart
```

### Goal Mode: Config-Driven Extraction

**Files: `goal_detector.py`, `kickoff_detector.py`, `match_config.json`**

For known matches, provide a config with goal times:

```json
{
  "match": {
    "home": "Real Madrid",
    "away": "Barcelona",
    "goals": [
      {"minute": 12, "team": "away", "scorer": "Christensen", "score_after": "0-1"},
      {"minute": 45, "team": "home", "scorer": "Vinicius", "score_after": "1-1"}
    ]
  }
}
```

The pipeline uses OCR to read the broadcast clock, computes kickoff offsets, and cuts precise goal windows with audio peak refinement (±120s window).

```bash
python pipeline.py --input match.mp4 --mode goals --config match_config.json
```

---

## Phase 3 — Custom Soccer YOLO Model

### Why Custom?

COCO-pretrained `yolov8n.pt` detects "sports ball" (class 32) and "person" (class 0). Ball detection is only **14-27%** in broadcast footage because:
- Ball is tiny (few pixels) in 720p wide shots
- COCO's "sports ball" covers all sports, not tuned for football
- Nano model trades accuracy for speed

### Two-Phase Approach

**Phase 1: Quick Win** — Download pre-trained Soccana YOLOv11n model. Immediate improvement, zero training.

```bash
python train_model.py --test-soccana --video match.mp4
```

**Phase 2: Custom Training** — Merge SoccerNet_v3_H250 (19K images) + Soccana (25K images), fine-tune YOLOv8s.

```bash
python train_model.py                    # full: download + merge + train
python train_model.py --download-only    # just download and prepare data
python train_model.py --skip-download    # train on existing data
python train_model.py --resume           # resume from last checkpoint
python train_model.py --epochs 30        # custom epoch count
```

### Unified Class Mapping

```
SoccerNet_v3_H250:              Soccana:                 Unified:
  0: ball      → 0: ball         0: player → 1: player     0: ball
  1: person    → 1: player       1: referee → 3: referee    1: player
                                  2: ball   → 0: ball       2: goalkeeper (reserved)
                                                             3: referee
```

### Training Config

- Base model: YOLOv8s (small — 11.2M params)
- Image size: 640
- Epochs: 20 (default)
- Batch: 8 (safe for MPS) or 16 (with TAL patch)
- Device: MPS (Apple Silicon) with TAL patch, or CPU fallback
- Augmentation: mosaic, mixup, copy_paste, scale, HSV shifts
- Checkpoints: saved every epoch for resume support

### MPS Bug Workaround

Ultralytics has a known shape mismatch bug in `tal.py` on MPS. The fix patches `venv/.../ultralytics/utils/tal.py` to do masked tensor assignment on CPU then move back to MPS. Without the patch, training crashes randomly during the first few epochs.

### Training Dashboard

```bash
python dashboard.py    # opens http://localhost:8501
```

Real-time web UI showing:
- Epoch/batch progress bars
- Loss values (box, class, DFL)
- Completed epochs with mAP scores
- Start/Stop/Resume buttons
- Live training log

### SoccerDetector API

```python
from soccer_detector import SoccerDetector

det = SoccerDetector()                          # auto-detect best model
det = SoccerDetector(use_sahi=True)             # with SAHI sliced inference
det = SoccerDetector(model_path="custom.pt")    # explicit model

result = det.detect(frame)
# result.ball_x          — normalized ball x (0-1), None if not detected
# result.ball_conf       — ball confidence
# result.players_x       — list of player x positions
# result.goalkeepers_x   — list of goalkeeper x positions
# result.referees_x      — list of referee x positions

# Backward-compatible interface for smart_crop:
ball_x, players_x = det.detect_frame_compat(frame, width)
```

Confidence thresholds: ball=0.3, player=0.5, goalkeeper=0.4, referee=0.4.

---

## Phase 4 — 9:16 Vertical Crop

### Center Crop (Simple)

Takes center strip of the frame, scales to 1080x1920.

```bash
python pipeline.py --input match.mp4 --crop center
```

### Smart Crop (Ball-Tracking)

**File: `smart_crop.py`**

1. **Detect camera cuts** via histogram difference
2. **Track ball position** with YOLO, fall back to player centroid
3. **Smooth per-scene** with bidirectional EMA
4. **Clamp velocity** — no more than 8% frame width per frame
5. **Render** via OpenCV crop piped to ffmpeg

```bash
python pipeline.py --input match.mp4 --crop smart --model models/soccer_yolov8s.pt --sahi
```

With the custom model + SAHI, ball detection jumps from ~14% to ~60-80%, dramatically improving crop accuracy.

---

## Phase 5 — Post-Processing

### Score Overlay

**File: `score_overlay.py`**

Uses ffmpeg `drawtext` with `enable='between(t,START,END)'` to show score at the right time in each goal clip. Requires ffmpeg with freetype support:

```bash
brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-fdk-aac
```

### Color Grading

Four styles via ffmpeg `eq` filter:
- `cinematic` — boosted contrast + warm tint (default)
- `dramatic` — high contrast, desaturated
- `vibrant` — bright, punchy saturation
- `none` — no grading

---

## Phase 6 — Running the Full Pipeline

### Generic Highlights

```bash
# Audio-only (fastest)
python pipeline.py --input match.mp4

# Audio + CV (best quality)
python pipeline.py --input match.mp4 --strategy combined --crop smart

# Custom model with SAHI
python pipeline.py --input match.mp4 --strategy combined --crop smart \
    --model models/soccer_yolov8s.pt --sahi
```

### Goal Clips

```bash
# Individual goal clips with score overlay
python pipeline.py --input match.mp4 --mode goals --config match_config.json \
    --crop smart --grade cinematic

# With custom model
python pipeline.py --input match.mp4 --mode goals --config match_config.json \
    --crop smart --model models/soccer_yolov8s.pt --sahi
```

### CLI Flags

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--input` | path | required | Full match .mp4 |
| `--output` | path | `highlights_9x16.mp4` | Output file |
| `--mode` | `highlights`, `goals` | `highlights` | Pipeline mode |
| `--config` | path | — | match_config.json (required for goals mode) |
| `--strategy` | `audio`, `combined` | `audio` | Detection strategy |
| `--top-percent` | int | `10` | Top N% loudest moments |
| `--crop` | `center`, `smart` | `center` | Crop mode |
| `--grade` | `cinematic`, `dramatic`, `vibrant`, `none` | `cinematic` | Color grade |
| `--max-duration` | float | `3` | Max highlight minutes |
| `--model` | path | auto-detect | YOLO model weights |
| `--sahi` | flag | off | SAHI sliced inference |

---

## Dependencies

```bash
pip install moviepy numpy scipy opencv-python ultralytics yt-dlp requests sahi huggingface_hub
```

System: `ffmpeg` with freetype support (see score overlay section).

---

## Kickoff Detection — Technical Notes

- YOLO center-circle detection does NOT work for broadcast video (lineup graphics, pre-match ceremonies confuse it)
- **OCR-based approach works**: read broadcast match clock via EasyOCR, compute video-to-match offset
- Clock position for LaLiga 2023-24: top-left scoreboard, ~y:60-100, x:10-220 in 1280x720
- EasyOCR reads ":" as "4", ".", "'", "*" — parser handles these artifacts
- 5-digit OCR results like "13411" = "13:11" (middle digit is misread colon)
- Need 3x upscale of scoreboard crop for reliable OCR
- Scan every 60s during gameplay, use median of all readings for robustness
- Config fallback: `kickoff_offsets.first_half` and `kickoff_offsets.second_half` in match_config.json

---

## Build Order (Recommended)

1. Audio-only detection → working MVP in ~50 lines
2. ffmpeg cutting + center crop → end-to-end vertical highlights
3. Match config + kickoff OCR → precise goal extraction with score overlay
4. YOLO vision detection → catch events audio misses
5. Smart crop → vertical framing follows the ball
6. Custom soccer model + SAHI → dramatically better ball detection
7. Training dashboard → monitor long training runs

Each step is independently useful. You don't need all of them to get value.
