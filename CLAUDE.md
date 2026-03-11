# Football Highlighter

Automated soccer highlight pipeline: full match video in, polished 9:16 vertical clips out.

## Architecture

```
Full Match .mp4 → Detect Highlights → Cut Clips → Crop 9:16 → Score Overlay + Color Grade → Final .mp4
```

### Core Modules

| File | Purpose |
|------|---------|
| `pipeline.py` | Main entry point, orchestrates all stages |
| `highlight_audio.py` | Audio energy analysis — detects crowd noise spikes |
| `highlight_cv.py` | YOLO-based event detection — ball near goal, fast ball, player clustering, goalkeeper saves |
| `highlight_combined.py` | Multi-signal scoring — merges audio + CV signals |
| `smart_crop.py` | YOLO-tracked 9:16 reframe — follows ball with velocity-clamped smoothing |
| `soccer_detector.py` | Unified inference wrapper with SAHI support, auto-detects best model |
| `kickoff_detector.py` | OCR-based broadcast clock reading to find kickoff timestamps |
| `goal_detector.py` | Computes goal windows from match_config.json + kickoff offsets |
| `score_overlay.py` | ffmpeg drawtext score overlay with `enable='between(t,S,E)'` |
| `train_model.py` | Downloads SoccerNet + Soccana datasets, trains custom YOLOv8s |
| `dashboard.py` | Web dashboard for monitoring training at localhost:8501 |

### Detection Strategies

- **Audio-only** (`--strategy audio`): Fastest, ~80% accuracy. Detects crowd noise peaks.
- **Combined** (`--strategy combined`): Audio + YOLO CV. Better but slower.
- **Goal mode** (`--mode goals --config match_config.json`): Extracts individual goal clips with score overlay using known match data.

### Custom Soccer Model

- Unified classes: `0=ball, 1=player, 2=goalkeeper, 3=referee`
- `SoccerDetector` auto-selects: `soccer_yolov8s.pt` > `soccana_yolov11n.pt` > `yolov8n.pt` (COCO fallback)
- COCO yolov8n ball detection: 14-27%. Custom model target: 60-80%.
- SAHI sliced inference (`--sahi`) for better small ball detection

## Usage

```bash
# Generic highlights (audio-based, fast)
python pipeline.py --input match.mp4

# Goal clips with score overlay
python pipeline.py --input match.mp4 --mode goals --config match_config.json --crop smart

# With custom model and SAHI
python pipeline.py --input match.mp4 --mode goals --config match_config.json --crop smart --model models/soccer_yolov8s.pt --sahi

# Train custom model
python train_model.py                      # full: download + merge + train
python train_model.py --skip-download      # reuse downloaded data
python train_model.py --resume             # resume from checkpoint
python train_model.py --test-soccana       # test pre-trained Soccana model

# Training dashboard
python dashboard.py                        # open http://localhost:8501
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

### Training
- Datasets: SoccerNet_v3_H250 (Zenodo, 19K imgs) + Soccana (HuggingFace, 25K imgs)
- MPS has a known shape mismatch bug in Ultralytics TAL — patched in `venv/.../ultralytics/utils/tal.py` (CPU fallback for masked assignment)
- Checkpoints saved every epoch to `runs/soccer_v1/weights/last.pt` — resume with `--resume`

### ffmpeg
- `drawtext` filter requires `homebrew-ffmpeg/ffmpeg` tap (default brew ffmpeg lacks freetype)
- Install: `brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-fdk-aac`

## Directory Structure

```
.
├── pipeline.py              # Main entry point
├── highlight_audio.py       # Audio energy detection
├── highlight_cv.py          # YOLO event detection
├── highlight_combined.py    # Multi-signal fusion
├── smart_crop.py            # Ball-tracking 9:16 crop
├── soccer_detector.py       # Unified YOLO wrapper + SAHI
├── kickoff_detector.py      # OCR clock reading
├── goal_detector.py         # Goal window computation
├── score_overlay.py         # Score text overlay
├── train_model.py           # Dataset download + training
├── dashboard.py             # Training monitor web UI
├── match_config.json        # Match metadata (goals, teams, etc.)
├── requirements.txt         # Python dependencies
├── models/                  # Trained weights (.gitignored)
├── data/                    # Datasets (.gitignored)
├── runs/                    # Training artifacts (.gitignored)
└── venv/                    # Python virtual environment
```

## Dependencies

Python: `moviepy numpy scipy opencv-python ultralytics yt-dlp requests sahi huggingface_hub`
System: `ffmpeg` (with freetype/drawtext support)
