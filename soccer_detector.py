"""
soccer_detector.py — Inference wrapper for custom soccer YOLO model

Supports:
    - Standard YOLO inference (fast)
    - SAHI sliced inference (better for small ball detection)
    - Automatic fallback to COCO-pretrained yolov8n.pt

Classes (custom model):
    0: ball
    1: player
    2: goalkeeper
    3: referee

Usage:
    from soccer_detector import SoccerDetector

    det = SoccerDetector()  # auto-detects best available model
    result = det.detect(frame)
    print(result.ball_x, result.ball_conf)
"""

import os
from dataclasses import dataclass, field

import numpy as np

# Class IDs for the custom soccer model
BALL = 0
PLAYER = 1
GOALKEEPER = 2
REFEREE = 3

# Confidence thresholds per class
DEFAULT_CONF = {
    BALL: 0.3,
    PLAYER: 0.5,
    GOALKEEPER: 0.4,
    REFEREE: 0.4,
}

# COCO class IDs (fallback)
COCO_PERSON = 0
COCO_BALL = 32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CUSTOM_MODEL = os.path.join(BASE_DIR, "models", "soccer_yolov8s.pt")
DEFAULT_SOCCANA_MODEL = os.path.join(BASE_DIR, "models", "soccana_yolov11n.pt")
DEFAULT_COCO_MODEL = os.path.join(BASE_DIR, "yolov8n.pt")


@dataclass
class DetectionResult:
    """Detection results for a single frame."""
    # Ball
    ball_x: float = None      # normalized x position (0-1), None if not detected
    ball_y: float = None      # normalized y position (0-1)
    ball_conf: float = 0.0    # confidence score

    # Players (list of normalized x positions)
    players_x: list = field(default_factory=list)

    # Goalkeepers (list of normalized x positions)
    goalkeepers_x: list = field(default_factory=list)

    # Referees (list of normalized x positions)
    referees_x: list = field(default_factory=list)

    # Raw boxes for advanced use: list of (cls, conf, x1, y1, x2, y2)
    raw_boxes: list = field(default_factory=list)


class SoccerDetector:
    """
    Soccer-specific object detector with SAHI support.

    Automatically selects the best available model:
        1. models/soccer_yolov8s.pt  (custom trained — best)
        2. models/soccana_yolov11n.pt (pre-trained Soccana — good)
        3. yolov8n.pt (COCO generic — fallback)
    """

    def __init__(self, model_path=None, use_sahi=False, conf_thresholds=None):
        """
        Args:
            model_path: explicit path to .pt weights. None = auto-detect.
            use_sahi: enable SAHI sliced inference for better small object detection.
            conf_thresholds: dict of class_id -> confidence threshold overrides.
        """
        self.use_sahi = use_sahi
        self.conf = {**DEFAULT_CONF, **(conf_thresholds or {})}
        self._is_coco = False
        self._sahi_model = None

        # Auto-detect model
        if model_path is None:
            model_path = self._find_best_model()

        self.model_path = model_path
        self._load_model(model_path)

    def _find_best_model(self):
        """Find the best available model, in priority order."""
        for path in [DEFAULT_CUSTOM_MODEL, DEFAULT_SOCCANA_MODEL]:
            if os.path.isfile(path):
                return path
        return DEFAULT_COCO_MODEL

    def _load_model(self, model_path):
        """Load the YOLO model and determine class mapping."""
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        model_names = self.model.names

        # Detect if this is a COCO model by checking class names
        if model_names.get(32) == "sports ball" or model_names.get(0) == "person":
            self._is_coco = True
            print(f"  [detector] Using COCO model: {os.path.basename(model_path)}")
            print(f"  [detector] Ball detection will be limited (COCO generic)")
        else:
            self._is_coco = False
            print(f"  [detector] Using soccer model: {os.path.basename(model_path)}")
            print(f"  [detector] Classes: {model_names}")

        # Set up SAHI if requested
        if self.use_sahi:
            self._setup_sahi(model_path)

    def _setup_sahi(self, model_path):
        """Initialize SAHI for sliced inference."""
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model_path,
                confidence_threshold=0.2,  # low threshold, we filter per-class later
                device="cpu",  # SAHI handles device internally
            )
            self._sahi_predict = get_sliced_prediction
            print(f"  [detector] SAHI sliced inference enabled")
        except ImportError:
            print(f"  [detector] SAHI not installed, falling back to standard inference")
            print(f"  [detector] Install with: pip install sahi")
            self.use_sahi = False

    def detect(self, frame) -> DetectionResult:
        """
        Run detection on a single frame.

        Args:
            frame: BGR numpy array (OpenCV format)

        Returns:
            DetectionResult with normalized positions
        """
        if self.use_sahi and self._sahi_model is not None:
            return self._detect_sahi(frame)
        return self._detect_standard(frame)

    def _detect_standard(self, frame) -> DetectionResult:
        """Standard YOLO inference."""
        h, w = frame.shape[:2]
        result = DetectionResult()

        if self._is_coco:
            results = self.model(frame, verbose=False, classes=[COCO_PERSON, COCO_BALL])
        else:
            results = self.model(frame, verbose=False)

        boxes = results[0].boxes
        best_ball_conf = 0

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if self._is_coco:
                cls, conf = self._map_coco(cls, conf)
                if cls is None:
                    continue

            # Apply per-class confidence threshold
            min_conf = self.conf.get(cls, 0.5)
            if conf < min_conf:
                continue

            result.raw_boxes.append((cls, conf, x1, y1, x2, y2))

            if cls == BALL and conf > best_ball_conf:
                result.ball_x = cx / w
                result.ball_y = cy / h
                result.ball_conf = conf
                best_ball_conf = conf
            elif cls == PLAYER:
                result.players_x.append(cx / w)
            elif cls == GOALKEEPER:
                result.goalkeepers_x.append(cx / w)
            elif cls == REFEREE:
                result.referees_x.append(cx / w)

        return result

    def _detect_sahi(self, frame) -> DetectionResult:
        """SAHI sliced inference for better small object detection."""
        h, w = frame.shape[:2]
        result = DetectionResult()

        sahi_result = self._sahi_predict(
            detection_model=self._sahi_model,
            image=frame,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=True,   # also run full-image detection
            postprocess_type="NMS",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
            verbose=0,
        )

        best_ball_conf = 0

        for pred in sahi_result.object_prediction_list:
            cls = pred.category.id
            conf = pred.score.value
            bbox = pred.bbox

            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if self._is_coco:
                cls, conf = self._map_coco(cls, conf)
                if cls is None:
                    continue

            min_conf = self.conf.get(cls, 0.5)
            if conf < min_conf:
                continue

            result.raw_boxes.append((cls, conf, x1, y1, x2, y2))

            if cls == BALL and conf > best_ball_conf:
                result.ball_x = cx / w
                result.ball_y = cy / h
                result.ball_conf = conf
                best_ball_conf = conf
            elif cls == PLAYER:
                result.players_x.append(cx / w)
            elif cls == GOALKEEPER:
                result.goalkeepers_x.append(cx / w)
            elif cls == REFEREE:
                result.referees_x.append(cx / w)

        return result

    def _map_coco(self, cls, conf):
        """Map COCO class IDs to unified soccer classes."""
        if cls == COCO_BALL:
            return BALL, conf
        elif cls == COCO_PERSON:
            return PLAYER, conf
        return None, 0

    def detect_frame_compat(self, frame, width):
        """
        Backward-compatible interface matching smart_crop._detect_frame().
        Returns (ball_x_norm, players_x_norm_list).
        """
        result = self.detect(frame)

        # Combine players and goalkeepers for crop purposes
        all_players = result.players_x + result.goalkeepers_x

        return result.ball_x, all_players
