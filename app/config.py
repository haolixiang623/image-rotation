"""
Core configuration for the image orientation API service.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Model ──────────────────────────────────────────────────────────────────────
# DuarteBarbosa/deep-image-orientation-detection — EfficientNetV2-S 4-class
#   class 0 → 0°  (correct orientation)
#   class 1 → 90° clockwise  correction needed
#   class 2 → 180°  correction needed
#   class 3 → 90° counter-clockwise  correction needed
MODEL_DIR = BASE_DIR / "models" / "orientation_detector"
ONNX_MODEL_PATH = MODEL_DIR / "orientation_model_v2_0.9882.onnx"

# Inference settings
THUMBNAIL_SIZE = 384                # px — model's expected square input size
ONNX_PROB_THRESHOLD = 0.5           # confidence threshold for ONNX label
ONNX_INTRA_OP_THREADS = 4            # balance between per-call latency & parallelism

# Rotation angle thresholds (degrees)
BIG_ANGLE_THRESHOLD = 180.0          # >= this → ONNX handles it directly
                                       # Set to 180° (not 15°) because ONNX 90° predictions
                                       # are unreliable on landscape/portrait aspect-ratio
                                       # confusion (e.g. zm.jpg 1440×810 → conf=0.92 for 90°).
                                       # 180° predictions are unambiguous (倒置特征) and safe to trust.
SMALL_ANGLE_THRESHOLD = 0.3         # < this → skip deskew (avoid noise)
DESKEW_COEFFICIENT = 1.0            # multiply deskew angle (tune to taste)

# Image quality & format
OUTPUT_FORMAT = "PNG"                # lossless; use "JPEG" to save space
OUTPUT_QUALITY = 95                  # only used for JPEG

# Concurrency
MAX_UPLOAD_SIZE_MB = 30
MAX_WORKERS = int(os.environ.get("WORKERS", 2 * os.cpu_count() + 1))

# Application
APP_VERSION = "2.0.0"

# Logging
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
