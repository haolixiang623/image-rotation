# Government Document Auto-Orientation API

High-concurrency, lossless image orientation correction pipeline for government administrative materials.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gunicorn (N workers)                     │
│  ┌──────────────┐  ┌──────────────┐       ┌──────────────┐     │
│  │ UvicornWorker│  │ UvicornWorker│  ...  │ UvicornWorker│     │
│  │   Process 1  │  │   Process 2  │       │   Process N  │     │
│  │              │  │              │       │              │     │
│  │ ImageProcessor│  │ ImageProcessor│      │ ImageProcessor│    │
│  │  (ONNX sess) │  │  (ONNX sess) │       │  (ONNX sess) │     │
│  │ intra_op=4   │  │ intra_op=4   │       │ intra_op=4   │     │
│  └──────────────┘  └──────────────┘       └──────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

**Each worker = 1 process = 1 ONNX session pinned to 4 CPU threads.**
Workers are independent — no shared mutable state.

## Processing Pipeline

```
Upload (multipart/form-data)
        │
        ▼
┌──────────────────┐     ┌─────────────────────────┐
│  EXIF Tag 0x0112 │ ──▶ │ orientation-corrected   │
│  auto-correction │     │ PIL Image               │
└──────────────────┘     └───────────┬─────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  Create 384px centre-crop        │
                    │  square thumbnail (for ONNX)    │
                    └───────────────┬──────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │  ONNX 4-Class Orientation Classification   │
              │  EfficientNetV2-S (DuarteBarbosa model)   │
              │  class 0 → 0°  class 1 → 270°             │
              │  class 2 → 180° class 3 → 90°            │
              │  accuracy: 98.82%                         │
              │  latency: ~110ms (CPU, intra_op=4)        │
              └─────────────────────┬────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │  Deskew (Hough lines + median angle)       │
              │  • angle < 0.3° → skip (avoid noise)      │
              └─────────────────────┬────────────────────┘
                                    │
        ┌───────────────────────────▼─────────────────────────┐
        │  Apply INTER_CUBIC rotation on ORIGINAL full-res     │
        │  canvas-expansion so no content is clipped         │
        └───────────────────────────┬─────────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Return PNG binary  │
                         │  + response headers │
                         └─────────────────────┘
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI app, endpoints, lifespan
│   ├── config.py             # Centralised configuration
│   └── processors/
│       ├── __init__.py
│       └── image_processor.py # ImageProcessor class (pipeline)
├── models/
│   └── orientation_detector/
│       └── orientation_model_v2_0.9882.onnx  # EfficientNetV2-S ONNX
├── tests/
│   └── test_client.py        # Concurrent test client
├── logs/                     # Auto-created at runtime
├── requirements.txt
├── deployment.sh             # Production launcher
└── README.md
```

## Prerequisites

| Dependency | Purpose |
|---|---|
| Python ≥ 3.9 | Runtime |
| ONNX model file | AI inference (see below) |
| `pip install -r requirements.txt` | Python packages |

### Downloading the ONNX Model

The orientation detection model must be downloaded separately (~80 MB).

```bash
# 1. Create directory
mkdir -p models/orientation_detector

# 2. Download from HuggingFace (~77 MB)
curl -L -o models/orientation_detector/orientation_model_v2_0.9882.onnx \
  "https://huggingface.co/DuarteBarbosa/deep-image-orientation-detection" \
  "/resolve/main/orientation_model_v2_0.9882.onnx"

# 3. Verify
ls -lh models/orientation_detector/
# Expected: orientation_model_v2_0.9882.onnx  (~77 MB)
```

## Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Make deployment script executable
chmod +x deployment.sh
```

## Running the Service

### Development (single worker, hot reload)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production (Gunicorn multi-process, recommended)

```bash
# Auto-detect CPU cores and launch
./deployment.sh

# Or with explicit worker count
./deployment.sh --workers 17 --port 8080

# Or with environment variable
WORKERS=8 LOG_LEVEL=DEBUG ./deployment.sh
```

**Formula:** `workers = 2 × physical_CPU_cores + 1`

On an 8-core machine this gives 17 workers. Each worker runs one ONNX session
pinned to 4 threads (`intra_op_num_threads=4`), so 4 × 17 = 68 threads
consumed at full load. With logical cores doubling physical, this stays within bounds.

### Verify the Service

```bash
# Health check
curl http://localhost:8000/health | python3 -m json.tool

# Service metadata
curl http://localhost:8000/ | python3 -m json.tool
```

## API Reference

### `POST /v1/image/auto-orient`

Correct the orientation of an uploaded government document image.

**Request**

```
Content-Type: multipart/form-data

file            (required)  — Image file: JPEG, PNG, TIFF, WebP, BMP
return_base64                — If "true", return JSON with base64-encoded image
return_metadata              — If "true", attach processing metadata to headers (default: true)
```

**Binary Response** (`return_base64=false`, default)

```
HTTP 200
Content-Type: image/png
X-Orientation-Corrected-Deg: <float>   — Total rotation applied (degrees)
X-Processing-Time-Ms:        <float>   — Server-side processing time
X-Request-Time-Ms:          <float>   — End-to-end request time
X-Exif-Correction-Deg:       <float>   — EXIF tag rotation component
X-ONNX-Angle-Deg:           <float>   — ONNX-predicted rotation component
X-Deskew-Angle-Deg:         <float>   — Hough-line deskew component
Content-Disposition: attachment; filename="oriented_<original_name>"
```

**JSON Response** (`return_base64=true`)

```json
{
  "image_base64": "<base64-encoded PNG>",
  "content_type": "image/png",
  "metadata": {
    "original_filename": "form.jpg",
    "exif_correction_deg": 0.0,
    "onnx_angle_deg": 270.0,
    "deskew_angle_deg": 0.0,
    "total_correction_deg": 270.0,
    "processing_time_ms": 487.3,
    "request_time_ms": 523.1
  }
}
```

### `GET /health`

Liveness and readiness probe.

```json
{
  "status": "healthy",
  "pid": 12345,
  "cpu_count": 8,
  "cpu_count_physical": 8,
  "memory_available_mb": 8192,
  "model_loaded": true,
  "uptime": "45s"
}
```

## Concurrency Testing

```bash
# 1. Drop a test image into tests/sample.jpg
cp /path/to/your/image.jpg tests/sample.jpg

# 2. Warm up the service (2 requests)
python tests/test_client.py --warmup 2

# 3. Concurrent load test: 10 parallel workers, 100 total requests
python tests/test_client.py \
    --concurrency 10 \
    --total 100 \
    --output-dir ./output/

# 4. Single request with base64 response
python tests/test_client.py --file tests/sample.jpg --base64
```

Expected performance targets (8-core machine, --concurrency 10):

```
  Latency (ms)
    p50                                ~200
    p90                                ~350
    p99                                ~600
    mean                               ~220
  Throughput
    Effective throughput               >5 req/s per worker pool
```

## Memory Management

- **Thumbnail (384 px, square)** used for ONNX + deskew inference — minimal RAM.
- **Original full-res array** only allocated for the final rotation step.
- `gc.collect()` called immediately after each large numpy array goes out of scope.
- ONNX `SessionOptions.intra_op_num_threads = 4` balances per-call latency
  with overall system parallelism.

## Performance Breakdown

| Stage | Time (typical) | Notes |
|---|---|---|
| ONNX 4-class inference | ~110ms | EfficientNetV2-S, CPU, 4 threads |
| Deskew (Hough) | ~3ms | Fixed overhead |
| INTER_CUBIC rotation | ~300ms | Scales with image size |
| **Total per image** | **~420ms** | Well under 1s target |

## Configuration Reference

All settings are in `app/config.py`:

| Variable | Default | Description |
|---|---|---|
| `THUMBNAIL_SIZE` | 384 | Size of square thumbnail for ONNX (model expects 384×384) |
| `ONNX_INTRA_OP_THREADS` | 4 | ONNX threads per worker (balance latency vs. parallelism) |
| `ONNX_PROB_THRESHOLD` | 0.5 | Confidence threshold for ONNX label |
| `BIG_ANGLE_THRESHOLD` | 15.0 | Degrees — above this, ONNX result takes priority |
| `SMALL_ANGLE_THRESHOLD` | 0.3 | Degrees — below this, deskew is skipped |
| `DESKEW_COEFFICIENT` | 1.0 | Multiply deskew angle (tune to taste) |
| `OUTPUT_FORMAT` | PNG | Output image format (PNG = lossless, JPEG = smaller) |
| `OUTPUT_QUALITY` | 95 | JPEG quality (only used when OUTPUT_FORMAT=JPEG) |
| `MAX_UPLOAD_SIZE_MB` | 30 | Maximum upload file size |
| `LOG_LEVEL` | INFO | Logging verbosity |

## Model Details

**DuarteBarbosa/deep-image-orientation-detection**

- Architecture: EfficientNetV2-S
- Task: 4-class image orientation classification
- Classes: 0° / 90° / 180° / 270°
- Accuracy: 98.82% on validation set
- Training data: 189,018 unique images, 756,072 augmented samples
- Format: ONNX (80.6 MB)
- License: Compatible with government / commercial use

Comparison with previous PaddlePaddle model:

| Aspect | Old (Paddle) | New (ONNX + EfficientNetV2-S) |
|---|---|---|
| Classes | 2 (0°/180°) | 4 (0°/90°/180°/270°) |
| ONNX support | Broken (shape inference error) | Native |
| 90°/270° support | Via Hough only (fragile) | Direct model prediction |
| Model loading | ~1850ms (cold) | ~200ms (cold) |
| Inference (warm) | ~35ms | ~110ms |
| Quality | High | High (OCR-safe) |
