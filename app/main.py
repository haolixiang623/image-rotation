"""
FastAPI application — government document auto-orientation & lossless repair API.

Endpoints
--------
  GET  /health              — liveness / readiness probe
  GET  /                    — service metadata
  POST /v1/image/auto-orient — image orientation correction

Deployment
----------
  Development:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  Production (Gunicorn multi-process):
    ./deployment.sh   # auto-detects CPU cores
    # or manually:
    gunicorn app.main:app \
      -w 2 \
      -k uvicorn.workers.UvicornWorker \
      --bind 0.0.0.0:8000 \
      --timeout 120 \
      --access-logfile -
"""

from __future__ import annotations

import logging
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.config import (
    BASE_DIR,
    MAX_UPLOAD_SIZE_MB,
    LOG_LEVEL,
    LOG_DIR,
    MODEL_DIR,
    APP_VERSION,
)
from app.processors.image_processor import ImageProcessor

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  [%(process)d/%(thread)d]  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("api")


# ── App lifespan — initialise ONNX session once per worker process ───────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run on every worker process startup and shutdown.

    ImageProcessor is process-global; we pre-warm it here so the first
    production request doesn't pay the model-loading latency penalty.
    """
    logger.info(
        "Worker %d starting — pre-loading ONNX model …",
        id(getattr(psutil.Process(), "pid", 0)),
    )
    t0 = time.perf_counter()
    try:
        ImageProcessor().process(b"", filename="__warmup__")
    except Exception:
        pass  # warmup with empty bytes may fail; model still loads
    logger.info("Worker warm-up complete (%.1fs).", time.perf_counter() - t0)
    yield
    logger.info("Worker shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Government Document Auto-Orientation API",
    description=(
        "High-concurrency, lossless image orientation correction pipeline for "
        "government administrative materials.  Stages: EXIF → ONNX 4-class "
        "orientation classification (0°/90°/180°/270°) → deskew small-angle "
        "detection → original-resolution INTER_CUBIC rotation."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# 允许本地 file:// 或其它端口页面调用 API（测试页）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_test_page_dir = BASE_DIR / "app" / "static" / "test_page"
if _test_page_dir.is_dir():
    app.mount(
        "/test",
        StaticFiles(directory=str(_test_page_dir), html=True),
        name="test_ui",
    )

# Shared thread pool — limits per-worker parallelism to prevent
# a single request from exhausting CPU resources.
_thread_pool = ThreadPoolExecutor(max_workers=4)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _validate_upload(file: UploadFile) -> bytes:
    """Read and sanity-check uploaded file bytes."""
    if file.size and file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            413,
            f"File exceeds {MAX_UPLOAD_SIZE_MB} MB limit (got {file.size / 1024 / 1024:.1f} MB)",
        )

    content = file.file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty upload body.")

    # Basic magic-byte check
    img_magic = {b"\xff\xd8\xff", b"\x89PNG", b"RIFF"}
    if not any(content[:4] == m for m in [*img_magic, b"GIF", b"WEBP"]):
        logger.warning(
            "Upload '%s' has unexpected magic bytes %s",
            file.filename,
            content[:8].hex(),
        )

    return content


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["monitoring"])
async def health_check() -> JSONResponse:
    """
    Liveness / readiness probe for load balancers and orchestrators.

    Returns basic health metadata including worker PID and CPU affinity info.
    """
    proc = psutil.Process()
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "pid": proc.pid,
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_available_mb": psutil.virtual_memory().available // (1024 * 1024),
            "model_loaded": ImageProcessor._model_loaded,
            "uptime": f"{time.time() - _start_time:.0f}s",
        },
    )


@app.get("/", tags=["info"])
async def root() -> JSONResponse:
    """Service metadata — useful for debugging and API discovery."""
    return JSONResponse(
        status_code=200,
        content={
            "service": "Government Document Auto-Orientation API",
            "version": APP_VERSION,
            "endpoints": {
                "health": "GET  /health",
                "orient": "POST /v1/image/auto-orient",
            },
            "model": "DuarteBarbosa/deep-image-orientation-detection (EfficientNetV2-S, 4-class)",
            "model_loaded": ImageProcessor._model_loaded,
        },
    )


@app.post(
    "/v1/image/auto-orient",
    response_class=StreamingResponse,
    tags=["image"],
    summary="Auto-orient an uploaded image",
    responses={
        200: {
            "description": "Orientation-corrected image binary (PNG by default).",
            "content": {"image/png": {}},
        },
        400: {"description": "Empty or unsupported file."},
        413: {"description": "File too large."},
        500: {"description": "Processing error."},
    },
)
async def auto_orient(
    file: UploadFile = File(..., description="Image file (JPEG / PNG / TIFF / WebP / BMP)."),
    return_base64: bool = Query(
        False,
        description="If true, return a JSON body with base64-encoded image instead of raw binary.",
    ),
    return_metadata: bool = Query(
        True,
        description="If true, attach processing metadata to the response headers.",
    ),
):
    """
    Correct the orientation of a government document image.

    The full-resolution original image is rotated using cv2.INTER_CUBIC and
    returned as a PNG stream by default (lossless, safe for downstream OCR).
    A 384 px centre-cropped thumbnail is used for ONNX inference.

    **Concurrency note:** each Gunicorn worker handles one request at a time
    (UvicornWorker); the ONNX session is pinned to 4 threads per worker
    (`intra_op_num_threads=4`) to maximise multi-process throughput.
    """
    t_request = time.perf_counter()
    filename = file.filename or "upload"

    try:
        image_bytes = _validate_upload(file)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Upload validation error for '%s': %s", filename, exc)
        raise HTTPException(400, f"Failed to read upload: {exc}")

    logger.info("Processing '%s' (%d bytes) …", filename, len(image_bytes))

    try:
        loop = __import__("asyncio").get_running_loop()
        processor = ImageProcessor()

        def sync_process():
            return processor.process(image_bytes, filename=filename)

        result_bytes, exif_angle, onnx_angle, deskew_angle, total_angle, proc_ms = (
            await loop.run_in_executor(_thread_pool, sync_process)
        )

    except Exception as exc:
        logger.exception("Processing failed for '%s': %s", filename, exc)
        raise HTTPException(500, f"Image processing failed: {exc}")

    finally:
        del image_bytes

    request_ms = (time.perf_counter() - t_request) * 1000

    if return_metadata:
        logger.info(
            "[%s]  exif=%.1f°  onnx=%.1f°  deskew=%.2f°  total=%.2f°  "
            "proc=%.1fms  req=%.1fms",
            filename,
            exif_angle,
            onnx_angle,
            deskew_angle,
            total_angle,
            proc_ms,
            request_ms,
        )

    if return_base64:
        import base64

        b64 = base64.b64encode(result_bytes).decode("ascii")
        content_type = "image/png" if result_bytes[:4] == b"\x89PNG" else "image/jpeg"
        return JSONResponse(
            status_code=200,
            content={
                "image_base64": b64,
                "content_type": content_type,
                "metadata": {
                    "original_filename": filename,
                    "exif_correction_deg": round(exif_angle, 2),
                    "onnx_angle_deg": round(onnx_angle, 2),
                    "deskew_angle_deg": round(deskew_angle, 2),
                    "total_correction_deg": round(total_angle, 2),
                    "processing_time_ms": round(proc_ms, 1),
                    "request_time_ms": round(request_ms, 1),
                },
            },
            media_type="application/json",
        )

    media_type = "image/png" if result_bytes[:4] == b"\x89PNG" else "image/jpeg"
    headers: dict[str, str] = {
        "Content-Disposition": f'attachment; filename="oriented_{filename}"',
        "X-Orientation-Corrected-Deg": f"{total_angle:.2f}",
        "X-Processing-Time-Ms": f"{proc_ms:.1f}",
        "X-Request-Time-Ms": f"{request_ms:.1f}",
        "X-Exif-Correction-Deg": f"{exif_angle:.1f}",
        "X-ONNX-Angle-Deg": f"{onnx_angle:.1f}",
        "X-Deskew-Angle-Deg": f"{deskew_angle:.2f}",
    }

    return StreamingResponse(
        iter([result_bytes]),
        media_type=media_type,
        headers=headers,
    )


# ── Module-level start time ───────────────────────────────────────────────────
_start_time = time.time()
