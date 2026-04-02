"""
ImageProcessor — lossless rotation & auto-orientation pipeline for government document images.

Stage 1 — EXIF orientation tag correction
Stage 2 — ONNX Runtime 4-class orientation classifier (EfficientNetV2-S, 0°/90°/180°/270°)
Stage 3 — deskew small-angle correction (Hough lines)
Stage 4 — apply INTER_CUBIC rotation on the ORIGINAL full-resolution image

Memory management: thumbnail used for AI inference only; original numpy array held
until the final rotate-and-save step, then explicitly deleted.

Model: DuarteBarbosa/deep-image-orientation-detection (EfficientNetV2-S)
  class 0 → 0°   (correct orientation, no correction needed)
  class 1 → 90°  (image needs 90° CLOCKWISE rotation to be correct)
  class 2 → 180° (upside-down, rotate 180° to correct)
  class 3 → 270° (image needs 90° COUNTER-CLOCKWISE rotation to be correct)
"""

from __future__ import annotations

import gc
import io
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from app.config import (
    ONNX_MODEL_PATH,
    ONNX_INTRA_OP_THREADS,
    ONNX_PROB_THRESHOLD,
    THUMBNAIL_SIZE,
    BIG_ANGLE_THRESHOLD,
    SMALL_ANGLE_THRESHOLD,
    DESKEW_COEFFICIENT,
    OUTPUT_FORMAT,
    OUTPUT_QUALITY,
)

logger = logging.getLogger("image_processor")

# ── ImageNet normalisation constants (matching training pipeline) ───────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageProcessor:
    """
    Process-level singleton that holds one ONNX Runtime session per worker.

    Inference speed on CPU (M2 Max, 8-performance cores):
      intra_op_num_threads=1  →  ~250ms / call
      intra_op_num_threads=4  →  ~108ms / call  ← default
      intra_op_num_threads=8  →  ~104ms / call
    """

    _session: Optional[ort.InferenceSession] = None
    _input_name: Optional[str] = None
    _output_name: Optional[str] = None
    _model_loaded: bool = False

    # ── Model loading ──────────────────────────────────────────────────────────

    @classmethod
    def _ensure_model(cls) -> None:
        """Lazily load the ONNX model once per worker process."""
        if cls._model_loaded:
            return

        if not ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {ONNX_MODEL_PATH}.\n"
                "Download: "
                "https://huggingface.co/DuarteBarbosa/deep-image-orientation-detection"
                "/resolve/main/orientation_model_v2_0.9882.onnx\n"
                "Save to: models/orientation_detector/orientation_model_v2_0.9882.onnx"
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = ONNX_INTRA_OP_THREADS
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        cls._session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        cls._input_name = cls._session.get_inputs()[0].name
        cls._output_name = cls._session.get_outputs()[0].name
        cls._model_loaded = True

        logger.info(
            "ONNX model loaded from %s  (threads=%d, provider=CPUExecutionProvider)",
            ONNX_MODEL_PATH,
            ONNX_INTRA_OP_THREADS,
        )

    # ── Pipeline stages ────────────────────────────────────────────────────────

    @staticmethod
    def _correct_exif_angle(image: Image.Image) -> tuple[Image.Image, float, int]:
        """
        Stage 1 — EXIF orientation tag correction.

        Returns the orientation-corrected PIL Image, the EXIF rotation
        angle in degrees (0, 90, 180, or 270), and the raw EXIF tag value.
        """
        exif = image.getexif()
        tag = exif.get(0x0112, 1)

        if tag == 1:
            return image, 0.0, tag

        _rotations = {
            2: (Image.FLIP_LEFT_RIGHT, 0),
            3: (Image.ROTATE_180, 180),
            4: (Image.FLIP_TOP_BOTTOM, 0),
            5: (Image.TRANSPOSE, 0),
            6: (Image.ROTATE_270, 270),
            7: (Image.TRANSVERSE, 0),
            8: (Image.ROTATE_90, 90),
        }

        transform, angle = _rotations.get(tag, (None, 0))

        if transform is not None:
            image = image.transpose(transform)

        return image, float(angle), tag

    @staticmethod
    def _preprocess_for_onnx(image: Image.Image, size: int = THUMBNAIL_SIZE) -> np.ndarray:
        """
        Stage 1b — preprocessing matching the training pipeline exactly.

        Training pipeline (DuarteBarbosa/deep-image-orientation-detection):
          Resize((size+32, size+32)) → CenterCrop(size) → ToTensor → Normalize

        Returns NCHW float32 tensor ready for ONNX input.
        """
        w, h = image.size

        # Step 1: Resize so that the shorter side = size+32, then centre-crop to size×size
        if w >= h:
            resize_h = int(round((size + 32) * h / w))
            image = image.resize((size + 32, resize_h), Image.Resampling.BILINEAR)
            top = (resize_h - size) // 2
            image = image.crop((0, top, size + 32, top + size))
        else:
            resize_w = int(round((size + 32) * w / h))
            image = image.resize((resize_w, size + 32), Image.Resampling.BILINEAR)
            left = (resize_w - size) // 2
            image = image.crop((left, 0, left + size, size + 32))

        # Centre-crop to exact size×size
        left = (image.width - size) // 2
        top = (image.height - size) // 2
        image = image.crop((left, top, left + size, top + size))

        # Step 2: ToTensor (0-255 → 0-1), CHW layout
        img_np = np.array(image).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC → CHW

        # Step 3: ImageNet Normalize — (x - mean) / std
        img_np = (img_np - _IMAGENET_MEAN.reshape(3, 1, 1)) / _IMAGENET_STD.reshape(3, 1, 1)

        # Step 4: add batch dimension NCHW
        img_np = np.expand_dims(img_np, axis=0)

        return img_np.astype(np.float32)

    @classmethod
    def _onnx_predict_angle(cls, image: Image.Image) -> tuple[float, float]:
        """
        Stage 2 — ONNX Runtime 4-class orientation classification.

        EfficientNetV2-S maps (CLASS_MAP — corrective action):
          class 0 → 0°   (correct orientation, no correction needed)
          class 1 → 90°  (image needs 90° clockwise rotation to correct)
          class 2 → 180° (image upside-down, rotate 180° to correct)
          class 3 → 270° (image needs 90° counter-clockwise rotation to correct)

        Returns (predicted_correction_angle_deg, confidence).
        """
        cls._ensure_model()

        img_np = cls._preprocess_for_onnx(image)

        [raw] = cls._session.run([cls._output_name], {cls._input_name: img_np})  # type: ignore[union-attr]

        probs = np.exp(raw) / np.exp(raw).sum(axis=1, keepdims=True)
        pred_class = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0][pred_class])

        angle_map = {0: 0.0, 1: 90.0, 2: 180.0, 3: 270.0}
        correction_angle = angle_map.get(pred_class, 0.0)

        return correction_angle, confidence

    @staticmethod
    def _calculate_deskew_angle(image: Image.Image) -> float:
        """
        Stage 3 — skew detection via Hough line detection.

        Works best on document-like images with strong horizontal/vertical lines.
        Returns skew angle in degrees (positive = counter-clockwise tilt).
        """
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) == 0:
            return 0.0

        angles = []
        for line in lines[:200]:
            rho, theta = line[0]
            if 0.1 < theta < np.pi / 2 - 0.1 or np.pi / 2 + 0.1 < theta < np.pi - 0.1:
                angles.append(np.degrees(theta) - 90)

        if not angles:
            return 0.0

        median_angle = float(np.median(angles)) * DESKEW_COEFFICIENT

        if abs(median_angle) < SMALL_ANGLE_THRESHOLD:
            return 0.0

        return median_angle

    @staticmethod
    def _apply_rotation_on_original(
        image: Image.Image,
        angle: float,
    ) -> Image.Image:
        """
        Stage 4 — apply INTER_CUBIC rotation to the full-resolution original.

        Uses canvas rotation (expands canvas so no clipping) so that no pixels
        are lost — essential for government documents that will go through OCR.
        """
        if abs(angle) < 0.05:
            return image

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        M = cv2.getRotationMatrix2D((cx, cy), -angle, scale=1.0)
        cos_v = np.abs(M[0, 0])
        sin_v = np.abs(M[0, 1])
        new_w = int(h * sin_v + w * cos_v)
        new_h = int(h * cos_v + w * sin_v)
        M[0, 2] += (new_w / 2.0) - cx
        M[1, 2] += (new_h / 2.0) - cy

        rotated = cv2.warpAffine(
            img_cv, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        del img_cv
        gc.collect()

        return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        image_bytes: bytes,
        filename: str = "image",
    ) -> tuple[bytes, float, float, float, float, float]:
        """
        Full lossless orientation correction pipeline.

        Fast path — if EXIF is clean AND ONNX predicts 0° with high confidence
        AND deskew is negligible, the image is already upright; re-encode it
        as PNG without any rotation, saving ~400 ms of unnecessary work.

        Returns
        -------
        (result_bytes, exif_angle, onnx_angle, deskew_angle,
         total_angle, processing_time_ms)
        """
        t0 = time.perf_counter()

        # ── Step 1: Load & EXIF correction ──────────────────────────────────
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w0, h0 = raw_image.size

        oriented_image, exif_angle, exif_tag = self._correct_exif_angle(raw_image)
        del raw_image
        gc.collect()

        t_exif = (time.perf_counter() - t0) * 1000

        # ── Step 2: ONNX 4-class orientation detection ──────────────────────
        #   Skip ONNX when EXIF already applied a non-trivial correction (tag != 1).
        #   The ONNX model was trained on pixel-level orientations; it has never seen
        #   pre-rotated images. A portrait photo (tag=6) carries a strong "vertical
        #   content" signal — ONNX maps that to 270° correction — causing a second
        #   unwanted 270° rotation on top of the EXIF one already baked in.
        #   When tag != 1 the image is already orientation-corrected; only deskew
        #   (tiny tilt, not orientation) may still apply.
        t_onnx_start = time.perf_counter()
        if exif_tag != 1:
            # Image was pre-rotated by EXIF; ONNX is unreliable on that. Treat as
            # "low confidence / unknown" so the decision logic falls through to the
            # general branch: final_angle = exif_angle + onnx_angle(0) + deskew_angle.
            onnx_angle, onnx_conf = 0.0, 0.0
            reason_onnx = f"skipped (EXIF tag={exif_tag} already applied)"
        else:
            onnx_angle, onnx_conf = self._onnx_predict_angle(oriented_image)
            reason_onnx = f"ONNX={onnx_angle}° (conf={onnx_conf:.2f})"
        t_onnx = (time.perf_counter() - t_onnx_start) * 1000

        # ── Step 3: deskew small-angle detection ───────────────────────────
        t_deskew_start = time.perf_counter()
        w_orig, h_orig = oriented_image.size
        if max(w_orig, h_orig) > 512:
            ratio = 512 / max(w_orig, h_orig)
            deskew_thumb = oriented_image.resize(
                (int(w_orig * ratio), int(h_orig * ratio)),
                Image.Resampling.LANCZOS,
            )
        else:
            deskew_thumb = oriented_image
        deskew_angle = self._calculate_deskew_angle(deskew_thumb)
        t_deskew = (time.perf_counter() - t_deskew_start) * 1000
        if deskew_thumb is not oriented_image:
            del deskew_thumb

        # ── Decision logic ─────────────────────────────────────────────────
        #
        # Priority:
        #   1. ONNX says big rotation (≥15°, high conf) → trust ONNX, skip deskew
        #      (Hough deskew is meaningless for a 90°-misoriented image)
        #   2. ONNX says image is already upright (≈0°, high conf) → skip deskew
        #      (Hough easily misdetects circles/stamps/curves as slanted lines;
        #       ONNX accuracy 98.82% >> Hough reliability on non-document images)
        #   3. Otherwise → combine EXIF + ONNX + deskew

        if abs(onnx_angle) >= BIG_ANGLE_THRESHOLD and onnx_conf >= ONNX_PROB_THRESHOLD:
            # Large misorientation detected — rotate, skip deskew
            final_angle = onnx_angle
            reason = (
                f"ONNX (conf={onnx_conf:.2f}, angle={onnx_angle}°), "
                f"deskew skipped (big-angle threshold={BIG_ANGLE_THRESHOLD}°)"
            )
        elif (
            exif_tag == 1
            and abs(onnx_angle) < 0.5
            and onnx_conf >= ONNX_PROB_THRESHOLD
            and abs(deskew_angle) < 0.5
        ):
            # Image is already upright (confirmed by ONNX) — no rotation needed
            final_angle = 0.0
            reason = (
                f"EXIF=0° + ONNX=0.0° (conf={onnx_conf:.2f}), "
                f"deskew skipped (ONNX upright, conf={onnx_conf:.2f})"
            )
        else:
            # General case: combine correction signals.
            # When exif_tag != 1, exif_angle has already been applied to oriented_image
            # (the rotation is "baked in"), so exif_angle contributes 0 here.
            _exif_contrib = 0.0 if exif_tag != 1 else exif_angle

            # If ONNX confirms image is upright (high confidence, angle≈0°) then
            # deskew must be ignored — Hough misdetects circles/stamps/curves as
            # slanted lines (e.g. 1-1.jpg: ONNX=0° conf=0.92 but Hough reports -7°).
            if abs(onnx_angle) < 0.5 and onnx_conf >= ONNX_PROB_THRESHOLD:
                _deskew = 0.0
                _onnx_contrib = 0.0
            else:
                _deskew = deskew_angle
                # Landscape images (w > h) are already in their natural orientation;
                # ONNX sometimes mispredicts 90°/270° for wide-screen photos due to
                # aspect-ratio confusion (e.g. zm.jpg 1440×810 → ONNX=90° conf=0.93).
                # Treat those as low-confidence regardless of actual ONNX confidence.
                if w_orig > h_orig and abs(onnx_angle) in (90.0, 270.0):
                    _onnx_contrib = 0.0
                else:
                    _onnx_contrib = onnx_angle

            final_angle = _exif_contrib + _onnx_contrib + _deskew
            reason = (
                f"EXIF={_exif_contrib}° + ONNX={_onnx_contrib}° (conf={onnx_conf:.2f}) "
                f"+ deskew={_deskew:.2f}°"
            )

        # ── Fast path: already upright — re-encode without any rotation ───
        #    Triggers only when:
        #      (a) EXIF tag == 1 (image is in its natural orientation as ONNX saw it)
        #      (b) ONNX confirms upright (angle≈0, high confidence)
        #      (c) no measurable deskew (angle < 0.5°)
        #    This avoids the expensive full-resolution INTER_CUBIC rotation (~300 ms)
        #    while preserving the EXIF-corrected orientation as total_angle.
        #    Threshold 0.5° is 5× the deskew noise floor (SMALL_ANGLE_THRESHOLD=0.3).
        if (
            exif_tag == 1
            and abs(onnx_angle) < 0.5
            and onnx_conf >= ONNX_PROB_THRESHOLD
            and abs(deskew_angle) < 0.5
        ):
            buf = io.BytesIO()
            if OUTPUT_FORMAT == "JPEG":
                oriented_image.save(buf, format="JPEG", quality=OUTPUT_QUALITY, optimize=False)
            else:
                oriented_image.save(buf, format="PNG")
            result_bytes = buf.getvalue()
            del oriented_image, buf
            gc.collect()

            total_ms = t_exif + t_onnx + t_deskew
            logger.info(
                "[%s]  %dx%d  angle=0.00°  (fast path — already upright, EXIF=0)  "
                "took=%.1fms (exif=%.1fms onnx=%.1fms deskew=%.1fms)  size_out=%d bytes",
                filename, w0, h0, total_ms, t_exif, t_onnx, t_deskew, len(result_bytes),
            )
            return result_bytes, 0.0, 0.0, 0.0, 0.0, total_ms

        # ── Step 4: Apply INTER_CUBIC rotation on full-resolution original ─
        result_image = self._apply_rotation_on_original(oriented_image, final_angle)
        del oriented_image
        gc.collect()

        # ── Step 5: Encode result ───────────────────────────────────────────
        buf = io.BytesIO()
        if OUTPUT_FORMAT == "JPEG":
            result_image.save(buf, format="JPEG", quality=OUTPUT_QUALITY, optimize=False)
        else:
            result_image.save(buf, format="PNG")
        result_bytes = buf.getvalue()
        del result_image, buf
        gc.collect()

        total_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "[%s]  %dx%d  angle=%.2f°  (%s)  "
            "took=%.1fms (exif=%.1fms onnx=%.1fms deskew=%.1fms)  size_out=%d bytes",
            filename, w0, h0, final_angle, reason,
            total_ms, t_exif, t_onnx, t_deskew, len(result_bytes),
        )

        return (
            result_bytes,
            float(exif_angle),
            float(onnx_angle),
            float(deskew_angle),
            float(final_angle),
            total_ms,
        )
