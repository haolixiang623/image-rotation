"""
ImageProcessor — lossless rotation & auto-orientation pipeline for government document images.

Stage 1 — EXIF orientation tag correction
Stage 2 — ONNX Runtime 4-class orientation classifier (EfficientNetV2-S, 0°/90°/180°/270°)
Stage 3 — apply INTER_CUBIC cardinal rotation on the ORIGINAL full-resolution image
        (Hough deskew disabled — small-angle tilt skipped)

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

        Special case: tag=6 on a landscape image is treated as "no rotation".
        EXIF=6 means the camera was held vertically, but when a document/card
        is already placed horizontally in the frame (landscape w > h), the
        content is already in its natural orientation — applying 270° would
        turn a correct landscape card into a wrong portrait orientation.
        """
        exif = image.getexif()
        tag = exif.get(0x0112, 1)
        w, h = image.size

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

        # Tag 6 (camera rotated 90° CW) on a landscape image means the document
        # is already horizontally placed — applying EXIF=6 would make it portrait.
        if tag == 6 and w > h:
            return image, 0.0, 1   # pretend EXIF=1, no rotation applied

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
    def _apply_rotation_on_original(
        image: Image.Image,
        angle: float,
    ) -> Image.Image:
        """
        Apply INTER_CUBIC rotation to the full-resolution original.

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

        Fast path — if EXIF is clean AND ONNX predicts 0° with high confidence,
        the image is already upright; re-encode without rotation.

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

        # ── Step 3: deskew (disabled) ───────────────────────────────────────
        # Hough deskew is not applied — only cardinal ONNX rotations.  Small-angle
        # correction is skipped per product requirement.
        w_orig, h_orig = oriented_image.size
        deskew_angle = 0.0
        t_deskew = 0.0

        # ── Decision logic ─────────────────────────────────────────────────
        #
        # EXIF corrects camera orientation tags first.  ONNX supplies 0°/90°/180°/270°
        # correction angles.  No Hough deskew — only cardinal rotations.

        # Fast re-encode: confirmed upright — skip all processing.
        if (
            exif_tag == 1
            and abs(onnx_angle) < 0.5
            and onnx_conf >= ONNX_PROB_THRESHOLD
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
                "[%s]  %dx%d  angle=0.00°  (fast path — upright, EXIF=0)  "
                "took=%.1fms (exif=%.1fms onnx=%.1fms deskew=%.1fms)  size_out=%d bytes",
                filename, w0, h0, total_ms, t_exif, t_onnx, t_deskew, len(result_bytes),
            )
            return result_bytes, 0.0, 0.0, 0.0, 0.0, total_ms

        # Aspect ratio: used in multiple checks below.
        ar = w_orig / h_orig  # >1 = landscape, <1 = portrait

        # ── Path B: Near-square — only skip when ONNX says upright ───────────
        #    If |ar−1| < 0.08 but ONNX predicts 90°/180°/270° with confidence,
        #    apply that cardinal rotation (e.g. 5.jpg 800×799 needs 270°).
        if abs(ar - 1.0) < 0.08:
            if (
                exif_tag == 1
                and abs(onnx_angle) < 0.5
                and onnx_conf >= ONNX_PROB_THRESHOLD
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
                    "[%s]  %dx%d  angle=0.00°  "
                    "(near-square ar=%.2f, ONNX=0° upright)  "
                    "took=%.1fms (exif=%.1fms onnx=%.1fms deskew=%.1fms)  size_out=%d bytes",
                    filename, w0, h0, ar, total_ms, t_exif, t_onnx, t_deskew, len(result_bytes),
                )
                return result_bytes, 0.0, float(onnx_angle), 0.0, 0.0, total_ms
            # else: fall through — near-square but ONNX wants a cardinal rotation

        # ── Path C: EXIF already applied in Step 1 ───────────────────────────
        #    Must be plain `if`, not `elif` after near-square `if` — otherwise
        #    near-square + cardinal ONNX never reaches Path D (Python skips
        #    `elif`/`else` after a matching outer `if`).
        if exif_tag != 1:
            _exif_contrib = 0.0  # already baked into oriented_image
            final_angle = _exif_contrib
            reason = f"EXIF tag applied (exif_angle={exif_angle}°), no deskew"
            # Fall through to Step 4 — usually final_angle == 0.

        # ── Path D: Cardinal ONNX rotation (incl. near-square + exif=1) ─────
        #    Trust 90°/180°/270° when confidence ≥ threshold.  No deskew.
        else:
            final_angle = 0.0

            if abs(onnx_angle - 180.0) < 0.5 and onnx_conf >= ONNX_PROB_THRESHOLD:
                final_angle = 180.0
                reason = f"ONNX=180° conf={onnx_conf:.2f} (倒置修正)"
            elif abs(onnx_angle - 270.0) < 0.5 and onnx_conf >= ONNX_PROB_THRESHOLD:
                final_angle = 270.0
                reason = f"ONNX=270° conf={onnx_conf:.2f} (逆时针90°)"
            elif abs(onnx_angle - 90.0) < 0.5 and onnx_conf >= ONNX_PROB_THRESHOLD:
                final_angle = 90.0
                reason = f"ONNX=90° conf={onnx_conf:.2f} (顺时针90°)"
            else:
                reason = (
                    f"ONNX suppressed (onnx={onnx_angle}° conf={onnx_conf:.2f})"
                )

            if abs(final_angle) < 0.05:
                # No correction needed — re-encode without rotation.
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
                    "[%s]  %dx%d  angle=0.00°  "
                    "(normal-AR, %s)  "
                    "took=%.1fms (exif=%.1fms onnx=%.1fms deskew=%.1fms)  size_out=%d bytes",
                    filename, w0, h0, reason,
                    total_ms, t_exif, t_onnx, t_deskew, len(result_bytes),
                )
                return result_bytes, 0.0, float(onnx_angle), float(deskew_angle), 0.0, total_ms

            # Fall through to Step 4 — apply rotation.

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
