# ============================================================
# 离线镜像 — Government Document Auto-Orientation API
#
# 构建命令（本机架构，如 Apple Silicon = arm64）:
#   docker build -t image-rotation-api:2.0.0 .
#
# 部署到 x86_64 / CentOS 7 等服务器（必须 amd64 镜像）:
#   ./build-amd64.sh
#   或:
#   docker buildx create --use --name amd64 2>/dev/null || docker buildx use amd64
#   docker buildx build --platform linux/amd64 -t image-rotation-api:2.0.0 --load .
#
# 若在 ARM Mac 上只跑 `docker build`，拷到 x86 机会出现:
#   exec /usr/local/bin/gunicorn: exec format error
#
# 镜像大小预估: ~1.8 GB（Python + ONNX Runtime + OpenCV + 模型）
#
# ============================================================

FROM python:3.11-slim

LABEL maintainer="dev-team"
LABEL version="2.0.0"
LABEL description="Government Document Auto-Orientation API — lossless image rotation"

# ── 系统依赖 ──────────────────────────────────────────────────
# opencv-python-headless 不需要 GUI/GL 库；libgomp1 为 OpenMP（onnxruntime 并行）必需
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依赖 ──────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages \
    fastapi==0.128.0 \
    uvicorn[standard]==0.32.1 \
    gunicorn==23.0.0 \
    "onnxruntime>=1.21.0" \
    opencv-python-headless==4.13.0.92 \
    Pillow==11.1.0 \
    numpy>=2.0.0 \
    python-multipart==0.0.20 \
    pydantic==2.10.6 \
    pydantic-settings==2.7.1 \
    psutil==6.1.1 \
    httpx==0.28.1

# ── 复制应用代码和模型 ───────────────────────────────────────
#   models/orientation_detector/orientation_model_v2_0.9882.onnx
#   已在构建上下文中有，直接 COPY 即可，无需网络
COPY app/ ./app/
COPY models/orientation_detector/ ./models/orientation_detector/

# ── 创建运行时目录 ───────────────────────────────────────────
RUN mkdir -p /app/logs && \
    chown -R nobody:nogroup /app && \
    chmod 755 /app/logs

# ── 健康检查 ──────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5).raise_for_status()"

# ── 环境变量 ─────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# ── 非 root 用户（安全）───────────────────────────────────────
USER nobody:nogroup
WORKDIR /app

# ── 启动命令 ──────────────────────────────────────────────────
EXPOSE 8000

# 生产模式（Gunicorn 多进程）
CMD ["gunicorn", "app.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-"]
