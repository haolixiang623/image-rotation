# 生产环境部署指南

## 目录

1. [前提条件](#前提条件)
2. [镜像打包（在一台有网络的机器上执行）](#1-镜像打包在一台有网络的机器上执行)
3. [导出离线镜像](#2-导出离线镜像)
4. [导入生产服务器](#3-导入生产服务器)
5. [启动服务](#4-启动服务)
6. [验证](#5-验证)
7. [更新版本](#6-更新版本)

---

## 前提条件

| 项目 | 要求 |
|------|------|
| 网络机器（构建用） | 安装了 Docker，能访问 HuggingFace Hub |
| 生产服务器 | 安装了 Docker，无外网访问 |
| 镜像传输介质 | U盘 / 内网文件服务器 / 光盘等 |

---

## 1. 镜像打包（在一台有网络的机器上执行）

### 1.1 下载项目代码

```bash
git clone <项目仓库地址> /path/to/image-rotation-api
cd /path/to/image-rotation-api
```

### 1.2 构建镜像

```bash
# 标准构建（与构建机 CPU 架构一致：Mac M 系列 = arm64，x86 Linux = amd64）
docker build -t image-rotation-api:2.0.0 .

# ──────────────────────────────────────────────────────────
# 生产机是 x86_64 / CentOS 7，但你在 Apple Silicon Mac 上构建时，必须用 amd64：
#   ./build-amd64.sh
# 或手动：
#   docker buildx create --use --name amd64 2>/dev/null || docker buildx use amd64
#   docker buildx build --platform linux/amd64 -t image-rotation-api:2.0.0 --load .
# 否则容器内会出现：exec format error（二进制架构不匹配）
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
# 国内加速（如果构建机器能访问 hf-mirror.com）：
docker build \
  --build-arg HF_ENDPOINT=https://hf-mirror.com \
  -t image-rotation-api:2.0.0 .
# ──────────────────────────────────────────────────────────
```

首次构建需要下载：
- Python 基础镜像（`python:3.11-slim`）~ 130 MB
- ONNX 模型（从 HuggingFace Hub）~ 77 MB
- Python 依赖包 ~ 300 MB

预计耗时（视网络速度）：**5–20 分钟**。

### 1.3 验证镜像

```bash
docker run --rm image-rotation-api:2.0.0 \
  python -c "import fastapi,uvicorn,onnxruntime,cv2,PIL; print('包导入 OK')"

docker images image-rotation-api:2.0.0
# 确认 IMAGE SIZE ≈ 1.0–1.1 GB
```

---

## 2. 导出离线镜像

### 2.1 导出为 tar 包

```bash
docker save image-rotation-api:2.0.0 \
  -o image-rotation-api-2.0.0.tar

# 压缩（推荐，节省传输时间）
gzip -c image-rotation-api-2.0.0.tar > image-rotation-api-2.0.0.tar.gz
```

文件大小约 **600–800 MB**（压缩后）。

### 2.2 复制到生产服务器

```bash
# 方式 A：通过内网文件服务器
scp image-rotation-api-2.0.0.tar.gz user@生产服务器:/opt/images/

# 方式 B：挂载 U 盘拷贝（物理隔离网段）
```

---

## 3. 导入生产服务器

```bash
# 解压（如果传输的是压缩包）
gunzip -c image-rotation-api-2.0.0.tar.gz | docker load

# 或直接导入（未压缩的 tar）
docker load -i image-rotation-api-2.0.0.tar

# 确认导入成功
docker images image-rotation-api:2.0.0
```

---

## 4. 启动服务

### 4.1 基础启动

```bash
docker run -d \
  --name image-rotation-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /opt/logs/image-rotation-api:/app/logs \
  image-rotation-api:2.0.0
```

### 4.2 生产级启动（推荐）

```bash
docker run -d \
  --name image-rotation-api \
  --restart unless-stopped \
  --log-opt max-size=100m \
  --log-opt max-file=5 \
  -p 8000:8000 \
  -e LOG_LEVEL=INFO \
  -e WORKERS=2 \
  -v /opt/logs/image-rotation-api:/app/logs \
  -v /opt/data/input:/data/input \
  -v /opt/data/output:/data/output \
  --health-cmd="python -c 'import httpx; httpx.get(\"http://localhost:8000/health\", timeout=5).raise_for_status()'" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  --health-start-period=20s \
  image-rotation-api:2.0.0
```

### 4.3 使用 docker-compose（推荐方式）

创建 `/opt/image-rotation-api/docker-compose.yml`：

```yaml
services:
  api:
    image: image-rotation-api:2.0.0
    container_name: image-rotation-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - WORKERS=2
    volumes:
      - /opt/logs/image-rotation-api:/app/logs
    healthcheck:
      test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8000/health', timeout=5).raise_for_status()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s

networks:
  default:
    driver: bridge
```

启动：

```bash
cd /opt/image-rotation-api
docker compose up -d
```

---

## 5. 验证

```bash
# 健康检查
curl http://localhost:8000/health | python -m json.tool

# 预期输出：
# {
#     "status": "healthy",
#     "pid": <数字>,
#     "cpu_count": <数字>,
#     "model_loaded": true,
#     "uptime": "Xs"
# }

# 测试一张图片（正向，应返回 0°）
curl -s -X POST http://localhost:8000/v1/image/auto-orient \
  -F "file=@/path/to/upright.jpg;type=image/jpeg" | \
  python -c "import json,sys; d=json.load(sys.stdin); print('旋转角:', d['metadata']['total_correction_deg'], '°')"

# 预期：旋转角: 0.0 °

# 测试一张倒置图片
curl -s -X POST http://localhost:8000/v1/image/auto-orient \
  -F "file=@/path/to/rotated.jpg;type=image/jpeg" | \
  python -c "import json,sys; d=json.load(sys.stdin); print('旋转角:', d['metadata']['total_correction_deg'], '°')"

# 预期：旋转角: 90.0 ° 或 180.0 ° 或 270.0 °
```

**API 文档**（浏览器打开）：
```
http://<服务器IP>:8000/docs
```

**测试页面**：
```
http://<服务器IP>:8000/test/
```

---

## 6. 更新版本

### 6.1 在构建机器上更新

```bash
cd /path/to/image-rotation-api
git pull  # 或手动替换代码

# 重新构建
docker build -t image-rotation-api:2.1.0 .

# 导出
docker save image-rotation-api:2.1.0 | gzip -c > image-rotation-api-2.1.0.tar.gz
```

### 6.2 在生产服务器上热更新

```bash
# 方法 A：docker-compose（推荐）
docker compose pull   # 不会生效（无外网），改用 load
docker compose down
gunzip -c /path/to/new-image.tar.gz | docker load
docker compose up -d

# 方法 B：滚动更新
gunzip -c /path/to/new-image.tar.gz | docker load
docker stop image-rotation-api
docker rm image-rotation-api
# 然后执行 4.2 或 4.3 的启动命令
```

---

## 快速命令速查

```bash
# 构建
docker build -t image-rotation-api:2.0.0 .

# 导出（压缩）
docker save image-rotation-api:2.0.0 | gzip -c > image-rotation-api-2.0.0.tar.gz

# 导入
gunzip -c image-rotation-api-2.0.0.tar.gz | docker load

# 启动
docker run -d --name image-rotation-api -p 8000:8000 -v /opt/logs/image-rotation-api:/app/logs image-rotation-api:2.0.0

# 查看日志
docker logs -f image-rotation-api

# 查看健康状态
curl http://localhost:8000/health

# 停止
docker stop image-rotation-api

# 删除
docker rm image-rotation-api
```
