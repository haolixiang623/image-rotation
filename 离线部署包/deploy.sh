#!/bin/bash
# ==============================================================================
# 一键部署脚本 — 图片旋转 API 离线安装
# ==============================================================================
# 用法: ./deploy.sh <镜像包路径>
# 示例: ./deploy.sh /mnt/usb/image-rotation-api-2.0.0.tar.gz
# ==============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
IMAGE_NAME="image-rotation-api"
IMAGE_TAG="2.0.0"
CONTAINER_NAME="image-rotation-api"
APP_DIR="/opt/image-rotation-api"
LOG_DIR="/opt/logs/image-rotation-api"
DATA_INPUT_DIR="/opt/data/input"
DATA_OUTPUT_DIR="/opt/data/output"

echo_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
echo_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
echo_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ==============================================================================
# 检查参数
# ==============================================================================
if [ -z "$1" ]; then
    echo "用法: $0 <镜像包路径>"
    echo ""
    echo "示例:"
    echo "  $0 /mnt/usb/image-rotation-api-2.0.0.tar.gz"
    echo "  $0 ./image-rotation-api-2.0.0.tar.gz"
    echo ""
    echo "提示: 如果不带参数运行，脚本会引导你完成部署"
    echo ""
    exit 1
fi

IMAGE_PACKAGE="$1"

# ==============================================================================
# 检查 Docker
# ==============================================================================
echo_info "检查 Docker 环境..."
if ! command -v docker &> /dev/null; then
    echo_error "Docker 未安装，请先安装 Docker"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo_error "Docker 未运行，请先启动 Docker"
    exit 1
fi

echo_success "Docker 环境正常"

# ==============================================================================
# 检查镜像包
# ==============================================================================
echo_info "检查镜像包: $IMAGE_PACKAGE"
if [ ! -f "$IMAGE_PACKAGE" ]; then
    echo_error "镜像包不存在: $IMAGE_PACKAGE"
    exit 1
fi

FILE_SIZE=$(du -h "$IMAGE_PACKAGE" | cut -f1)
echo_success "镜像包大小: $FILE_SIZE"

# ==============================================================================
# 创建目录
# ==============================================================================
echo_info "创建必要目录..."
mkdir -p "$APP_DIR" "$LOG_DIR" "$DATA_INPUT_DIR" "$DATA_OUTPUT_DIR"
echo_success "目录创建完成"

# ==============================================================================
# 导入镜像
# ==============================================================================
echo_info "导入镜像（请稍候...）..."
if ! gunzip -c "$IMAGE_PACKAGE" | docker load; then
    echo_error "镜像导入失败"
    exit 1
fi
echo_success "镜像导入成功"

# ==============================================================================
# 检查镜像
# ==============================================================================
echo_info "确认镜像..."
if docker images "$IMAGE_NAME:$IMAGE_TAG" | grep -q "$IMAGE_TAG"; then
    echo_success "镜像已就绪: $IMAGE_NAME:$IMAGE_TAG"
else
    echo_error "镜像导入后未找到，请检查"
    exit 1
fi

# ── 架构预检（避免 arm64 镜像在 x86_64 上出现 exec format error）──────────
IMG_ARCH=$(docker image inspect "$IMAGE_NAME:$IMAGE_TAG" --format '{{.Architecture}}' 2>/dev/null || echo "")
HOST_ARCH=$(uname -m 2>/dev/null || echo "")
NEED=""
case "$HOST_ARCH" in
    x86_64|amd64) NEED="amd64" ;;
    aarch64|arm64) NEED="arm64" ;;
esac
if [[ -n "$NEED" && -n "$IMG_ARCH" && "$IMG_ARCH" != "$NEED" ]]; then
    echo_error "镜像 CPU 架构与当前主机不一致，无法运行。"
    echo_error "  镜像架构: $IMG_ARCH   本机需要: $NEED (uname -m=$HOST_ARCH)"
    echo_error "  常见原因: 在 Apple Silicon Mac 上构建了 arm64 镜像，却部署到 x86_64 CentOS。"
    echo_error "  解决: 在构建机上执行 linux/amd64 构建后重新打包，例如:"
    echo_error "    ./build-amd64.sh   # 项目根目录"
    echo_error "    docker save image-rotation-api:2.0.0 | gzip -c > image-rotation-api-2.0.0.tar.gz"
    exit 1
fi

# ==============================================================================
# 停止旧容器（如果存在）
# ==============================================================================
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo_warning "发现旧容器，正在停止..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    echo_success "旧容器已清理"
fi

# ==============================================================================
# 启动容器
# ==============================================================================
echo_info "启动服务..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --log-opt max-size=100m \
    --log-opt max-file=5 \
    -u root \
    -p 8000:8000 \
    -e LOG_LEVEL=INFO \
    -e WORKERS=2 \
    -v "$LOG_DIR:/app/logs" \
    -v "$DATA_INPUT_DIR:/data/input" \
    -v "$DATA_OUTPUT_DIR:/data/output" \
    --health-cmd="python -c 'import httpx; httpx.get(\"http://localhost:8000/health\", timeout=5).raise_for_status()'" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    --health-start-period=20s \
    "$IMAGE_NAME:$IMAGE_TAG"

echo_success "容器已启动"

# ==============================================================================
# 等待服务就绪
# ==============================================================================
echo_info "等待服务就绪（最多 60 秒）..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health &>/dev/null; then
        echo_success "服务已就绪"
        break
    fi
    if [ $i -eq 60 ]; then
        echo_error "服务启动超时，请检查容器日志: docker logs $CONTAINER_NAME"
        exit 1
    fi
    sleep 1
done

# ==============================================================================
# 验证部署
# ==============================================================================
echo_info "验证部署..."
HEALTH=$(curl -s http://localhost:8000/health | python -c "import json,sys; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")

if [ "$HEALTH" = "healthy" ]; then
    echo_success "部署成功！"
else
    echo_warning "服务响应异常，请检查: docker logs $CONTAINER_NAME"
fi

# ==============================================================================
# 输出信息
# ==============================================================================
echo ""
echo "================================================================================"
echo -e "${GREEN}  部署完成！${NC}"
echo "================================================================================"
echo ""
echo "服务地址:"
echo "  - API:        http://localhost:8000"
echo "  - 健康检查:   http://localhost:8000/health"
echo "  - API 文档:   http://localhost:8000/docs"
echo "  - 测试页面:   http://localhost:8000/test/"
echo ""
echo "常用命令:"
echo "  - 查看日志:   docker logs -f $CONTAINER_NAME"
echo "  - 重启服务:   docker restart $CONTAINER_NAME"
echo "  - 停止服务:   docker stop $CONTAINER_NAME"
echo "  - 删除容器:   docker rm $CONTAINER_NAME"
echo ""
echo "目录说明:"
echo "  - 日志目录:   $LOG_DIR"
echo "  - 输入目录:   $DATA_INPUT_DIR"
echo "  - 输出目录:   $DATA_OUTPUT_DIR"
echo "  - 应用配置:   $APP_DIR"
echo ""
echo "================================================================================"

exit 0