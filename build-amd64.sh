#!/usr/bin/env bash
# =============================================================================
# 在 Apple Silicon (ARM) Mac 上构建 **CentOS / 通用 x86_64 服务器** 可用的镜像
#
# 原因: 在 M 系列 Mac 上执行 `docker build` 默认产出 linux/arm64 镜像，
#       拷到 x86_64 的 CentOS 7 上会出现:
#         exec /usr/local/bin/gunicorn: exec format error
#
# 用法:
#   ./build-amd64.sh
#   ./build-amd64.sh image-rotation-api:2.0.0
#
# 导出离线包（给 deploy.sh 用）:
#   docker save image-rotation-api:2.0.0 | gzip -c > 离线部署包/image-rotation-api-2.0.0.tar.gz
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TAG="${1:-image-rotation-api:2.0.0}"
PLATFORM="linux/amd64"
BUILDER="${BUILDX_BUILDER:-imgrot-linux-amd64}"

echo ">>> Building $TAG for $PLATFORM (x86_64 / amd64 servers)"
echo

if ! docker buildx version &>/dev/null; then
    echo "ERROR: docker buildx not available. Update Docker Desktop or install buildx plugin."
    exit 1
fi

docker buildx create --name "$BUILDER" --driver docker-container --use 2>/dev/null \
    || docker buildx use "$BUILDER"
docker buildx inspect --bootstrap &>/dev/null

docker buildx build \
    --platform "$PLATFORM" \
    -t "$TAG" \
    --load \
    -f Dockerfile \
    .

echo
echo ">>> Done. Verify: docker run --rm $TAG python -c \"import platform; print(platform.machine())\""
echo "    (inside container should show x86_64)"
echo ">>> Export: docker save $TAG | gzip -c > 离线部署包/image-rotation-api-2.0.0.tar.gz"
