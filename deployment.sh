#!/usr/bin/env bash
# =============================================================================
# deployment.sh — Production launcher for the government document auto-orientation API.
#
# Usage:
#   ./deployment.sh [--workers N] [--host HOST] [--port PORT]
#
# Environment variables:
#   WORKERS    Override the auto-detected worker count (default: 2 * CPU_CORES + 1)
#   LOG_LEVEL  Logging level: DEBUG | INFO | WARNING | ERROR (default: INFO)
#   BIND       Host:port to bind (default: 0.0.0.0:8000)
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*" >&2; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
section() { echo -e "\n${CYAN}${BOLD}━━━ $* ━━━${NC}"; }

# ── Parse CLI flags ────────────────────────────────────────────────────────────
WORKERS_ARG=""
HOST_ARG="0.0.0.0"
PORT_ARG="8000"

while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--workers) WORKERS_ARG="$2"; shift 2 ;;
        -h|--host)    HOST_ARG="$2";    shift 2 ;;
        -p|--port)    PORT_ARG="$2";    shift 2 ;;
        -*)            error "Unknown flag: $1" ;;
        *)             break ;;
    esac
done

BIND="${HOST_ARG}:${PORT_ARG}"

# ── Detect hardware ───────────────────────────────────────────────────────────
CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
CPU_PHYSICAL=$(sysctl -n hw.physicalcpu 2>/dev/null || echo "$CPU_CORES")

if [[ -n "${WORKERS_ARG:-}" ]]; then
    WORKERS="$WORKERS_ARG"
else
    WORKERS=$((2 * CPU_PHYSICAL + 1))
fi

MEM_TOTAL_GB=$(sysctl -n hw.memsize 2>/dev/null || cat /proc/meminfo 2>/dev/null | awk '/MemTotal/{print $2}')
if [[ -n "${MEM_TOTAL_GB:-}" ]]; then
    MEM_TOTAL_GB=$((MEM_TOTAL_GB / 1024 / 1024 / 1024))
fi

LOG_LEVEL="${LOG_LEVEL:-INFO}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info "Detected hardware:"
info "  CPU logical cores : $CPU_CORES"
info "  CPU physical cores: $CPU_PHYSICAL"
info "  RAM                : ${MEM_TOTAL_GB:-?} GB"
info "  Recommended workers: $WORKERS (formula: 2 × physical_cores + 1)"
echo

# ── Validate prerequisites ────────────────────────────────────────────────────
section "Prerequisites"

command -v gunicorn &>/dev/null || error "gunicorn not found — run: pip install gunicorn"
command -v "$PYTHON_BIN" &>/dev/null || error "Python not found at: $PYTHON_BIN"
info "Python:     $($PYTHON_BIN --version)"
info "Gunicorn:   $(gunicorn --version 2>&1 | head -1)"

# Check model
MODEL_PATH="${SCRIPT_DIR}/models/orientation_detector/orientation_model_v2_0.9882.onnx"
if [[ -f "$MODEL_PATH" ]]; then
    MODEL_SIZE_MB=$(du -h "$MODEL_PATH" | cut -f1)
    info "ONNX model : $MODEL_PATH  (${MODEL_SIZE_MB})"
else
    warn "ONNX model not found at: $MODEL_PATH"
    warn "  Download: https://huggingface.co/DuarteBarbosa/deep-image-orientation-detection"
    warn "            /resolve/main/orientation_model_v2_0.9882.onnx"
    warn "  Save to: ./models/orientation_detector/orientation_model_v2_0.9882.onnx"
fi

# ── Log directory ─────────────────────────────────────────────────────────────
mkdir -p "${SCRIPT_DIR}/logs"
LOG_FILE="${SCRIPT_DIR}/logs/gunicorn.log"
ACCESS_LOG="${SCRIPT_DIR}/logs/access.log"
ERROR_LOG="${SCRIPT_DIR}/logs/error.log"
info "Log dir   : ${SCRIPT_DIR}/logs/"

# ── Build Gunicorn command ────────────────────────────────────────────────────
section "Launching Gunicorn"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export LOG_LEVEL
export WORKERS

GUNICORN_CMD=(
    gunicorn
    app.main:app
    --workers          "$WORKERS"
    --worker-class     uvicorn.workers.UvicornWorker
    --bind             "$BIND"
    --timeout          120
    --keep-alive       5
    --max-requests     2048
    --max-requests-jitter 128
    --access-logfile   "$ACCESS_LOG"
    --error-logfile    "$ERROR_LOG"
    --log-file         "$LOG_FILE"
    --log-level        "${LOG_LEVEL,,}"
    --capture-output
)

info "Command: ${GUNICORN_CMD[*]}"
echo
echo -e "${GREEN}${BOLD}Server starting on http://${BIND}${NC}"
echo -e "  Health check : ${GREEN}GET  http://${BIND}/health${NC}"
echo -e "  Orient API   : ${GREEN}POST http://${BIND}/v1/image/auto-orient${NC}"
echo
echo "Press Ctrl+C to stop."
echo

exec "${GUNICORN_CMD[@]}"
