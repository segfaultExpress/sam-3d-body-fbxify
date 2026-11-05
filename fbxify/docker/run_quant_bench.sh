#!/usr/bin/env bash
# Run the quantized-vs-fp32 x normal-vs-fast benchmark matrix in Docker, then the comparison.
#
# Usage (from repo root):
#   HF_TOKEN=hf_xxx ./fbxify/docker/run_quant_bench.sh            # all 4 configs + compare
#   ./fbxify/docker/run_quant_bench.sh fp32-normal fp32-fast      # subset
#   QUANT_MODE=pt2 ./fbxify/docker/run_quant_bench.sh             # torch.export instead of TensorRT
#
# Env: IMAGE (default mordommin94/fbxify-worker:quantized), QUANT_MODE (trt|pt2), FRAMES (20),
#      BENCH_DIR (workspace/quant_bench), HF_TOKEN (needed once to download embedl files),
#      BUILD=1 to (re)build the image first.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-mordommin94/fbxify-worker:quantized}"
QUANT_MODE="${QUANT_MODE:-trt}"
FRAMES="${FRAMES:-20}"
BENCH_DIR="${BENCH_DIR:-workspace/quant_bench}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/cache}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$REPO_DIR/checkpoints}"

if [ "${BUILD:-0}" = "1" ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building $IMAGE..."
  docker build -f "$REPO_DIR/fbxify/docker/Dockerfile.quantized" -t "$IMAGE" "$REPO_DIR"
fi

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(fp32-normal fp32-fast "${QUANT_MODE}-normal" "${QUANT_MODE}-fast")
fi

run_in_docker() {
  docker run --rm --gpus all --shm-size=8g \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e CHECKPOINTS_DIR=/fbxify/checkpoints \
    -e CACHE_DIR=/fbxify/cache \
    -v "$REPO_DIR":/fbxify \
    -v "$CHECKPOINTS_DIR":/fbxify/checkpoints \
    -v "$CACHE_DIR":/fbxify/cache \
    -w /fbxify \
    "$IMAGE" "$@"
}

for cfg in "${CONFIGS[@]}"; do
  backbone="${cfg%-*}"
  mode="${cfg##*-}"
  extra=()
  [ "$mode" = "fast" ] && extra+=(--fast_inference)
  echo "=== $cfg (backbone=$backbone, $mode) ==="
  run_in_docker python -m fbxify.cli_bench_quantized \
    --backbone "$backbone" --frames "$FRAMES" --output_dir "$BENCH_DIR" --config_name "$cfg" "${extra[@]}" \
    2>&1 | tee "$REPO_DIR/$BENCH_DIR/$cfg.log" || true
done

echo "=== compare ==="
run_in_docker python -m fbxify.cli_compare_quantized --bench_dir "$BENCH_DIR" --baseline fp32-normal
echo "Results: $REPO_DIR/$BENCH_DIR/compare/summary.json"
