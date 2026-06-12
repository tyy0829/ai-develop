#!/bin/bash
# Version-gated vllm 0.20.x circular import patches
# Usage: bash /root/.config/opencode/skills/deploy-vllm-on-ascend/references/apply-vllm-v0.20.x-patches.sh

set -euo pipefail

VLLM_DIR="/vllm-workspace/vllm"
VLLM_ASCEND_DIR="/vllm-workspace/vllm-ascend"

# Detect version
VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")

case "$VLLM_VERSION" in
  0.20.*)
    echo "Applying circular import patches for vllm $VLLM_VERSION ..."
    ;;
  0.21.*|0.22.*|0.23.*)
    echo "vllm $VLLM_VERSION: these patches were for 0.20.x. Checking if still needed..."
    # Try importing first to see if the issue exists
    if python3 -c "from vllm import SamplingParams" 2>/dev/null; then
      echo "Import works fine. No patches needed."
      exit 0
    fi
    echo "Import still broken, applying patches anyway..."
    ;;
  *)
    echo "vllm version '$VLLM_VERSION' is unknown. Skipping patches (may not be needed)."
    echo "If you hit ImportError later, run this script again after fixing vllm."
    exit 0
    ;;
esac

# --- vllm patches ---
cd "$VLLM_DIR"

echo "Patching SamplingParams imports..."
for f in vllm/v1/sample/logits_processor/builtin.py \
         vllm/v1/sample/logits_processor/interface.py \
         vllm/v1/worker/gpu/warmup.py; do
  if [ -f "$f" ] && grep -q '^from vllm import SamplingParams$' "$f"; then
    sed -i 's|^from vllm import SamplingParams$|from vllm.sampling_params import SamplingParams|' "$f"
    echo "  patched: $f"
  fi
done

echo "Patching PoolingParams imports..."
for f in vllm/entrypoints/pooling/classify/protocol.py \
         vllm/entrypoints/pooling/embed/io_processor.py \
         vllm/entrypoints/pooling/embed/protocol.py \
         vllm/entrypoints/pooling/pooling/protocol.py \
         vllm/entrypoints/pooling/scoring/protocol.py \
         vllm/entrypoints/pooling/scoring/serving.py; do
  if [ -f "$f" ] && grep -q '^from vllm import PoolingParams$' "$f"; then
    sed -i 's|^from vllm import PoolingParams$|from vllm.pooling_params import PoolingParams|' "$f"
    echo "  patched: $f"
  fi
done

# --- vllm-ascend patches ---
cd "$VLLM_ASCEND_DIR"

echo "Patching ModelRegistry import..."
if [ -f vllm_ascend/models/__init__.py ] && grep -q '^from vllm import ModelRegistry' vllm_ascend/models/__init__.py; then
  sed -i 's|^from vllm import ModelRegistry|from vllm.model_executor.models import ModelRegistry|' \
    vllm_ascend/models/__init__.py
  echo "  patched: vllm_ascend/models/__init__.py"
fi

echo "All patches applied successfully for vllm $VLLM_VERSION."
