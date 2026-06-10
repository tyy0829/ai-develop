#!/bin/bash
# vllm-ascend patches for compatibility with vllm v0.20.x/v0.21.x
#
# 1. ModelRegistry - avoid circular import
# 2. profiling_config - avoid vllm.__version__ AttributeError during lazy loading

VLLM_ASCEND_DIR="${1:-/vllm-workspace/vllm-ascend}"

echo "[PATCH] Applying vllm-ascend fixes to ${VLLM_ASCEND_DIR}"

# 1. ModelRegistry circular import
sed -i 's|^from vllm import ModelRegistry|from vllm.model_executor.models import ModelRegistry|' \
  "${VLLM_ASCEND_DIR}/vllm_ascend/models/__init__.py" 2>/dev/null

# 2. profiling_config.py - fix __version__ access
PROF_FILE="${VLLM_ASCEND_DIR}/vllm_ascend/profiling_config.py"
if [ -f "${PROF_FILE}" ]; then
  # Add 'import os' if not present
  grep -q "^import os" "${PROF_FILE}" || sed -i '/^import contextlib/a import os' "${PROF_FILE}"

  # Replace vllm.__version__ with env-override fallback
  sed -i 's|^VLLM_VERSION = vllm.__version__$|VLLM_VERSION = os.environ.get("VLLM_VERSION") or getattr(vllm, "__version__", "unknown")|' \
    "${PROF_FILE}" 2>/dev/null
fi

echo "[PATCH] Done"
