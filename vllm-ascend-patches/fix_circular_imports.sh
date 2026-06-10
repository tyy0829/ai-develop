#!/bin/bash
# Circular import patches for vllm v0.20.x (and v0.21.x)
# vllm uses lazy __getattr__ in __init__.py which breaks imports during plugin registration
#
# Apply BEFORE running pip install -e . in /vllm-workspace/vllm

VLLM_DIR="${1:-/vllm-workspace/vllm}"

echo "[PATCH] Applying circular import fixes to ${VLLM_DIR}"

# SamplingParams
sed -i 's|^from vllm import SamplingParams$|from vllm.sampling_params import SamplingParams|' \
  "${VLLM_DIR}/vllm/v1/sample/logits_processor/builtin.py" \
  "${VLLM_DIR}/vllm/v1/sample/logits_processor/interface.py" 2>/dev/null

# PoolingParams + SamplingParams (warmup.py)
sed -i 's|^from vllm import PoolingParams, SamplingParams$|from vllm.pooling_params import PoolingParams\nfrom vllm.sampling_params import SamplingParams|' \
  "${VLLM_DIR}/vllm/v1/worker/gpu/warmup.py" 2>/dev/null

# PoolingParams
sed -i 's|^from vllm import PoolingParams$|from vllm.pooling_params import PoolingParams|' \
  "${VLLM_DIR}/vllm/entrypoints/pooling/classify/protocol.py" \
  "${VLLM_DIR}/vllm/entrypoints/pooling/embed/io_processor.py" \
  "${VLLM_DIR}/vllm/entrypoints/pooling/embed/protocol.py" \
  "${VLLM_DIR}/vllm/entrypoints/pooling/pooling/protocol.py" \
  "${VLLM_DIR}/vllm/entrypoints/pooling/scoring/protocol.py" \
  "${VLLM_DIR}/vllm/entrypoints/pooling/scoring/serving.py" 2>/dev/null

# Multi-symbol imports
sed -i 's|^from vllm import PoolingParams, PoolingRequestOutput, envs$|from vllm.pooling_params import PoolingParams\nfrom vllm.outputs import PoolingRequestOutput\nfrom vllm import envs|' \
  "${VLLM_DIR}/vllm/entrypoints/pooling/base/serving.py" 2>/dev/null

sed -i 's|^from vllm import PoolingParams, PoolingRequestOutput, PromptType$|from vllm.pooling_params import PoolingParams\nfrom vllm.outputs import PoolingRequestOutput\nfrom vllm.inputs import PromptType|' \
  "${VLLM_DIR}/vllm/entrypoints/pooling/base/io_processor.py" 2>/dev/null

sed -i 's|^from vllm import PromptType, TextPrompt$|from vllm.inputs import PromptType, TextPrompt|' \
  "${VLLM_DIR}/vllm/entrypoints/pooling/scoring/utils.py" 2>/dev/null

sed -i 's|^from vllm import TokensPrompt$|from vllm.inputs import TokensPrompt|' \
  "${VLLM_DIR}/vllm/v1/engine/async_llm.py" 2>/dev/null

echo "[PATCH] Done"
