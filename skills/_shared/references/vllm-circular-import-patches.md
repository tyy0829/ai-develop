# vllm 0.20.x 循环导入补丁

## Problem

vllm v0.20.x uses `__getattr__` for lazy imports in `__init__.py`. Many submodules do `from vllm import SamplingParams` which triggers `AttributeError` during plugin registration.

## Version Gate

These patches apply ONLY to vllm 0.20.x. If the installed version is different, skip.

## Patch Script

See `apply-vllm-v0.20.x-patches.sh` for the automated version-gated script.

## Manual Reference

If the script is unavailable, apply these patches manually in `/vllm-workspace/vllm`:

### SamplingParams patches
```bash
sed -i 's|^from vllm import SamplingParams$|from vllm.sampling_params import SamplingParams|' \
  vllm/v1/sample/logits_processor/builtin.py \
  vllm/v1/sample/logits_processor/interface.py \
  vllm/v1/worker/gpu/warmup.py
```

### PoolingParams patches
```bash
sed -i 's|^from vllm import PoolingParams$|from vllm.pooling_params import PoolingParams|' \
  vllm/entrypoints/pooling/classify/protocol.py \
  vllm/entrypoints/pooling/embed/io_processor.py \
  vllm/entrypoints/pooling/embed/protocol.py \
  vllm/entrypoints/pooling/pooling/protocol.py \
  vllm/entrypoints/pooling/scoring/protocol.py \
  vllm/entrypoints/pooling/scoring/serving.py
```

### vllm-ascend ModelRegistry patch
```bash
cd /vllm-workspace/vllm-ascend
sed -i 's|^from vllm import ModelRegistry|from vllm.model_executor.models import ModelRegistry|' \
  vllm_ascend/models/__init__.py
```

## Reference

This constraint is defined once here and referenced by:
- `deploy-vllm-on-ascend` (Hard constraint #3)
- `run-ais-bench-accuracy` (Core constraint #5)
