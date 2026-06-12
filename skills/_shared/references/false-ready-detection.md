# "Application startup complete" is NOT success (False-Ready Detection)

## Problem

vLLM prints "Application startup complete" when the HTTP server starts listening, but the model may not be fully loaded. The first inference request can fail with HTTP 500.

## Rule

NEVER treat "Application startup complete" as deployment success. You MUST verify with a smoke inference request:

```bash
curl -sf http://localhost:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<MODEL_NAME>","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}'

# Exit code 0 = real success
# Exit code non-zero = false-ready, treat as runtime failure
```

## Common causes of false-ready

| Symptom | Cause | Fix |
|---------|-------|-----|
| First request returns 500 | Model weights not fully loaded (warmup in progress) | Wait 30-60s, retry |
| `AttributeError: 'stored_requests'` | `use_layerwise:true` in kv config (incomplete in 0.20.2rc1) | Remove `use_layerwise:true` |
| `RuntimeError: ACLNN error` | NPU device not ready or memory allocation failed | Check `npu-smi info`, restart service |

## Reference

This constraint is defined once here and referenced by:
- `deploy-vllm-on-ascend` (Phase 5 Verification)
- `run-ais-bench-accuracy` (Constraint #6 - verify service before test)
- `vllm-ascend-model-adapter` (Dummy vs Real weight gate)
