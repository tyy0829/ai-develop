# pip install 必须使用 --no-build-isolation

Network from this container to pypi.org is extremely slow or fails. All dependencies are pre-installed in the environment.

## Rule

```bash
# WRONG - will try to download torch from pypi and hang for 30+ minutes
pip install -e .

# CORRECT - uses existing env packages, finishes in minutes
pip install --no-build-isolation -e .
```

## Applies to

- `vllm` (with `VLLM_TARGET_DEVICE=empty`)
- `vllm-ascend`
- Any other package in `/vllm-workspace/`
- Any package that requires build dependencies already present in the environment

## Reference

This constraint is defined once here and referenced by:
- `deploy-vllm-on-ascend` (Hard constraint #1)
- `run-ais-bench-accuracy` (Core constraint #1)
- `cann-npu-deploy` (Phase 7 container setup)
