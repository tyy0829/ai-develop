# torch + torch-npu 版本配对表

## Rule

torch-npu is compiled against a specific torch ABI. A mismatch causes `RuntimeError: Failed to load backend extension: torch_npu`.

| torch | torch-npu | vllm | vllm-ascend |
|---|---|---|---|
| 2.10.0+cpu | 2.10.0 | v0.20.2 | v0.20.2rc1 |
| 2.11.0+cpu | 2.11.0 | main/latest | main/latest |

> **注意**: 此表更新于 2026.06，新版本发布时请执行 `python3 -c "import torch_npu; print(torch_npu.__version__)"` 确认当前环境版本。

## Check Before Install

```bash
python3 -c "import torch, torch_npu; print(f'torch={torch.__version__} npu={torch_npu.__version__}')"
```

Then pick matching versions:
```bash
# torch-npu 2.10.0 -> vllm v0.20.2 + vllm-ascend v0.20.2rc1
cd /vllm-workspace/vllm && git checkout v0.20.2
cd /vllm-workspace/vllm-ascend && git checkout v0.20.2rc1
```

## Reference

This constraint is defined once here and referenced by:
- `deploy-vllm-on-ascend` (Hard constraint #2)
- `run-ais-bench-accuracy` (Core constraint #5)
