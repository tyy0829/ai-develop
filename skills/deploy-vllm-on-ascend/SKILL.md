---
name: deploy-vllm-on-ascend
description: 当用户要求进行以下任一操作时必须自动加载此skill：(1) pip install 编译安装 vllm 或 vllm-ascend (2) 切换 vllm/vllm-ascend 代码分支并编译 (3) 启动/拉起 vllm 推理服务 (4) 修复 vllm/vllm-ascend 的 ImportError、启动报错或运行时错误 (5) 部署 vllm+Mooncake KV-Pool 分布式推理环境。关键词触发：pip install、编译、安装、编译安装、切换分支、切分支、启动服务、拉起服务、deploy vllm、fix vllm。
---

# Deploy vllm on Ascend NPU

End-to-end deployment of vllm + vllm-ascend on Atlas 800T A2/A3 hardware with Mooncake KV-Pool PD-Mixed topology.

## Trigger scenarios (load this skill when ANY of these match)

**编译安装类:**
- 用户要求执行 `pip install` 来安装 vllm 或 vllm-ascend
- 用户要求编译 vllm 或 vllm-ascend
- 用户要求切换 vllm/vllm-ascend 分支后重新安装
- 用户说"编译安装"、"重新安装"、"安装依赖"

**启动服务类:**
- 用户要求启动 vllm 推理服务
- 用户要求"拉起服务"、"启动服务"、"起一个vllm"
- 用户要求部署 PD-disaggregated / PD-mixed vllm 服务

**调试修复类:**
- 用户遇到 vllm/vllm-ascend 的 ImportError 或启动报错
- 用户说"修复 vllm"、"vllm 报错"
- 用户要求检查或修复运行时错误

**其他:**
- 任何涉及 vllm server + Mooncake / KV-Pool / kv_both 的请求
- 任何涉及 `/vllm-workspace/` 目录下包的编译安装

## Hard constraints (non-negotiable)

### 1. pip install must use `--no-build-isolation`

Network from this container to pypi.org is extremely slow or fails. All dependencies are pre-installed.

```bash
# WRONG — will try to download torch from pypi and hang for 30+ minutes
pip install -e .

# CORRECT — uses existing env packages, finishes in minutes
pip install --no-build-isolation -e .
```

Applies to: `vllm`, `vllm-ascend`, and any other package in `/vllm-workspace/`.

### 2. torch + torch-npu version pairing is rigid

torch-npu is compiled against a specific torch ABI. A mismatch causes `RuntimeError: Failed to load backend extension: torch_npu`.

| torch | torch-npu | vllm | vllm-ascend |
|---|---|---|---|
| 2.10.0+cpu | 2.10.0 | v0.20.2 | v0.20.2rc1 |
| 2.11.0+cpu | 2.11.0 | main/latest | main/latest |

**Check installed torch-npu before picking vllm/vllm-ascend versions:**
```bash
python3 -c "import torch_npu; print(torch_npu.__version__)"
```

Then pick matching versions:
```bash
# torch-npu 2.10.0 → vllm v0.20.2 + vllm-ascend v0.20.2rc1
cd /vllm-workspace/vllm && git checkout v0.20.2
cd /vllm-workspace/vllm-ascend && git checkout v0.20.2rc1
```

### 3. vllm 0.20.x has circular import bugs from lazy `__getattr__`

vllm v0.20.x uses `__getattr__` for lazy imports in `__init__.py`. Many submodules do `from vllm import SamplingParams` which triggers AttributeError during plugin registration.

**Mandatory patches before startup** (apply to `/vllm-workspace/vllm`):

```bash
cd /vllm-workspace/vllm

# SamplingParams
sed -i 's|^from vllm import SamplingParams$|from vllm.sampling_params import SamplingParams|' \
  vllm/v1/sample/logits_processor/builtin.py \
  vllm/v1/sample/logits_processor/interface.py \
  vllm/v1/worker/gpu/warmup.py

# PoolingParams, PoolingRequestOutput, PromptType, TokensPrompt, TextPrompt
sed -i 's|^from vllm import PoolingParams$|from vllm.pooling_params import PoolingParams|' \
  vllm/entrypoints/pooling/classify/protocol.py \
  vllm/entrypoints/pooling/embed/io_processor.py \
  vllm/entrypoints/pooling/embed/protocol.py \
  vllm/entrypoints/pooling/pooling/protocol.py \
  vllm/entrypoints/pooling/scoring/protocol.py \
  vllm/entrypoints/pooling/scoring/serving.py

# Also fix multi-symbol imports
# vllm/entrypoints/pooling/base/serving.py
# vllm/entrypoints/pooling/base/io_processor.py
# vllm/entrypoints/pooling/scoring/utils.py
# vllm/v1/engine/async_llm.py
# — replace each `from vllm import X, Y, Z` with individual submodule imports
```

**Patch vllm-ascend too** (in `/vllm-workspace/vllm-ascend`):

```bash
# ModelRegistry circular import
sed -i 's|^from vllm import ModelRegistry|from vllm.model_executor.models import ModelRegistry|' \
  vllm_ascend/models/__init__.py
```

### 4. vllm `__version__` not accessible during plugin registration

vllm-ascend's `profiling_config.py` and `utils.py` access `vllm.__version__` which fails due to lazy loading.

**Fix:**
```bash
# In vllm/vllm/__init__.py, ensure __version__ is in MODULE_ATTRS dict (usually is)
# In serve script, export VLLM_VERSION=0.20.2 so vllm-ascend uses env override
# In vllm-ascend profiling_config.py, change to:
# VLLM_VERSION = os.environ.get("VLLM_VERSION") or getattr(vllm, "__version__", "unknown")
```

### 5. KV transfer config JSON must not have outer quotes in heredoc

```bash
# WRONG — outer single quotes become part of the value, causes Pydantic JSON parse error
'{"kv_connector":"AscendStoreConnector",...}'

# CORRECT — pure JSON, no wrapping quotes
{"kv_connector":"AscendStoreConnector",...}
```

### 6. Container must have `/etc/hccn.conf` for Mooncake

Mooncake requires `/etc/hccn.conf` from host. If missing, copy it:

```bash
ssh host "cat /etc/hccn.conf" > /etc/hccn.conf
```

SSH alias `host` should be pre-configured in `~/.ssh/config`:
```
Host host
    HostName 173.149.1.2
    User root
    StrictHostKeyChecking no
```

## Execution workflow

### Phase 0: Pre-flight checks

```bash
# 1. Verify torch/torch-npu alignment
python3 -c "import torch, torch_npu; print(f'torch={torch.__version__} npu={torch_npu.__version__}')"

# 2. Verify hccn.conf exists
ls -la /etc/hccn.conf

# 3. Verify no stale v llm processes
pgrep -af "vllm.entrypoints" && echo "STALE — must kill first"

# 4. Verify mooncake is installed
which mooncake_master && mooncake_master --help 2>&1 | head -3
```

### Phase 1: Install vllm (MUST complete before Phase 2)

```bash
cd /vllm-workspace/vllm
git checkout v0.20.2        # match torch-npu version per table above

# Apply circular import patches (see Hard constraint #3 and #4)
# ...

VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
python3 -c "import vllm; print(vllm.__version__)"    # verify
```

### Phase 2: Install vllm-ascend (MUST wait for Phase 1)

```bash
cd /vllm-workspace/vllm-ascend
git checkout v0.20.2rc1     # match table above

# Apply ModelRegistry patch (see Hard constraint #3)
# ...

pip install --no-build-isolation -e .
python3 -c "import vllm_ascend"    # verify
```

### Phase 3: Pre-start validation

```bash
# Test full import chain BEFORE launching
VLLM_VERSION=0.20.2 VLLM_USE_V1=1 \
  python3 -m vllm.entrypoints.openai.api_server --help 2>&1 | head -5

# If ImportError surfaces, patch and re-test until all imports resolve
```

### Phase 4: Launch service

## ⚠️ MANDATORY: Confirm launch parameters with user BEFORE starting

Do NOT use hardcoded values from existing scripts. Before launching, you MUST ask the user to confirm:

1. **Model path** — `ls /root/models/` to list available models, ask user which one to use
2. **Served model name** — derive from model path, confirm with user
3. **Port** (default 8100)
4. **TP size** (default 8)
5. **Max model len / max num seqs** — especially important for large models (V3.2, V3.1) that need smaller values than V2-Lite

Only proceed to launch after user confirms all parameters.

Use a startup script (e.g. `/vllm-workspace/serve_pd_mix_mooncake.sh`).

Key parameters for PD-Mixed (single instance, all cards):
```bash
export VLLM_VERSION="0.20.2"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --tensor-parallel-size 8 \
    --kv-transfer-config '{"kv_connector":"AscendStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"lookup_rpc_port":"1","backend":"mooncake"}}' \
    ...
```

### Phase 5: Verification

```bash
# 1. Wait for "Application startup complete" in logs
tail -f /tmp/vllm_logs/instance_1.log | grep "startup complete"

# 2. Sanity: list models
curl -s http://localhost:8100/v1/models

# 3. Smoke: send a chat request (CRITICAL — startup alone is not success)
curl -s http://localhost:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v2-lite-chat","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"temperature":0}'
```

"Application startup complete" is NOT success. If the first request returns 500, treat as runtime failure and inspect logs.

### Phase 6: Debug failures

Most common first-request crash: `AttributeError` in kv pool worker. Check:
```bash
grep -A 15 "ERROR\|AttributeError\|RuntimeError" /tmp/vllm_logs/instance_1.log | tail -30
```

Known trap: `use_layerwise:true` in `kv_connector_extra_config` triggers `KVCacheStoreLayerSendingThread` which is incomplete in 0.20.2rc1. Remove it if you hit `'KVCacheStoreLayerSendingThread' object has no attribute 'stored_requests'`.

## Output requirements

When this skill completes, it must deliver:

1. **vllm service reachable** at target port (default 8100)
2. **HTTP 200 on `/v1/models`** listing the requested model
3. **HTTP 200 with non-empty response** on a smoke-chat request
4. **No ERROR lines** in log after startup completion
5. **One clean commit** in working repo (if any code patches were made)

### Verification commands (must all pass)

```bash
curl -sf http://localhost:8100/v1/models > /dev/null && echo "✅ models OK" || echo "❌ models FAIL"
curl -sf http://localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<NAME>","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
  > /dev/null && echo "✅ inference OK" || echo "❌ inference FAIL"
grep -c "ERROR\|Traceback" /tmp/vllm_logs/instance_1.log  # should be 0 after startup
```

## Troubleshooting shortcuts

| Symptom | Fix |
|---|---|
| `ImportError: cannot import name 'SamplingParams'` | Re-apply patches from §3 (`cd /vllm-workspace/vllm && ...`) |
| `torch_npu` load failure after torch upgrade | Rollback torch to match torch-npu, reinstall vllm |
| `AttributeError: module 'vllm' has no attribute '__version__'` | Export `VLLM_VERSION=x.y.z` in startup env |
| `AttributeError: ... 'stored_requests'` in kv pool worker | Remove `use_layerwise:true` from kv config JSON |
| Hang during `pip install -e .` forever at "Installing build dependencies" | Add `--no-build-isolation` |
| `Application startup complete` but first request 500 | Check logs; this is "false-ready", treat as failure |
| `/etc/hccn.conf` missing errors in mooncake | `ssh host "cat /etc/hccn.conf" > /etc/hccn.conf` |
