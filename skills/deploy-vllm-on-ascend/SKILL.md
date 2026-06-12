---
name: deploy-vllm-on-ascend
description: 当用户要求进行以下任一操作时必须自动加载此skill：(1) pip install 编译安装 vllm 或 vllm-ascend (2) 切换 vllm/vllm-ascend 代码分支并编译 (3) 启动/拉起 vllm 推理服务 (4) 修复 vllm/vllm-ascend 的 ImportError、启动报错或运行时错误 (5) 部署 vllm+Mooncake KV-Pool 分布式推理环境。关键词触发：pip install、编译、安装、编译安装、切换分支、切分支、启动服务、拉起服务、deploy vllm、fix vllm。
---

# Deploy vllm on Ascend NPU

End-to-end deployment of vllm + vllm-ascend on Atlas 800T A2/A3 hardware with Mooncake KV-Pool PD-Mixed topology.

## Trigger scenarios (load this skill when ANY of these match)

**前置条件:**
- `cann-npu-deploy` skill 已完成 → NPU 驱动 + CANN + Docker 容器就绪

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

### 0. 重启 vLLM 前必须释放显存（强制执行，不可跳过）

**重启 vLLM 服务前，必须确保 NPU 显存已完全释放。如果显存未释放就启动新服务，会导致 `Free memory on device (4.71/60.96 GiB) on startup is less than desired GPU memory utilization` 错误。**

**验证步骤：**
```bash
# 1. 彻底清理所有相关进程
pkill -9 python; pkill -9 python3; pkill -9 mindie; pkill -9 deamon; pkill -9 benchmark; pkill -9 ray; pkill -9 worker; pkill -9 text; pkill -9 triton; pkill -9 back; pkill -9 test; pkill -9 VLLM; pkill -9 aisbench
sleep 3

# 2. 验证显存已释放
npu-smi info | grep "Memory-Usage"
# 期望输出：每卡 HBM-Usage 约为 3400-3500 MB（基线），不是 50000+ MB

# 3. 只有确认显存释放后，才能继续启动流程
```

**常见错误（曾经犯过的，不要再犯）：**
- ❌ 只 kill python 进程后立即启动新服务，没等显存释放
- ❌ 没检查 `npu-smi info` 就假设显存已释放
- ❌ 看到僵尸进程（`<defunct>`）就认为显存未释放（僵尸进程不占显存，但要确认端口是否释放）
- ❌ 没有等待足够时间让系统回收资源（至少 sleep 3 秒）
- ❌ 只清理了 vllm 进程，没清理 aisbench 测试进程（也会占显存）

**正确做法：**
- ✅ 使用完整的清理命令（包含 aisbench）
- ✅ 执行 `npu-smi info` 并**人工确认**显存已释放
- ✅ 等待至少 3 秒让系统回收资源
- ✅ 确认 mooncake 端口（50088/9003）已释放或仍在使用

### 1. pip install must use `--no-build-isolation`

Network from this container to pypi.org is extremely slow or fails. All dependencies are pre-installed.

```bash
# WRONG — will try to download torch from pypi and hang for 30+ minutes
pip install -e .

# CORRECT — uses existing env packages, finishes in minutes
pip install --no-build-isolation -e .
```

Applies to: `vllm`, `vllm-ascend`, and any other package in `/vllm-workspace/`.

### 2. torch-npu ABI 兼容性（动态检测，不要写死版本号）

torch-npu 是针对特定 torch ABI 编译的。版本不匹配会导致 `RuntimeError: Failed to load backend extension: torch_npu`。

**核心原则：不要硬编码版本映射表，要动态读取。**

正确的版本探测顺序（在每次操作前必须执行）：

```bash
# Step 1: 查看当前安装的 torch-npu 版本
python3 -c "import torch_npu; print(torch_npu.__version__)"

# Step 2: 查看用户选择的 vllm-ascend 分支的 requirements.txt / pyproject.toml
#         它写明了依赖的 torch、torch-npu、vllm 版本
cd /vllm-workspace/vllm-ascend
grep -E "^torch|^torch-npu|^torchvision|^triton-ascend" requirements.txt
grep -E "torch-npu|torch==" pyproject.toml

# Step 3: 查看 vllm 对应 tag/commit 的 pyproject.toml 对 torch 的要求
#         确保三者一致（torch 版本 + ABI 版本 + 当前 torch-npu 已装版本）
cd /vllm-workspace/vllm
git show <target-tag>:pyproject.toml | grep "torch ==\|torch =="

# Step 4: 三者冲突时报错给用户，不要静默继续
```

**已知踩坑案例**（仅供识别，不作为操作依据）：
- `prefill_offload_0604` 分支的 `requirements.txt` 写的是 `torch==2.10.0` / `torch-npu==2.10.0`，但它对应的 vllm 是 **v0.22.1**（不是 v0.20.2）
- 旧规则"torch-npu 2.10.0 ↔ vllm v0.20.2"只适用于 vllm-ascend 的 `v0.20.2rc1` tag，对其他分支是错误的
- 当 vllm-ascend 分支使用 `vllm.models.deepseek_v4.*` / `model_executor.layers.mamba.gdn.base` 等路径时，说明其目标是 vllm ≥ v0.22.x

**如果用户指定了 vllm-ascend 分支但不知道 vllm 该用哪个 tag：**

```bash
cd /vllm-workspace/vllm
# 找到 vllm 仓库里存在该分支依赖模块的最近 tag
for tag in $(git tag --sort=-version:refname | head -20); do
  if git ls-tree -r "$tag" -- vllm/models/deepseek_v4/__init__.py 2>/dev/null | grep -q blob; then
    echo "vllm.models.deepseek_v4 exists in $tag"
  fi
done
```

### 3. vllm 某些版本有循环导入 bug（动态检测）

多个版本的 vllm 都可能存在 `__getattr__` 懒加载导致的循环导入问题（`from vllm import SamplingParams` 在 plugin 注册时抛 `AttributeError`）。**不是每个版本都有**，需要按以下流程判断：

**启动前先测试 import 链：**

```bash
python3 -c "from vllm import SamplingParams, PoolingParams; print('OK')" 2>&1
```

- 如果通过：无需 patch
- 如果抛 `AttributeError` 或 `ImportError`：运行版本感知补丁脚本（内部会读取当前 vllm 版本决定修哪些文件）：

```bash
bash /root/.config/opencode/skills/deploy-vllm-on-ascend/references/apply-vllm-v0.20.x-patches.sh
```

> 如果脚本没覆盖你当前的版本，根据报错信息手工 patch 对应的 `from vllm import X` → `from vllm.x_submodule import X`。

### 4. vllm `__version__` 在 plugin 注册期可能不可访问

vllm-ascend 的 `profiling_config.py` / `utils.py` 会访问 `vllm.__version__`，在某些 vllm 版本的懒加载下会失败。

**Fix:**
```bash
# 在 serve 启动脚本中 export VLLM_VERSION=<实际运行的 vllm tag，例如 0.22.1>，
# 让 vllm-ascend 优先读取环境变量而不调用 vllm.__version__
# 在 vllm-ascend profiling_config.py：
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

### 7. 重启 vllm 服务前的进程清理流程（CRITICAL，不可跳过）

**绝对不要在未清理历史进程的情况下直接重启 vllm 服务，这会导致端口占用、显存泄漏、mooncake 连接失败等一系列问题。**

**⚠️ 重启 vLLM 前必须确认显存已释放** - 这是强制前置条件，不可跳过

**必须按顺序执行以下四步：**

**Step 1: 彻底杀死所有相关进程**

优先尝试以下命令：

```bash
pkill -9 python; pkill -9 python3; pkill -9 mindie; pkill -9 deamon; pkill -9 benchmark; pkill -9 ray; pkill -9 worker; pkill -9 text; pkill -9 triton; pkill -9 back; pkill -9 test; pkill -9 VLLM
```

**如果执行后仍有残留进程**，必须自行尝试其他方法（例如 `kill -9 <具体PID>`、`killall <进程名>`、针对不同进程名逐一 pkill 等），**直到所有相关进程都被彻底清除为止**。不能因为一条命令没杀掉就放弃或跳过。

**Step 2: 验证进程已清理 & 显存已释放**

```bash
# 检查是否还有残留进程
ps aux | grep -E "vllm|python|mooncake" | grep -v grep

# 检查 NPU 显存是否释放（应该为 0 或接近 0）
npu-smi info
```

**Step 3: 先重启后端服务（mooncake 或 memcache），再启动 vllm**

**重要：后端服务每次重启后端口可能变化，必须同步更新 vllm 启动脚本。**

```bash
# 1. 启动 mooncake_master
nohup mooncake_master > /tmp/mooncake_master.log 2>&1 &
sleep 3

# 2. 读取 mooncake 实际监听的端口
netstat -tlnp | grep mooncake_master
# 假设输出显示端口为 50051

# 3. 更新 vllm 启动脚本中的端口配置
# 修改 /vllm-workspace/serve_v32.sh 或对应的启动脚本：
# MOONCAKE_MASTER_PORT=50051  # 必须与实际的 mooncake_master 端口一致

# 4. 确认端口一致后再启动 vllm
bash /vllm-workspace/serve_v32.sh
```

**常见错误（曾经犯过的，不要再犯）：**
- ❌ 只 kill python 进程，没 kill mooncake_master
- ❌ kill 后用 `ps aux | grep vllm` 验证，没注意到 mooncake_master 还活着
- ❌ 重启 mooncake 后没检查新端口，导致 vllm 连接的还是旧端口（connection refused）
- ❌ 直接 `bash serve_v32.sh` 没先清理历史进程
- ❌ 用 `pkill vllm` 代替完整的清理命令
- ❌ **kill 进程后没有用 `npu-smi info` 验证显存是否真的释放了**

**正确示例流程：**

```bash
# 1. 彻底清理
pkill -9 python; pkill -9 python3; pkill -9 mindie; pkill -9 deamon; pkill -9 benchmark; pkill -9 ray; pkill -9 worker; pkill -9 text; pkill -9 triton; pkill -9 back; pkill -9 test; pkill -9 VLLM
sleep 2

# 2. 验证清理完成
npu-smi info  # 确认显存已释放

# 3. 启动 mooncake master
nohup mooncake_master > /tmp/mooncake_master.log 2>&1 &
sleep 3

# 4. 读取新端口
netstat -tlnp | grep mooncake_master

# 5. 更新启动脚本（根据实际端口）
# 如果端口变化，修改 serve_v32.sh 中 MOONCAKE_MASTER_PORT

# 6. 启动 vllm
bash /vllm-workspace/serve_v32.sh
```

### 8. 启动 vllm 前必须验证后端服务（mooncake/memcache）是否存活，排除僵尸进程

**容器环境中，子进程退出后常变成僵尸进程（`<defunct>`，状态 `Z`）。`pgrep`、`ps aux | grep` 等命令会误判僵尸进程为"仍在运行"，但实际端口无人监听，vllm 启动时必然 `Initialize mooncake failed` / `Connection refused`。**

**启动 vllm 前必须执行以下检查：**

```bash
# 1. 检查真实存活的进程（排除僵尸 Z 状态）
ps -eo pid,stat,args | grep mooncake_master | grep -v grep | grep -v defunct
# 如果没有输出，说明没有存活的 mooncake_master，需要启动新的

# 或者用更精确的方式：检查是否存在非 Z 状态的进程
ps -eo pid,stat,args | grep '[0-9] [^Z].*mooncake_master' | grep -v grep
```

```bash
# 2. 检查端口是否有人在监听
netstat -tlnp | grep <MOONCAKE_MASTER_PORT>
# 或
lsof -i :<MOONCAKE_MASTER_PORT>
# 如果没有输出，说明端口没人占用，即使 ps 显示了进程也是僵尸
```

```bash
# 3. 如果以上两项都没有有效结果，说明 mooncake_master 没有正常运行
# 必须先启动新的 mooncake_master，再启动 vllm：
nohup mooncake_master --port <PORT> > /tmp/vllm_logs/mooncake_master.log 2>&1 &
sleep 2
# 再次验证
netstat -tlnp | grep <PORT>   # 必须有 LISTEN
```

**对 memcache 同理适用**：

```bash
ps -eo pid,stat,args | grep memcached | grep -v grep | grep -v defunct
netstat -tlnp | grep memcache   # 确认端口在监听
```

**启动脚本中的进程判断也必须排除僵尸：**

```bash
# WRONG — pgrep 会把僵尸进程也匹配到，导致跳过启动
if pgrep -x mooncake_master &>/dev/null; then
    echo "[SKIP] mooncake_master already running"
else
    nohup mooncake_master ... &
fi

# CORRECT — 用 ps + stat 判断，排除 Z 状态
if ps -eo pid,stat,args | grep -q '[0-9] [^Z].*mooncake_master'; then
    echo "[SKIP] mooncake_master already running"
else
    nohup mooncake_master ... &
fi
```

**常见踩坑（曾经犯过的）：**
- ❌ 用 `pgrep mooncake_master` 判断进程存活 → 匹配到 6 个 `<defunct>` 僵尸，跳过启动 → 端口无人监听 → vllm `Initialize mooncake failed`
- ❌ 用 `ps aux | grep mooncake_master` 判断 → 看到僵尸进程以为在运行
- ❌ 容器内无法 reap 僵尸进程（PID 1 不处理 SIGCHLD），只能忽略僵尸，另起新进程

## Execution workflow

### Phase 0: Pre-flight checks

```bash
# Run automated pre-flight check
bash /root/.config/opencode/skills/deploy-vllm-on-ascend/scripts/verify.sh

# Or check manually:
# 1. Verify torch/torch-npu alignment
python3 -c "import torch, torch_npu; print(f'torch={torch.__version__} npu={torch_npu.__version__}')"

# 2. Verify hccn.conf exists
ls -la /etc/hccn.conf

# 3. Verify Redis running (Mooncake metadata backend)
redis-cli ping  # should return PONG

# 4. Verify no stale vllm processes
pgrep -af "vllm.entrypoints" && echo "STALE — must kill first"

# 5. Verify mooncake_master is alive (not zombie)
ps -eo pid,stat,args | grep '[0-9] [^Z].*mooncake_master' | grep -v grep \
  || echo "mooncake_master NOT running or ZOMBIE — must start before vllm (see §8)"

# 6. Verify mooncake port is listening
netstat -tlnp | grep <MOONCAKE_MASTER_PORT> \
  || echo "mooncake port NOT listening — must start mooncake_master first (see §8)"

# 7. Verify mooncake is installed
which mooncake_master && mooncake_master --help 2>&1 | head -3
```

### Phase 1: Install vllm (MUST complete before Phase 2)

```bash
cd /vllm-workspace/vllm

# 目标 tag/commit 来自：
#   - 用户明确指定的分支/tag
#   - 或 vllm-ascend 分支的 requirements/pyproject.toml 对应的 vllm 版本
#   - 或对当前 torch-npu 版本兼容的最近 vllm tag（见 Hard constraint #2）
git checkout <确定的 vllm tag 或 commit>

# 按 Hard constraint #3 的方式测试 import 链，按需打补丁

VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
python3 -c "import vllm; print(vllm.__version__)"    # verify
```

### Phase 2: Install vllm-ascend (MUST wait for Phase 1)

```bash
cd /vllm-workspace/vllm-ascend

# 用户明确指定分支就用用户指定的；否则根据 Hard constraint #2 选择
git checkout <用户指定的 vllm-ascend 分支>

# 按需打补丁（见 Hard constraint #3）

pip install --no-build-isolation -e .
python3 -c "import vllm_ascend"    # verify
```

### Phase 3: Pre-start validation

```bash
# VLLM_VERSION 必须与 Phase 1 实际安装的 vllm tag 一致（见 Hard constraint #4）
VLLM_VERSION=<实际 vllm tag> VLLM_USE_V1=1 \
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
# VLLM_VERSION 必须与当前实际安装的 vllm tag 一致（否则 vllm-ascend 版本判断会走错分支）
export VLLM_VERSION="<当前 vllm 实际 tag，例：0.22.1>"
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
| 多个连续的 `ModuleNotFoundError: vllm.models.*` / `mamba.gdn.base` / `spec_decode.rejection_sampler_utils` | **vllm-ascend 分支与 vllm tag 不匹配**（见 §2）。不要逐个修补丁，要重新按 §2 动态探测 vllm 该用哪个 tag |
| `AttributeError: 'AscendFusedMoE' object has no attribute 'expert_map_manager'` | vllm-ascend 分支调用了 vllm 中已被重构掉的内部 API，同上：说明 vllm tag 错误 |
| `ImportError: cannot import name 'SamplingParams'` | Re-apply patches from §3 |
| `torch_npu` load failure after torch upgrade | torch-npu 是 ABI 绑定的，torch 升级后必须重装匹配的 torch-npu 版本，并重新装 vllm |
| `AttributeError: module 'vllm' has no attribute '__version__'` | Export `VLLM_VERSION=<实际 vllm tag>` 到 startup env；同时把 vllm-ascend 的 `profiling_config.py` 改成 fallback 写法 |
| `AttributeError: ... 'stored_requests'` in kv pool worker | 当前 vllm-ascend 分支的 kv pool 代码有 bug，移除 `use_layerwise:true` 或切到更新的分支 |
| Hang during `pip install -e .` forever at "Installing build dependencies" | Add `--no-build-isolation` |
| `Application startup complete` but first request 500 | "假就绪"，模型未完全加载，检查日志等待 warmup 完成 |
| `/etc/hccn.conf` missing errors in mooncake | 容器未从宿主机拷贝配置文件 | `ssh host "cat /etc/hccn.conf" > /etc/hccn.conf` |
| `ConnectionRefused` on Redis port 6379 | Mooncake 依赖 Redis 作为元数据后端 | `redis-server --daemonize yes` 后重启 mooncake_master |
| **端口被占用 / Address already in use** 或 **Initialize mooncake failed / Connection refused** | **见 §7 + §8：先清理进程（§7），然后用 `ps -eo pid,stat,args` 检查 mooncake_master 是否真实存活（排除 `<defunct>` 僵尸），确认端口有人在监听，再启动 vllm** |
| vllm 启动报 `Initialize mooncake failed`，但 `ps aux` 显示 mooncake_master "在运行" | **mooncake_master 是僵尸进程（`<defunct>`，状态 Z）**。`pgrep` 会匹配到僵尸导致误判（见 §8）。用 `ps -eo pid,stat,args \| grep mooncake_master \| grep -v defunct` 检查；容器内僵尸无法 reap，必须另起新进程：`nohup mooncake_master &`，启动后验证 `netstat -tlnp \| grep <PORT>` 有 LISTEN |
| vllm 启动报错 `failed to connect to mooncake` 但 mooncake 进程在运行 | mooncake 的端口变了。**检查 mooncake 实际端口**（`netstat -tlnp \| grep mooncake`），更新启动脚本中 `MOONCAKE_MASTER_PORT`，再重启 vllm |
| 重启 vllm 后显存占用不正常 / OOM on first request | **历史进程没清理干净**，显存被僵尸进程占着。执行 §7 的完整清理流程，用 `npu-smi info` 确认显存释放后再重启 |

## 共享约束索引

| 约束内容 | 所在共享文件 |
|---------|------------|
| pip install --no-build-isolation | `@_shared/references/pip-build-isolation.md` |
| torch-npu 版本配对 | `@_shared/references/torch-npu-version-pairing.md` |
| vllm 循环导入补丁 | `@_shared/references/vllm-circular-import-patches.md` |
| "假就绪"检测 | `@_shared/references/false-ready-detection.md` |

## 导航

### 前置 Skill
| Skill | 说明 |
|-------|------|
| `cann-npu-deploy` | NPU 驱动 + CANN + Docker 环境搭建 |

### 后续 Skill
| Skill | 说明 |
|-------|------|
| `run-ais-bench-accuracy` | 模型精度评测（GSM8K、MMLU 等） |
| `run-ais-bench-performance` | 推理性能测试（TTFT、吞吐量、Prefix Cache） |
