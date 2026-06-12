---
name: run-ais-bench-performance
description: 执行 AISBench 性能测试（TTFT、吞吐量、prefix cache命中率等）
version: 1.0.0
author: system
tags: [performance, benchmark, ais-bench, ttft, throughput, kv-pool]
---

# AIS-Bench 性能测试 Skill

## 前置条件

本 skill 依赖以下 skill 已完成：
- `cann-npu-deploy` → NPU 驱动 + CANN 已安装，容器已运行
- `deploy-vllm-on-ascend` → vLLM 推理服务已启动且 smoke test 通过

## 触发条件

当用户提出以下类型的需求时触发：
- "测试模型性能"
- "测试TTFT" 或 "测试延迟"
- "测试吞吐量"
- "测试prefix cache" 或 "测试前缀缓存"
- 需要分析KV Pool命中率和性能数据

## 核心约束

### 1. output_len 必须为 1
性能测试只测试TTFT，不需要生成多个token。

**理由**：如果output_len大于1，模型会生成多个token，大幅增加测试时间（8-12分钟 vs 2-3分钟），而且我们关注的是首token时间（TTFT），不是生成速度。

**示例**：
```python
# ✅ 正确
output_len=1

# ❌ 错误
output_len=100  # 会浪费时间
```

### 2. request_rate 必须为 0
突发模式，一次性发送所有请求。

**理由**：避免速率限制导致测试时间过长，确保能够快速完成性能测试。

**示例**：
```python
# ✅ 正确
request_rate=0  # 突发发送所有请求

# ❌ 错误
request_rate=10  # 需要16秒才能发完160个请求
```

### 3. dp 参数必须删除
当前环境是单实例，无DP配置。

**理由**：当前只有1个vLLM实例运行在端口8100，没有数据并行（DP）配置，因此dp参数无意义。

**示例**：
```python
# ✅ 正确 - 不添加dp参数
# 直接运行测试

# ❌ 错误
dp=2  # 参数无意义
```

### 4. dataset_type 必须为 prefix_cache 并指定 repeat_rate
验证前缀缓存收益。

**理由**：要测试prefix cache性能，必须使用`dataset_type=prefix_cache`，并指定`repeat_rate`（典型值0.9表示90%的数据有相同前缀）。

**示例**：
```python
# ✅ 正确
dataset_type=prefix_cache
repeat_rate=0.9

# ❌ 错误
dataset_type=normal  # 无法测试prefix cache
```

### 5. 使用 aisbench_auto_tools_prefix 工具
从GitHub clone工具生成前缀cache测试数据。

**理由**：不要修改ais-bench原生的gsm8k.py文件（会影响其他数据集测试），而是使用专用的工具生成带前缀的测试数据。

**工具地址**：https://github.com/rayn-zzz/aisbench_auto_tools_prefix

### 6. vLLM服务配置
- 默认端口：8100
- 需要预先启动vLLM API服务器
- 需要启动KV Pool后端服务（如Mooncake或Ascend Store）
- 需要启用prefix caching

### 7. ais-bench版本要求
- 已安装版本：3.1.20260609（/vllm-workspace/ais-benchmark）
- 不要重新安装，直接使用现有安装

### 8. 结果分析要点
需要关注的性能指标：
- **TTFT（Time To First Token）**：首token响应时间
- **Request throughput**：请求吞吐量（req/s）
- **Input Token Throughput**：输入token吞吐量（tokens/s）
- **Output Token Throughput**：输出token吞吐量（tokens/s）
- **External hit rate**：外部缓存命中率（%）
- **HBM hit rate**：HBM缓存命中率（%）

### 9. 命中率计算方式
- **External hit**：外部KV Pool缓存命中
- **HBM hit**：本地HBM缓存命中
- 命中率 = (hits / queries) × 100%

### 10. TTFT指标解读
TTFT分布通常有双峰：
- **低值（~225ms）**：命中cache的请求
- **高值（~2700ms）**：未命中cache的请求

需要根据命中率综合分析。

## 标准工作流 (SOP)

### Phase 0: 前置预检

1. **检查 Redis 是否运行**（Mooncake 依赖 Redis 作为 metadata 后端）
```bash
redis-cli ping  # 应返回 PONG
# 如果 Redis 未运行，先启动：
# redis-server --daemonize yes
```

2. **检查 NPU 利用率**（确保非NVIDIA环境）
```bash
npu-smi info  # 确认所有 NPU 卡状态正常
```

3. **运行预检脚本**
```bash
bash /root/.config/opencode/skills/run-ais-bench-performance/scripts/verify.sh
```

### Phase 1: 环境准备

1. **启动 vLLM API 服务器**
```bash
# 使用 Mooncake KV Pool（PD混合模式）
export VLLM_VERSION="0.20.2"
export PYTHONPATH=/vllm-workspace/vllm:${PYTHONPATH}
nohup python -m vllm.entrypoints.openai.api_server \
  --model /models/DeepSeek-V2-Lite-Chat \
  --served-model-name deepseek-v2-lite-chat \
  --port 8100 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_buffer_size":10485760}' \
  > vllm_server.log 2>&1 &
```

2. **启动 Mooncake master 服务**
```bash
nohup mooncake_master \
  --config /etc/mooncake/mooncake_config.json \
  --port 50088 \
  --metadata-backend redis \
  --metadata-connection redis://localhost:6379 \
  > mooncake_master.log 2>&1 &
```

3. **验证服务状态**
```bash
curl -s http://localhost:8100/v1/models
# 应该返回模型列表

curl -s http://localhost:50088/metrics
# 应该返回 Mooncake 指标
```

### Phase 2: 克隆和配置工具

1. **克隆工具**
```bash
cd /opt
git clone https://github.com/rayn-zzz/aisbench_auto_tools_prefix.git
# 或者使用 /tmp/aisbench_auto_tools_prefix
```

2. **编辑配置文件**
```bash
cd /opt/aisbench_auto_tools_prefix
cat > config.py << 'EOF'
{
  "model_name": "deepseek-v2-lite-chat",
  "model_path": "/models/DeepSeek-V2-Lite-Chat",
  "host_ip": "localhost",
  "host_port": 8100,
  "ais_bench_path": "/vllm-workspace/ais-benchmark",
  "dataset_dir": "/home/dataset"
}
EOF
```

3. **确保数据集目录存在**
```bash
mkdir -p /home/dataset
```

### Phase 3: 执行性能测试

1. **运行 prefix cache 性能测试**
```bash
cd /opt/aisbench_auto_tools_prefix
python3 aisbench_test.py \
  --input_len 8192 \
  --output_len 1 \
  --data_num 125 \
  --concurrency 20 \
  --dataset_type prefix_cache \
  --repeat_rate 0.9 \
  --prefix_test \
  --request_rate 0
```

2. **参数说明**
- `--input_len 8192`：输入长度8K tokens
- `--output_len 1`：输出长度1（只测TTFT）
- `--data_num 125`：测试样本数125
- `--concurrency 20`：并发数20
- `--dataset_type prefix_cache`：使用前缀缓存数据集类型
- `--repeat_rate 0.9`：90%的数据共享同一前缀
- `--prefix_test`：启用前缀缓存测试模式
- `--request_rate 0`：突发发送所有请求

### Phase 4: 查看结果

1. **实时查看日志**
```bash
tail -f /tmp/perf_test_output.log
```

2. **查看结果文件**
```bash
cat perf_result.json
```

返回示例：
```json
{
  "timestamp": "2026-06-09T12:45:30",
  "config": {
    "input_len": 8192,
    "output_len": 1,
    "repeat_rate": 0.9,
    "concurrency": 20,
    "data_num": 125
  },
  "results": {
    "ttft_avg_ms": 234.5,
    "ttft_p95_ms": 312.8,
    "ttft_p99_ms": 456.2,
    "request_throughput": 85.2,
    "input_token_throughput": 698765.4,
    "output_token_throughput": 85.2,
    "total_token_throughput": 698850.6
  },
  "cache_hit_rates": {
    "external_hit_rate": 82.58,
    "hbm_hit_rate": 0.0
  }
}
```

### Phase 5: 清理和停止

1. **停止性能测试（如果需要）**
```bash
ps aux | grep aisbench_test.py | grep -v grep | awk '{print $2}' | xargs kill -9
```

2. **查看vLLM服务指标**
```bash
curl -s http://localhost:8100/metrics | grep -E "ttft|throughput|cache"
```

3. **查看Mooncake指标**
```bash
curl -s http://localhost:50088/metrics | grep -E "hit|query"
```

## 验证标准

### 成功标准

1. **测试完成标志**
- 日志中包含 "Performance test completed successfully"
- 生成 `perf_result.json` 文件

2. **性能指标合理**
- TTFT 平均值在 100-500ms 之间
- 请求吞吐量 > 50 req/s
- 输入token吞吐量 > 100,000 tokens/s

3. **命中率符合预期**
- External hit rate 应该接近 repeat_rate (例如 repeat_rate=0.9 时，hit rate ~90%)
- HBM hit rate 通常为 0（取决于配置）

### 常见问题排查

| 症状 | 根因 | 修复动作 |
|------|------|---------|
| repeat_rate=0.9 但 external_hit_rate < 50% | vLLM 未启用 `--enable-prefix-caching`，或 mooncake_master 未运行，或 KV connector 配置错误 | 检查启动参数含 `--enable-prefix-caching`；`pgrep -f mooncake_master` 确认进程；`curl -s http://localhost:50088/metrics` 验证 Mooncake |
| ttft_avg_ms > 1000 | 模型 fake-ready（启动日志显示 ready 但首次请求失败）、系统负载过高、KV buffer size 过小 | smoke chat 验证模型可用：`curl /v1/chat/completions`；`npu-smi info` 检查 NPU |
| request_throughput < 10 req/s | 并发数过小、NPU 利用率瓶颈、网络延迟 | 增大 `--concurrency`（建议 20-50）；`npu-smi info` 检查利用率 |
| `Error: Dataset directory not found` | /home/dataset 目录不存在 | `mkdir -p /home/dataset` 后重新运行 |
| 工具使用默认 config.py 而非自定义配置 | config.py JSON 格式错误，或路径不在工具根目录 | `python3 -m json.tool < config.py` 验证 JSON 格式 |
| `ConnectionError: Redis refused` | Redis 未运行，Mooncake 元数据后端不可用 | `redis-server --daemonize yes` 后重启 mooncake_master |
| `mooncake_master: command not found` | Mooncake 未安装或未加入 PATH | 确认 `/vllm-workspace/Mooncake` 已编译并 export PATH |
| `nvidia-smi: command not found` | 误用了 NVIDIA 命令，本环境是昇腾 NPU | 使用 `npu-smi info` 替代 |

## 高级技巧

### 技巧1：测试不同重复率对比
```bash
# 测试 50%, 70%, 90%, 99% 重复率
for rate in 0.5 0.7 0.9 0.99; do
  echo "Testing repeat_rate=$rate"
  python3 aisbench_test.py \
    --input_len 8192 \
    --output_len 1 \
    --data_num 125 \
    --concurrency 20 \
    --dataset_type prefix_cache \
    --repeat_rate $rate \
    --prefix_test \
    --request_rate 0
  sleep 5
done
```

### 技巧2：测试不同并发数
```bash
# 测试并发数 10, 20, 50, 100
for conc in 10 20 50 100; do
  echo "Testing concurrency=$conc"
  python3 aisbench_test.py \
    --input_len 8192 \
    --output_len 1 \
    --data_num 125 \
    --concurrency $conc \
    --dataset_type prefix_cache \
    --repeat_rate 0.9 \
    --prefix_test \
    --request_rate 0
  sleep 5
done
```

### 技巧3：测试不同输入长度
```bash
# 测试输入长度 2K, 4K, 8K, 16K, 32K
for len in 2048 4096 8192 16384 32768; do
  echo "Testing input_len=$len"
  python3 aisbench_test.py \
    --input_len $len \
    --output_len 1 \
    --data_num 50 \
    --concurrency 20 \
    --dataset_type prefix_cache \
    --repeat_rate 0.9 \
    --prefix_test \
    --request_rate 0
  sleep 5
done
```

## 参考资料

- **vLLM Prefix Caching**: https://docs.vllm.ai/en/latest/features/prefix_caching.html
- **AIS-Bench 官方文档**: https://github.com/ais-bench/ais-bench
- **Mooncake KV Pool**: https://github.com/kvcache-ai/mooncake
- **vLLM-Ascend**: https://github.com/vllm-project/vllm-ascend
- **性能测试工具**: https://github.com/rayn-zzz/aisbench_auto_tools_prefix

## 更新日志

- 2026-06-09: 初始版本，基于DeepSeek-V2-Lite-Chat模型验证
- 2026-06-09: 添加prefix cache测试支持，验证90%重复率场景
- 2026-06-09: 优化核心约束，强调output_len=1和request_rate=0的重要性
- 2026-06-10: 修复 nvidia-smi 为 npu-smi，添加 Phase 0 Redis 预检，添加 verify.sh，统一排障格式，添加跨 skill 导航

## 导航

### 前置 Skill
| Skill | 说明 |
|-------|------|
| `cann-npu-deploy` | NPU 驱动 + CANN + Docker 环境搭建 |
| `deploy-vllm-on-ascend` | vLLM 编译安装 + 推理服务启动 |

### 后续步骤
- 性能测试完成后，运行精度测试 → 调用 `run-ais-bench-accuracy` skill
- 如需重新部署服务 → 调用 `deploy-vllm-on-ascend` skill
