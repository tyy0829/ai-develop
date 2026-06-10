---
name: run-ais-bench-accuracy
description: 执行 AISBench 精度测试的完整工作流。当用户要求测试模型精度（如 GSM8K、MMLU、HumanEval 等基准）时自动触发。基于 vllm-ascend 环境中的 vLLM API 服务进行在线精度评测，输出与官方对比的分析报告。
---

# AISBench 精度测试工作流

## 触发条件

当用户的需求符合以下特征时触发：
- 要求测试模型精度（accuracy test, precision test, 精度测试）
- 提到特定基准数据集（GSM8K, MMLU, HumanEval, CMMLU, C-Eval 等）
- 要求与官方成绩对比
- 要求运行 benchmark 或 evaluation

## 核心约束

### 1. pip install 必须使用 --no-build-isolation
容器内无法访问 pypi.org，所有依赖已预安装。**必须**使用 `--no-build-isolation` 参数：
```bash
# ❌ 错误 - 会尝试从外网下载，15分钟后超时失败
pip install -e .

# ✅ 正确 - 使用本地已安装的依赖
pip install --no-build-isolation -e .
```

### 2. 不要创建冗余配置文件
**禁止**为了测试而创建新的 config 包装文件。直接修改现有配置文件：
```bash
# ❌ 错误 - 创建 wrapper config
cat > my_test_config.py << 'EOF'
from configs.models.vllm_api.vllm_api_general_chat import models
# 修改参数...
EOF

# ✅ 正确 - 直接修改现有配置文件
# 编辑 /vllm-workspace/ais-benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py
```

### 3. 不要混淆 stream_chat 和 general_chat
这两个配置文件有本质区别，绝不能混用：
- **stream_chat**: 流式输出模型（vLLM 使用 `--enable-streaming`）
- **general_chat**: 非流式输出模型（标准 vLLM 部署）

**检查方法**：
```bash
# 查看模型启动命令
ps aux | grep vllm.entrypoints
# 如果包含 --enable-streaming，用 stream_chat 配置
# 否则用 general_chat 配置
```

### 4. 问题诊断优先于修复
**禁止**在分析错误原因之前就修改后处理逻辑或其他代码：
1. 先运行完整测试
2. 分析结果中的错误模式
3. 识别是模型问题、评测框架问题、还是后处理 bug
4. 确认问题根因后再修复

**案例**：GSM8K 测试中，先发现评测器将 "65,000" 误判为 "000"，确认是 gsm8k_postprocess 的 bug 后才修复。

### 5. vllm v0.20.x 循环导入修复
如果 vllm 0.20.x 启动时出现以下错误：
```
AttributeError: module 'vllm' has no attribute '__version__'
AttributeError: module 'vllm' has no attribute 'SamplingParams'
AttributeError: module 'vllm' has no attribute 'ModelRegistry'
```

这是因为 `vllm/__init__.py` 使用延迟加载，而某些模块在 `__init__` 执行期间就被导入了。**修复方法**：

```bash
# 进入 vllm 源码目录
cd /vllm-workspace/vllm

# 编辑 vllm/__init__.py，将这些符号添加到 __all__ 列表
nano vllm/__init__.py
```

在 `__all__ = [...]` 中添加：
```python
__all__ = [
    # ... existing entries ...
    "SamplingParams",
    "ModelRegistry",
    # 根据报错添加其他符号
]
```

### 6. 评测前必须验证服务状态
运行 benchmark 前**必须**验证 vLLM 服务正常：
```bash
# 检查服务
curl -s http://localhost:8100/v1/models | python3 -m json.tool

# 期望输出包含模型信息
# 如果失败，先启动服务
```

### 7. 数据集路径规范
AISBench 期望数据集位于特定目录：
```
/vllm-workspace/ais-benchmark/ais_bench/datasets/<dataset_name>/
```

如果数据集不存在，先下载（如 GSM8K）：
```bash
# GSM8K 下载脚本已提供
python3 /vllm-workspace/ais-benchmark/download_gsm8k.py
```

### 8. 温度参数对精度有影响
- `temperature=0.0`：确定性输出，适合精度测试
- `temperature>0`：随机输出，每次结果略有不同

**精度对比测试**应使用 `temperature=0` 以保证可重复性。

### 9. 不要过早优化后处理逻辑
只有在确认以下情况时才修改后处理函数：
1. 错误确实是评测框架 bug（如千位分隔符解析错误）
2. 错误不影响模型本身的能力评估
3. 修复后的逻辑与标准处理一致

**反例**：不要因为准确率低于官方就修改后处理逻辑来"提高"分数。

## 标准工作流 (SOP)

### Phase 1: 环境验证
```bash
# Step 1.1: 确认容器启动
docker ps

# Step 1.2: 确认 vLLM 服务运行
curl -s http://localhost:8100/v1/models

# Step 1.3: 确认 AISBench 已安装
cd /vllm-workspace/ais-benchmark
python3 -c "import ais_bench"

# Step 1.4: 确认数据集存在
ls /vllm-workspace/ais-benchmark/ais_bench/datasets/gsm8k/
# 应看到 train.jsonl, test.jsonl
```

### Phase 2: 配置模型
```bash
# Step 2.1: 确定模型配置类型
ps aux | grep vllm.entrypoints
# 根据 --enable-streaming 参数选择 stream_chat 或 general_chat

# Step 2.2: 编辑模型配置文件
# 假设使用 non-streaming 模型
nano /vllm-workspace/ais-benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
# 修改 host_port, model, temperature 等参数

# Step 2.3: 确认数据集配置存在
ls /vllm-workspace/ais-benchmark/ais_bench/benchmark/configs/datasets/gsm8k/
# 应看到 gsm8k_gen_0_shot_cot_chat_prompt.py 等配置
```

### Phase 3: 运行评测
```bash
# Step 3.1: 启动评测任务
cd /vllm-workspace/ais-benchmark
nohup ais_bench test \
  --models vllm_api_general_chat \
  --datasets gsm8k_gen_0_shot_cot_chat_prompt \
  --work-dir /tmp/gsm8k_v1 \
  --debug > /tmp/gsm8k_v1.log 2>&1 &

# Step 3.2: 监控进度
tail -f /tmp/gsm8k_v1.log
# 等待 "Evaluation finished" 消息

# Step 3.3: 查看结果
cat /tmp/gsm8k_v1/20260609_084457/summary.csv
```

### Phase 4: 结果分析
```bash
# Step 4.1: 提取准确率
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('/tmp/gsm8k_v1/20260609_084457/summary.csv')
print(df.to_markdown(index=False))
EOF

# Step 4.2: 与官方成绩对比
echo "官方 DeepSeek-V2-Lite GSM8K: 72%"
echo "实测: $(python3 -c "import pandas as pd; df = pd.read_csv('/tmp/gsm8k_v1/20260609_084457/summary.csv'); print(df['accuracy'].values[0])")%"

# Step 4.3: 分析错误案例（如有必要）
python3 << 'EOF'
import json
results_dir = '/tmp/gsm8k_v1/20260609_084457'
with open(f'{results_dir}/vllm_api_general_chat/gsm8k_gen/results.json') as f:
    results = json.load(f)
# 分析错误模式...
EOF
```

### Phase 5: 问题诊断与修复（仅在需要时）
```bash
# Step 5.1: 检查后处理逻辑
cat /vllm-workspace/ais-benchmark/ais_bench/benchmark/datasets/gsm8k.py | grep -A 20 "def gsm8k_postprocess"

# Step 5.2: 测试后处理逻辑
python3 << 'EOF'
import re

def gsm8k_postprocess(text: str) -> str:
    """Extract numerical answer from model output."""
    numbers = re.findall(r'\d+', text)
    return numbers[-1] if numbers else ''

# Test comma parsing
test_cases = [
    "answer: 65,000",
    "answer: $65,000",
    "answer: 1,234,567",
]
for test in test_cases:
    result = gsm8k_postprocess(test)
    print(f"'{test}' -> '{result}'")
EOF

# Step 5.3: 如确认是 bug，修复后处理函数
nano /vllm-workspace/ais-benchmark/ais_bench/benchmark/datasets/gsm8k.py
# 添加千位分隔符处理逻辑

# Step 5.4: 重新运行评测（使用新的 work-dir）
# 重复 Phase 3，使用 /tmp/gsm8k_v2
```

### Phase 6: 生成报告
```bash
# Step 6.1: 汇总多轮测试结果
python3 << 'EOF'
import pandas as pd

versions = ['v1', 'v2', 'v3']
results = []
for ver in versions:
    try:
        df = pd.read_csv(f'/tmp/gsm8k_{ver}/*/summary.csv')
        results.append({
            'version': ver,
            'accuracy': df['accuracy'].values[0],
            'note': 'Baseline' if ver == 'v1' else 'Post bug fix'
        })
    except:
        pass

df_summary = pd.DataFrame(results)
print(df_summary.to_markdown(index=False))
EOF

# Step 6.2: 与官方成绩对比
echo "| Metric | Official | Our Best | Gap |"
echo "|--------|----------|----------|-----|"
echo "| GSM8K  | 72.0%    | XX.X%    | X.X% |"
```

## 验证标准

### 1. 测试成功标准
- ✅ vLLM 服务正常运行
- ✅ AISBench 评测完成（日志中出现 "Evaluation finished"）
- ✅ 结果 CSV 文件生成
- ✅ 准确率为合理范围（如 GSM8K 应在 30-80% 之间）

### 2. 结果验证标准
- ✅ 准确率与官方成绩在同一量级（差距不超过 20%）
- ✅ 错误案例可解释（如评测框架 bug、温度参数影响等）
- ✅ 如修复 bug 后，准确率提升幅度合理

### 3. 对比验证标准
- ✅ 使用相同参数（temperature, max_tokens 等）
- ✅ 使用相同评测配置（few-shot vs 0-shot）
- ✅ 明确记录与官方的差异原因

## 常见问题排查

### Q1: 评测卡住不动
**症状**：日志长时间无新输出

**诊断**：
```bash
# 检查进程
ps aux | grep ais_bench

# 检查是否有僵尸进程
ps aux | grep defunct

# 查看完整日志
cat /tmp/gsm8k_v1.log | tail -100
```

**解决**：
```bash
# 杀死卡住的进程
kill -9 <pid>

# 清理并重新运行
rm -rf /tmp/gsm8k_v1
# 重新执行 Phase 3
```

### Q2: 准确率异常低（如 <30%）
**可能原因**：
1. 模型未正确加载（检查 vLLM 服务）
2. 数据集路径错误
3. 后处理逻辑错误
4. 温度参数过高

**诊断**：
```bash
# 验证服务
curl -s http://localhost:8100/v1/models

# 检查数据集
head -5 /vllm-workspace/ais-benchmark/ais_bench/datasets/gsm8k/test.jsonl

# 检查后处理
python3 -c "from ais_bench.benchmark.datasets.gsm8k import gsm8k_postprocess; print(gsm8k_postprocess('answer: 42'))"

# 检查模型配置
cat /vllm-workspace/ais-benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
```

### Q3: 评测框架报错
**常见错误**：
```
ModuleNotFoundError: No module named 'mmengine'
```

**解决**：
```bash
pip install --no-build-isolation mmengine==0.10.4
# 或安装完整依赖
pip install --no-build-isolation -r /vllm-workspace/ais-benchmark/requirements/runtime.txt
```

### Q4: 结果与官方差距过大
**可能原因**：
1. 评测配置不同（few-shot vs 0-shot）
2. 温度参数不同
3. 后处理逻辑不同
4. 模型精度不同（fp16 vs fp32）

**处理**：不进行强行调优，记录差异原因并向用户报告。

## 案例记录

### 案例 1: GSM8K 精度测试
**用户请求**："测试 DeepSeek-V2-Lite-Chat 在 GSM8K 上的精度，对比官方成绩"

**执行过程**：
1. 验证 vLLM 服务运行（curl /v1/models）
2. 下载 GSM8K 数据集到指定目录
3. 修改 vllm_api_general_chat.py 配置（host_port=8100, model=deepseek-v2-lite-chat）
4. 运行 ais_bench test（第一次）
5. 获得结果：58.91%
6. 与官方 72% 对比，差距 13.09%
7. 分析错误案例，发现评测器将 "65,000" 误判为 "000"
8. 确认是 gsm8k_postprocess 的 bug（未处理千位分隔符）
9. 修复后处理逻辑
10. 重新运行评测（第二次）
11. 获得结果：62.77%（提升 3.86%）
12. 生成对比报告

**关键教训**：
- 不要在第一次测试后就直接修改代码
- 先分析错误模式，再决定是否需要修复
- 修复后必须重新运行完整测试验证

## 附录

### 附录 A: AISBench 常用命令
```bash
# 查看支持的模型
ais_bench list-models

# 查看支持的数据集
ais_bench list-datasets

# 运行测试
ais_bench test --models <model_config> --datasets <dataset_config> --work-dir <output_dir>

# 列出测试任务
ais_bench ls

# 查看任务状态
ais_bench status <task_id>
```

### 附录 B: vLLM 服务启动参数
```bash
# 标准服务（non-streaming）
vllm serve /root/models/DeepSeek-V2-Lite-Chat \
  --host 0.0.0.0 \
  --port 8100 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --trust-remote-code

# 流式服务（streaming）
# 添加 --enable-streaming 参数
```

### 附录 C: 官方成绩参考
| 模型 | GSM8K | MMLU | HumanEval |
|------|-------|------|-----------|
| DeepSeek-V2-Lite | 72.0% | 75.6% | 61.0% |
| Qwen2.5-7B | 58.9% | 76.2% | 57.3% |
| Llama-3-8B | 47.9% | 69.4% | 44.5% |

**注意**：官方成绩可能使用不同的评测配置，仅供参考。

### 附录 D: 后处理逻辑设计原则
GSM8K 后处理需要：
1. 提取模型输出中的数字答案
2. 处理千位分隔符（65,000 → 65000）
3. 处理货币符号（$65000 → 65000）
4. 处理"answer:"前缀
5. 返回最后一个数字（通常是最终答案）

**正确实现**：
```python
def gsm8k_postprocess(text: str) -> str:
    """Extract numerical answer from model output."""
    # 处理千位分隔符
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # 移除货币符号
    text = re.sub(r'[$€£]', '', text)
    # 提取所有数字
    numbers = re.findall(r'\d+', text)
    # 返回最后一个数字
    return numbers[-1] if numbers else ''
```
