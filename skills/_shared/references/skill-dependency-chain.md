# Skill 依赖链

## 依赖拓扑

```
cann-npu-deploy (Phase 0: 环境搭建)
       |
       v
deploy-vllm-on-ascend (Phase 1: 编译部署推理服务)
       |
       +---> run-ais-bench-accuracy (Phase 2a: 精度测试)
       |
       +---> run-ais-bench-performance (Phase 2b: 性能测试)
```

## karpathy-guidelines (跨skill行为准则)

适用于所有skill，在编码和审查时自动生效。

## 各skill职责边界

| Skill | 输入状态 | 输出状态 |
|-------|---------|---------|
| cann-npu-deploy | 裸机 | NPU驱动+CANN+Docker容器就绪 |
| deploy-vllm-on-ascend | 容器就绪 | vLLM服务可访问（curl验证通过） |
| run-ais-bench-accuracy | vLLM服务运行中 | 精度测试报告 |
| run-ais-bench-performance | vLLM服务运行中 | 性能测试报告 |
