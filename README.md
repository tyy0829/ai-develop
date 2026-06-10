# ai-develop

OpenCode skill configurations and deployment guides for Huawei Ascend NPU environments.

## Contents

### Skills

- `skills/deploy-vllm-on-ascend/` - Complete workflow for deploying vllm + vllm-ascend on Ascend NPU with Mooncake KV-Pool

### Guides

- `AGENTS.md` - Environment conventions and rules for AI assistants working in this environment

## Usage

### Install as OpenCode Skill

```bash
mkdir -p ~/.config/opencode/skills/deploy-vllm-on-ascend
cp skills/deploy-vllm-on-ascend/SKILL.md ~/.config/opencode/skills/deploy-vllm-on-ascend/
```

### Copy AGENTS.md to your project

```bash
cp AGENTS.md /path/to/your/project/
```

## Environment

- Hardware: Atlas 800T A2 (Ascend 910B3) with 8 NPUs
- OS: Ubuntu on aarch64
- Framework: vllm + vllm-ascend
- KV Pool: Mooncake
