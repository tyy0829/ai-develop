# AGENTS.md

## What This Repo Is

Deployment bootstrap scripts for `hidagent`, a precompiled systemd service. Runs on Huawei Ascend NPU hardware (aarch64) and x86_64 Linux. No application code is built here — binaries are fetched from internal servers.

## Files

| File | Purpose |
|---|---|
| `deployWithInstall.sh` | **Fresh install**: downloads obsutil + hidagent zip from internal server (`21.21.0.31:8888`), runs `install.sh`, configures firewall (port `51234/tcp`) |
| `fetch.sh` | **Restart or update**: tries `systemctl restart hidagent` first; if service doesn't exist, downloads `deployCode.sh` from an IP-routed deployment server |
| `opencode.json` | OpenCode provider config (nanyan/GLM-5.1, bailian/Qwen3.7 Max) |
| `HwHiAiUser/` | Dotfiles for the default Ascend NPU user |

## Conventions an Agent Would Miss

- **Two scripts, different purposes**: `deployWithInstall.sh` is for first-time install from a fixed internal server. `fetch.sh` is for field updates — it selects the deployment server based on the host's IP subnet (CIDR map in `IP_SEGMENT_MAP`).
- **Architecture auto-detection**: both scripts key on `uname -m` to pick `aarch64` vs `amd64` binaries. The primary target is aarch64 (Kunpeng/Ascend ARM).
- **Installed service path**: `hidagent` installs to `/usr/local/hidagent/` with systemd unit at `/etc/systemd/system/hidagent.service`.
- **Chroot/container caveat**: `systemctl` commands are silently ignored in chroot environments (visible in `deploy.log`). The service symlink is created but the service won't start until a real boot.
- **Firewall**: the agent listens on port `51234/tcp`; the install script adds this rule via `ufw` or `firewalld` depending on distro.
- **Distro support**: Ubuntu/Debian (`apt`/`ufw`) and CentOS/RHEL/openEuler (`yum`/`firewalld`). Distro detection uses `/etc/os-release`.

## pip install Critical Rule

**NEVER let pip download from pypi.org or any external index.** This environment is behind restricted network — external downloads are extremely slow or fail. All dependencies are pre-installed in the container.

- Always use `pip install --no-build-isolation -e .` (not bare `pip install -e .`)
- `--no-build-isolation` tells pip to reuse the existing environment instead of creating an isolated build env that tries to fetch packages
- Applies to: vllm, vllm-ascend, and any other package in `/vllm-workspace/`
- Correct commands (from `/home/vllm-ascend-skill/install.txt`):
  ```
  # vllm
  VLLM_TARGET_DEVICE=empty pip install -v --no-build-isolation -e .
  # vllm-ascend
  pip install -v --no-build-isolation -e .
  ```

## vllm-ascend Environment

This container runs on Atlas 800T A2 (Ascend 910B3) with 8 NPUs. Key paths and versions:
- vllm source: `/vllm-workspace/vllm`
- vllm-ascend: `/vllm-workspace/vllm-ascend`
- Mooncake: `/vllm-workspace/Mooncake` (v0.3.8.post1, pre-compiled)
- torch: 2.10.0+cpu, torch-npu: 2.10.0
- Models: `/root/models/` (DeepSeek-V2-Lite-Chat, DeepSeek-V3.2-Exp-W8A8, etc.)
- SSH to host: `ssh host` (alias configured in `~/.ssh/config`)
- Host IP: `173.149.1.2`, container `/etc/hccn.conf` populated

## Skill Usage Rules

**Always proactively load relevant skills** when performing tasks that match skill descriptions. Do NOT wait for the user to remind you. Specifically:

- `deploy-vllm-on-ascend` skill: MUST be loaded for any vllm/vllm-ascend pip install, compilation, branch switching + install, service startup, or runtime error fixes.
- `vllm-ascend-model-adapter` skill: MUST be loaded when adapting models for vllm on Ascend.

**Before launching vllm service**, always ask the user to confirm: model path (check `ls /root/models/`), served model name, port, TP size, and max model len. Never assume the user wants the default model from existing scripts.
