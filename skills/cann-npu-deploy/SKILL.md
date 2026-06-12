---
name: cann-npu-deploy
description: >
  Use when the user asks to install, deploy, or configure Ascend NPU driver (HDK),
  CANN toolkit, vllm-ascend environment, or create vllm-ascend Docker containers
  on a machine with Ascend 910B hardware. Also use when asked to set up the full
  Ascend AI software stack from scratch, or when encountering NPU/CANN related
  installation issues. Covers openEuler/CentOS (yum) and Ubuntu (apt) on aarch64/x86_64.
---

# Ascend NPU + CANN + vllm-ascend 全栈部署

本 Skill 指导在搭载昇腾 NPU 的机器上，完成从零到一的完整环境部署：NPU 驱动(Ascend HDK)、CANN 软件栈(v9.0.0)、vllm-ascend Docker 容器。

## 触发条件

当用户提出以下请求时触发：
- 安装昇腾 NPU 驱动 / Ascend HDK
- 安装或配置 CANN 环境
- 部署 vllm-ascend 环境
- 创建 vllm-ascend 容器
- 在容器内安装工具（如 opencode）
- 排查 NPU 相关安装问题

## 硬约束（不可违反）

| # | 约束 | 原因 |
|---|------|------|
| 1 | CANN 安装顺序必须为 Toolkit → Ops → NNAL | Ops 依赖 Toolkit 环境，NNAL 依赖两者；乱序会导致编译失败 |
| 2 | 安装 Ops 前必须 `source ascend-toolkit/set_env.sh` | Ops 安装脚本需要 Toolkit 环境变量，不 source 会报错退出 |
| 3 | 下载 CANN .run 包使用 `$(uname -m)` 而非 `$(uname -i)` | 某些 openEuler 版本 `uname -i` 返回 `unknown`，导致 404 下载失败 |
| 4 | `--shm-size=8g` 是容器必须参数 | 大模型推理共享内存需求大，默认 64MB 会立即 OOM |
| 5 | 非交互终端安装 CANN，必须用 .run 包 + `--quiet` | yum 在非交互终端下访问 `/dev/tty` 会静默失败 |
| 6 | pip 安装任何包必须加 `--no-build-isolation` | 详见 `@_shared/references/pip-build-isolation.md` |

## 工作流程总览

```
Phase 1: 环境预检 → Phase 2: 系统依赖 → Phase 3: NPU 驱动
    → Phase 4: CANN 软件栈 → Phase 5: Docker 安装
    → Phase 6: 拉取镜像 & 创建容器 → Phase 7: 容器内增强（可选）
```

---

## Phase 1: 环境预检

**目标**: 确认硬件、OS、架构和已有环境状态。

### 执行步骤

1. 检查系统架构和操作系统：
```bash
uname -m && cat /etc/os-release 2>/dev/null | head -5
```

2. 检查是否已有 NPU 设备：
```bash
ls /dev/davinci* 2>/dev/null || echo "No davinci devices"
```

3. 检查是否已有驱动安装：
```bash
ls /usr/local/Ascend/ 2>/dev/null
cat /etc/ascend_install.info 2>/dev/null
npu-smi info 2>/dev/null | head -20 || echo "No NPU driver"
```

4. 检查已有 Docker 和 Python 环境：
```bash
docker --version 2>/dev/null || echo "No Docker"
python3 --version 2>/dev/null || echo "No Python3"
```

### 输出要求

确认以下信息并告知用户：
- OS 发行版和架构（如 openEuler 22.03 aarch64）
- NPU 硬件是否已检测到（`/dev/davinci*`）
- 已有驱动/CANN 版本（如有）
- 判断是首次安装还是覆盖安装

---

## Phase 2: 安装系统依赖

**目标**: 安装编译驱动和运行 CANN 所需的系统级依赖。

### 执行步骤

1. 安装基础系统依赖：
```bash
# openEuler / CentOS / Kylin (yum)
yum install -y gcc gcc-c++ cmake numactl-devel wget git curl jq

# Ubuntu / Debian (apt)
apt-get update -y && apt-get install -y gcc g++ cmake libnuma-dev wget git curl jq
```

2. 安装驱动编译依赖：
```bash
# yum 系统
yum install -y make dkms gcc kernel-headers-$(uname -r) kernel-devel-$(uname -r)

# apt 系统
apt-get install -y make dkms gcc linux-headers-$(uname -r)
```

3. 创建驱动运行用户（首次安装时）：
```bash
groupadd -f HwHiAiUser
id HwHiAiUser 2>/dev/null || useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
```

### 输出要求

确认所有依赖安装成功，无报错。

---

## Phase 3: 安装 NPU 驱动 (Ascend HDK)

**目标**: 安装昇腾 NPU 固件和驱动，使 `npu-smi` 可用。

### 方式选择

**优先使用 yum 仓库在线安装**（适用于 openEuler/CentOS），**备选方案**为手动下载 .run 包安装。

### 3A: yum 在线安装（推荐）

1. 配置 CANN yum 仓库：
```bash
curl -sL https://repo.oepkgs.net/ascend/cann/ascend.repo -o /etc/yum.repos.d/ascend.repo
rpm --import https://repo.oepkgs.net/ascend/cann/RPM-GPG-KEY-CANN
```

2. 验证仓库可用并查看可用驱动版本：
```bash
yum list available Ascend* --showduplicates 2>/dev/null | grep -i "hdk\|910"
```

3. 安装 NPU 驱动（根据硬件选择包名）：

| 硬件 | 包名 |
|------|------|
| Atlas 800 A2 系列 (910B) | `Ascend910B-driver` |
| Atlas 800 A1 系列 (910) | `Ascend-hdk-910-npu-driver` |
| Atlas 800 A3 系列 | `Ascend-hdk-A3-npu-driver` |
| Atlas 300I A2 | `Ascend-hdk-310p-npu-driver` |
| Atlas A5 系列 (950) | `Ascend-hdk-950-npu-driver` |

> **注意**: 以上包名来自 `yum search` 实际可用列表，不同版本仓库可能略有差异。建议安装前先执行 `yum list available Ascend* --showduplicates` 确认可用包名和版本号。

> **关于固件**: yum 安装 `Ascend910B-driver` 时，驱动包已内置固件（安装日志中会出现 `Driver package installed successfully! The new version takes effect immediately.`）。如需单独更新固件，使用 .run 包方式安装。

```bash
# 示例：安装 Atlas A2 (910B) 驱动 25.5.0
yum install -y Ascend910B-driver-25.5.0
```

### 3B: 手动 .run 包安装（备选）

若无法使用 yum 仓库，从华为官网下载驱动包：
- **下载入口**: https://www.hiascend.com/hardware/firmware-drivers/community
- **文件名模式**: `Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run`

```bash
chmod +x Ascend-hdk-910b-npu-driver_<version>_linux-aarch64.run
./Ascend-hdk-910b-npu-driver_<version>_linux-aarch64.run --full --install-for-all
```

### 安装顺序

- **首次安装**: 先装驱动，再装固件
- **覆盖安装**: 先装固件，再装驱动
- 如提示需要重启，执行 `reboot`

### 输出要求

```
验证命令: npu-smi info
期望结果: 显示 NPU 卡信息，驱动版本正确，所有卡状态 OK
```

---

## Phase 4: 安装 CANN 9.0.0 软件栈

**目标**: 安装 CANN Toolkit、Ops 算子包和 NNAL 加速库。

CANN 需安装 3 个组件：

| 组件 | 说明 |
|------|------|
| **Toolkit** | CANN 开发套件，用于训练、推理、模型转换、编译 |
| **Ops** | 算子包（需匹配硬件类型，如 910b） |
| **NNAL** | 神经网络加速库（ATB + SiP） |

### 4A: yum 安装（推荐）

如果 Phase 3 已配置了 yum 仓库，直接安装：
```bash
yum install -y Ascend-cann-toolkit-9.0.0
yum install -y Ascend-cann-910b-ops-9.0.0
yum install -y Ascend-cann-nnal-9.0.0
```

> **注意**: yum 安装可能因 `/dev/tty` 不可用而失败（尤其在非交互终端中），此时切换到 4B 方案。

### 4B: .run 包安装（备选/推荐）

1. 下载 CANN 软件包：
```bash
# 注意：使用 $(uname -m) 即 uname -m，返回 aarch64 / x86_64，
# 而非 uname -i（某些 openEuler 版本会返回 unknown，导致 404）
wget --header="Referer: https://www.hiascend.com/" \
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-$(uname -m).run" \
  -O /root/ascend-packages/Ascend-cann-toolkit_9.0.0_linux-$(uname -m).run

wget --header="Referer: https://www.hiascend.com/" \
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-910b-ops_9.0.0_linux-$(uname -m).run" \
  -O /root/ascend-packages/Ascend-cann-910b-ops_9.0.0_linux-$(uname -m).run

wget --header="Referer: https://www.hiascend.com/" \
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-nnal_9.0.0_linux-$(uname -m).run" \
  -O /root/ascend-packages/Ascend-cann-nnal_9.0.0_linux-$(uname -m).run
```

2. 依次安装（顺序不可变: Toolkit → Ops → NNAL）：
```bash
cd /root/ascend-packages

# Toolkit
chmod +x Ascend-cann-toolkit_9.0.0_linux-$(uname -m).run
./Ascend-cann-toolkit_9.0.0_linux-$(uname -m).run --quiet --full --install-for-all

# Ops（需先 source toolkit 环境）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
chmod +x Ascend-cann-910b-ops_9.0.0_linux-$(uname -m).run
./Ascend-cann-910b-ops_9.0.0_linux-$(uname -m).run --quiet --install

# NNAL
chmod +x Ascend-cann-nnal_9.0.0_linux-$(uname -m).run
./Ascend-cann-nnal_9.0.0_linux-$(uname -m).run --quiet --install
```

> **注意**: 安装 Ops 前必须先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，否则脚本找不到 Toolkit 环境会报错退出。

### 配置环境变量

```bash
cat > /etc/profile.d/ascend.sh << 'EOF'
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
EOF
chmod +x /etc/profile.d/ascend.sh
source /etc/profile.d/ascend.sh
```

### 输出要求

```
验证: 确认 /usr/local/Ascend/ 下有 ascend-toolkit、cann-9.0.0、driver、nnal 目录
验证: source 后 echo $ASCEND_HOME 有值
```

---

## Phase 5: 安装 Docker

**目标**: 安装 Docker CE，使能容器化部署 vllm-ascend。

### 执行步骤

1. 检查现有 Docker：
```bash
docker --version 2>/dev/null
```

2. 若已有旧版 Docker（如 18.x 以下），先卸载再安装新版：
```bash
# openEuler / CentOS
systemctl stop docker 2>/dev/null
yum remove -y docker-engine 2>/dev/null

# 配置 Docker CE 仓库（华为镜像）
cat > /etc/yum.repos.d/docker-ce.repo << 'EOF'
[docker-ce-stable]
name=Docker CE Stable - $basearch
baseurl=https://mirrors.huaweicloud.com/docker-ce/linux/centos/8/$basearch/stable
enabled=1
gpgcheck=0
EOF

yum install -y docker-ce docker-ce-cli containerd.io
systemctl start docker
docker --version
```

3. 对于 Ubuntu：
```bash
apt-get install -y docker-ce docker-ce-cli containerd.io
```

### 输出要求

```
验证: docker --version 返回 20.x+
验证: systemctl is-active docker 返回 active
```

---

## Phase 6: 拉取镜像 & 创建容器

**目标**: 拉取 vllm-ascend 官方 Docker 镜像并创建带有 NPU 设备映射的容器。

### 6A: 拉取镜像

官方镜像仓库: `quay.io/ascend/vllm-ascend`

| 镜像标签 | 硬件 | OS |
|----------|------|-----|
| `v0.20.2rc1` | Atlas A2 | Ubuntu |
| `v0.20.2rc1-openeuler` | Atlas A2 | openEuler |
| `v0.20.2rc1-a3` | Atlas A3 | Ubuntu |
| `v0.20.2rc1-a3-openeuler` | Atlas A3 | openEuler |
| `v0.20.2rc1-310p` | Atlas 300I | Ubuntu |
| `v0.20.2rc1-310p-openeuler` | Atlas 300I | openEuler |

```bash
# 示例：Atlas A2 + openEuler 主机
docker pull quay.io/ascend/vllm-ascend:v0.20.2rc1-openeuler
```

> **注意**: 镜像约 18GB，拉取耗时较长（5~15 分钟），建议后台执行。

### 6B: 创建容器

参考模板（来自 `/root/vllm-ascend-skill/vllm-ascend创建容器方式.txt`）：

```bash
docker run -d --name <容器名> \
    --shm-size=8g \
    --net=host \
    -w /home \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /dev:/dev \
    -v /home:/home \
    -v /tmp:/tmp \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    <镜像名> sleep infinity
```

**关键参数说明**:

| 参数 | 说明 |
|------|------|
| `--shm-size=8g` | 共享内存，大模型推理建议至少 8G（1G 仅为最小值，容易 OOM） |
| `--net=host` | 使用宿主机网络 |
| `--device /dev/davinci[N]` | 按需挂载 NPU 卡，A2 最多 8 张，A3 最多 16 张 |
| `--device /dev/davinci_manager` | NPU 管理器（必选） |
| `--device /dev/devmm_svm` | SVM 内存管理（必选） |
| `--device /dev/hisi_hdc` | HDC 控制器（必选） |
| `-v /usr/local/Ascend/driver/lib64/` | 驱动库文件挂载（必选） |
| `sleep infinity` | 保持容器运行，避免 bash 立即退出 |

### 输出要求

```
验证: docker exec <容器名> npu-smi info → 显示 NPU 卡信息
验证: docker ps → 容器状态为 Up
```

---

## Phase 7: 容器内增强（可选）

在容器内安装额外工具。

### 安装 opencode

通过 npm 安装（推荐，速度快）：

```bash
docker exec <容器名> bash -c "
  dnf install -y nodejs npm && \
  npm install -g opencode-ai && \
  opencode --version
"
```

> **注意**: 容器内的 npm 默认可能指向华为云镜像，若安装失败可切换到官方源：
> `npm config set registry https://registry.npmjs.org`

---

## 安装后全局验证

### 一键验证

```bash
bash /root/.config/opencode/skills/cann-npu-deploy/scripts/verify.sh
```

### 验证清单

| 检查项 | 命令 | 期望结果 |
|--------|------|----------|
| NPU 驱动 | `npu-smi info` | 显示所有 NPU 卡，状态 OK |
| 驱动版本 | `cat /usr/local/Ascend/driver/version.info` | 显示正确版本号 |
| CANN Toolkit | `ls /usr/local/Ascend/ascend-toolkit/` | 目录存在 |
| CANN Ops | `ls /usr/local/Ascend/ascend-toolkit/latest/ops/` | 目录存在（Ops 与 Toolkit 合并安装在同一目录下） |
| NNAL | `ls /usr/local/Ascend/nnal/` | 包含 atb 和 asdsip |
| 环境变量 | `echo $ASCEND_HOME` | 有值 |
| Docker | `docker ps` | 容器运行中 |
| 容器内 NPU | `docker exec <容器名> npu-smi info` | 容器内可见 NPU |

## 回滚 / 卸载（覆盖安装时使用）

如需彻底清除环境重新安装，按以下顺序卸载：

```bash
# 1. 卸载 NNAL
rm -rf /usr/local/Ascend/nnal

# 2. 卸载 CANN Toolkit（包含 Ops）
rm -rf /usr/local/Ascend/ascend-toolkit

# 3. 卸载 NPU 驱动（谨慎操作）
# .run 包安装: 使用 --uninstall 参数
# yum 安装: yum remove Ascend910B-driver

# 4. 清理环境变量
rm -f /etc/profile.d/ascend.sh
```

> **注意**: 卸载驱动后需 `reboot` 使设备状态重置。

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `npu-smi` 无输出 | 驱动未安装或设备未就绪 | 确认 NPU 硬件已插入，重新安装驱动并重启 |
| yum 安装 toolkit 报 `/dev/tty` 错误 | 非交互模式下安装脚本尝试读终端 | 改用 .run 包 + `--quiet` 安装 |
| 容器启动后立即退出 | `bash` 命令在无 TTY 时不保持运行 | 用 `sleep infinity` 替代 `bash` |
| npm 安装报 404 not found | npm registry 指向了华为云镜像 | `npm config set registry https://registry.npmjs.org` |
| 驱动安装报 `ifconfig: command not found` | 缺少 net-tools | `yum install -y net-tools` 或忽略（不影响功能） |
| 容器内 `npu-smi` 找不到命令 | 未挂载 npu-smi 二进制 | 确保 `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` |
| CANN 安装后 `ASCEND_HOME` 环境变量为空 | 未 source set_env.sh 或未配置 profile.d | `source /etc/profile.d/ascend.sh` 后检查，或重新执行 Phase 4 配置步骤 |
| 容器启动后被 OOM killed | `--shm-size` 过小（默认 64MB） | 重新创建容器，指定 `--shm-size=8g` |
| 覆盖安装 CANN Ops 报 `Toolkit not found` | 安装 Ops 前未 source Toolkit 环境 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 后重试 |

---

## 参考链接

- [CANN 9.0.0 社区版文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/quickstart/instg_quick.html)
- [固件与驱动下载](https://www.hiascend.com/hardware/firmware-drivers/community)
- [vllm-ascend 安装文档](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)
- [CANN yum 仓库](https://repo.oepkgs.net/ascend/cann/)
- [CANN OBS 下载仓库](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/)
- [vllm-ascend Docker 镜像 (quay.io)](https://quay.io/repository/ascend/vllm-ascend)
<!-- 以下参考文件仅限本机，其他环境无此文件 -->
- 参考创建容器方式（本机）: `/root/vllm-ascend-skill/vllm-ascend创建容器方式.txt`
- vllm,vllm-ascend,Mooncake 编译安装方式（本机）: `/root/vllm-ascend-skill/安装编译vllm,vllm-ascend,Mooncake的方式.txt`

## 共享约束索引

| 约束内容 | 所在共享文件 |
|---------|------------|
| pip install --no-build-isolation | `@_shared/references/pip-build-isolation.md` |

## 导航

### 后续 Skill
| Skill | 说明 |
|-------|------|
| `deploy-vllm-on-ascend` | vLLM 编译安装 + 推理服务启动（Phase 6 容器就绪后执行） |
