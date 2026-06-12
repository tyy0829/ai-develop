#!/bin/bash
# cann-npu-deploy verification script
# Usage: bash /root/.config/opencode/skills/cann-npu-deploy/scripts/verify.sh
# Exit code 0 = all checks passed, non-zero = at least one failure

set -uo pipefail
PASS=0
FAIL=0

check() {
  local name="$1"
  shift
  if "$@" > /dev/null 2>&1; then
    echo "  OK  $name"
    PASS=$((PASS + 1))
  else
    echo "  FAIL $name"
    FAIL=$((FAIL + 1))
  fi
}

echo "=== cann-npu-deploy Verification ==="

check "NPU driver (npu-smi)" npu-smi info

check "CANN Toolkit directory" ls /usr/local/Ascend/ascend-toolkit/latest

check "CANN Ops directory" ls /usr/local/Ascend/ascend-toolkit/latest/ops

check "NNAL directory" ls /usr/local/Ascend/nnal

check "ASCEND_HOME env var" bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && test -n \"\$ASCEND_HOME\""

check "Docker daemon" docker info

check "Docker container running" docker ps --format '{{.Status}}' | grep -q Up

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit "$FAIL"
