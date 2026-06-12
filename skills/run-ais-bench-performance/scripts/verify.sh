#!/bin/bash
# run-ais-bench-performance verification script
# Usage: bash /root/.config/opencode/skills/run-ais-bench-performance/scripts/verify.sh [port]
# Exit code 0 = all checks passed, non-zero = at least one failure

set -uo pipefail
PASS=0
FAIL=0
PORT="${1:-8100}"

AISBENCH_DIR="/vllm-workspace/ais-benchmark"
TOOL_DIR="/opt/aisbench_auto_tools_prefix"

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

echo "=== run-ais-bench-performance Pre-flight (port=$PORT) ==="

check "vLLM service reachable" curl -sf "http://localhost:${PORT}/v1/models"

check "Prefix caching enabled" bash -c "curl -sf http://localhost:${PORT}/metrics | grep -qi prefix"

check "Mooncake master running" pgrep -f mooncake_master

check "Redis running" redis-cli ping

check "Mooncake metrics reachable" curl -sf "http://localhost:50088/metrics"

check "Performance test tool installed" test -x "${TOOL_DIR}/aisbench_test.py"

check "Dataset directory exists" test -d /home/dataset

check "npu-smi available" npu-smi info

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit "$FAIL"
