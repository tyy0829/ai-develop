#!/bin/bash
# run-ais-bench-accuracy verification script
# Usage: bash /root/.config/opencode/skills/run-ais-bench-accuracy/scripts/verify.sh [port] [dataset]
# Exit code 0 = all checks passed, non-zero = at least one failure

set -uo pipefail
PASS=0
FAIL=0
PORT="${1:-8100}"
DATASET="${2:-gsm8k}"

AISBENCH_DIR="/vllm-workspace/ais-benchmark"

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

echo "=== run-ais-bench-accuracy Pre-flight (port=$PORT, dataset=$DATASET) ==="

check "AISBench installed" python3 -c "import ais_bench"

check "vLLM service reachable" curl -sf "http://localhost:${PORT}/v1/models"

check "Dataset exists ($DATASET)" ls "${AISBENCH_DIR}/ais_bench/datasets/${DATASET}/test.jsonl"

check "Model config exists" bash -c "ls ${AISBENCH_DIR}/ais_bench/benchmark/configs/models/vllm_api/vllm_api_*.py"

check "Dataset config exists" bash -c "ls ${AISBENCH_DIR}/ais_bench/benchmark/configs/datasets/${DATASET}/"

check "No stale ais_bench processes" bash -c "! pgrep -f 'ais_bench test' > /dev/null 2>&1"

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit "$FAIL"
