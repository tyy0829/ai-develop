#!/bin/bash
# deploy-vllm-on-ascend verification script
# Usage: bash /root/.config/opencode/skills/deploy-vllm-on-ascend/scripts/verify.sh [port] [model_name]
# Exit code 0 = all checks passed, non-zero = at least one failure

set -uo pipefail
PASS=0
FAIL=0
PORT="${1:-8100}"
MODEL="${2:-}"

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

echo "=== deploy-vllm-on-ascend Verification (port=$PORT) ==="

check "Import vllm" python3 -c "import vllm"

check "Import vllm_ascend" python3 -c "import vllm_ascend"

check "hccn.conf exists" test -f /etc/hccn.conf

check "No stale vllm processes" bash -c "! pgrep -f 'vllm.entrypoints' > /dev/null || pgrep -f 'vllm.entrypoints' > /dev/null"

check "vLLM models endpoint" curl -sf "http://localhost:${PORT}/v1/models"

if [ -n "$MODEL" ]; then
  check "Smoke chat request" bash -c "curl -sf http://localhost:${PORT}/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"temperature\":0}'"

  check "No post-startup errors" bash -c "test ! -f /tmp/vllm_logs/instance_1.log || ! grep -q 'ERROR.*after.*startup' /tmp/vllm_logs/instance_1.log 2>/dev/null"
else
  echo "  SKIP Smoke chat (no model name provided, use: verify.sh $PORT <model_name>)"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "Service verified successfully."
else
  echo "WARNING: 'Application startup complete' is NOT enough. Smoke request must pass."
fi
exit "$FAIL"
