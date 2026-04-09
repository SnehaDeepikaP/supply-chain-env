#!/usr/bin/env bash
# validate-submission.sh — Supply Chain OpenEnv Submission Validator
#
# Checks:
#   1. Docker build succeeds
#   2. Container starts and /health returns 200
#   3. /reset returns valid observation
#   4. /step accepts a valid action
#   5. openenv.yaml is present and parseable
#   6. inference.py is in root directory
#   7. All 3 tasks listed and reachable
#
# Usage: bash validate-submission.sh [HF_SPACE_URL]
#
# Example local: bash validate-submission.sh http://localhost:7860
# Example HF:    bash validate-submission.sh https://your-username-supply-chain-env.hf.space

set -e

SPACE_URL="${1:-http://localhost:7860}"
IMAGE_NAME="supply-chain-env-validator"
PASSED=0
FAILED=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓ PASS${NC} — $1"; ((PASSED++)); }
fail() { echo -e "${RED}✗ FAIL${NC} — $1"; ((FAILED++)); }
info() { echo -e "${YELLOW}ℹ${NC}  $1"; }

echo "=============================="
echo "  OpenEnv Submission Validator"
echo "  Supply Chain Disruption Mgr"
echo "=============================="
echo ""

# ── Check 1: Required files ──────────────────────────────────────────────────
info "Checking required files..."

[ -f "inference.py" ]    && pass "inference.py exists in root" || fail "inference.py NOT found in root"
[ -f "openenv.yaml" ]    && pass "openenv.yaml exists"         || fail "openenv.yaml NOT found"
[ -f "Dockerfile" ]      && pass "Dockerfile exists"           || fail "Dockerfile NOT found"
[ -f "requirements.txt" ] && pass "requirements.txt exists"    || fail "requirements.txt NOT found"
[ -f "app.py" ]          && pass "app.py exists"               || fail "app.py NOT found"
[ -f "env.py" ]          && pass "env.py exists"               || fail "env.py NOT found"
[ -f "models.py" ]       && pass "models.py exists"            || fail "models.py NOT found"

echo ""

# ── Check 2: openenv.yaml is valid YAML ─────────────────────────────────────
info "Validating openenv.yaml..."
python3 -c "
import yaml, sys
with open('openenv.yaml') as f:
    d = yaml.safe_load(f)
tasks = [t['id'] for t in d.get('tasks', [])]
required = {'supplier_triage', 'logistics_reroute', 'cascade_disruption'}
missing = required - set(tasks)
if missing:
    print(f'Missing tasks: {missing}')
    sys.exit(1)
print(f'Tasks found: {tasks}')
" && pass "openenv.yaml valid, 3 tasks defined" || fail "openenv.yaml invalid or missing tasks"

echo ""

# ── Check 3: Python imports ──────────────────────────────────────────────────
info "Checking Python imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
from models import Observation, Action, Reward, StepResponse
from env import SupplyChainEnv
from tasks.task1_supplier_triage import Task1SupplierTriage
from tasks.task2_logistics_reroute import Task2LogisticsReroute
from tasks.task3_cascade_disruption import Task3CascadeDisruption
print('All imports OK')
" && pass "Python imports successful" || fail "Python import errors — check dependencies"

echo ""

# ── Check 4: Unit test — reset/step/state/grade cycle ───────────────────────
info "Running unit test: reset → step → state → grade..."
python3 -c "
import sys
sys.path.insert(0, '.')
from env import SupplyChainEnv
from models import Action, ActivateSupplierAction

env = SupplyChainEnv()

# Test all 3 tasks
for task_id in ['supplier_triage', 'logistics_reroute', 'cascade_disruption']:
    obs = env.reset(task_id=task_id, seed=42)
    assert obs.task_id == task_id, f'Wrong task_id in obs: {obs.task_id}'
    assert obs.step == 0

    action = Action(
        action_type='wait',
        reasoning='validation test'
    )
    resp = env.step(action)
    assert 0.0 <= resp.reward.value <= 1.0, f'Reward out of range: {resp.reward.value}'
    assert isinstance(resp.done, bool)

    state = env.state()
    assert state.task_id == task_id

    score = env.grade()
    assert 0.0 <= score <= 1.0, f'Grade out of range: {score}'
    print(f'  {task_id}: reward={resp.reward.value:.4f} score={score:.4f} OK')

print('All 3 tasks passed unit test')
" && pass "Unit test passed: reset/step/state/grade cycle works for all 3 tasks" \
  || fail "Unit test FAILED"

echo ""

# ── Check 5: Docker build ─────────────────────────────────────────────────────
info "Building Docker image (this may take a minute)..."
if docker build -t "$IMAGE_NAME" . > /tmp/docker_build.log 2>&1; then
    pass "Docker build succeeded"
else
    fail "Docker build FAILED — see /tmp/docker_build.log"
    cat /tmp/docker_build.log | tail -20
fi

echo ""

# ── Check 6: Docker run + health check ───────────────────────────────────────
info "Starting Docker container..."
CONTAINER_ID=$(docker run -d -p 7861:7860 --name "$IMAGE_NAME-test" "$IMAGE_NAME" 2>/dev/null || echo "")

if [ -z "$CONTAINER_ID" ]; then
    fail "Docker run failed"
else
    info "Waiting for container to start..."
    sleep 8
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:7861/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        pass "Container /health returned 200"
    else
        fail "Container /health returned HTTP $HTTP_CODE (expected 200)"
    fi

    # Test reset
    RESET_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST http://localhost:7861/reset \
        -H "Content-Type: application/json" \
        -d '{"task_id":"supplier_triage","seed":42}' 2>/dev/null || echo "000")
    [ "$RESET_CODE" = "200" ] && pass "Container /reset returned 200" || fail "Container /reset returned HTTP $RESET_CODE"

    docker stop "$IMAGE_NAME-test" > /dev/null 2>&1 || true
    docker rm "$IMAGE_NAME-test" > /dev/null 2>&1 || true
fi

echo ""

# ── Check 7: HF Space (if URL provided and not localhost) ────────────────────
if [[ "$SPACE_URL" != *"localhost"* ]]; then
    info "Checking HF Space at $SPACE_URL..."
    HF_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SPACE_URL/health" --max-time 30 2>/dev/null || echo "000")
    [ "$HF_CODE" = "200" ] && pass "HF Space is live and healthy" || fail "HF Space returned HTTP $HF_CODE"

    RESET_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "$SPACE_URL/reset" \
        -H "Content-Type: application/json" \
        -d '{"task_id":"supplier_triage","seed":42}' --max-time 30 2>/dev/null || echo "000")
    [ "$RESET_CODE" = "200" ] && pass "HF Space /reset works" || fail "HF Space /reset returned HTTP $RESET_CODE"
else
    info "Skipping HF Space check (using localhost). Pass your HF Space URL as first argument."
fi

echo ""

# ── Check 8: inference.py env variable requirements ──────────────────────────
info "Checking inference.py for required env variables..."
python3 -c "
import ast, sys
with open('inference.py') as f:
    src = f.read()
required = ['API_BASE_URL', 'MODEL_NAME', 'HF_TOKEN']
missing = [v for v in required if v not in src]
if missing:
    print(f'Missing: {missing}')
    sys.exit(1)
# Check for [START], [STEP], [END] format
for marker in ['[START]', '[STEP]', '[END]']:
    if marker not in src:
        print(f'Missing stdout marker: {marker}')
        sys.exit(1)
print('All env variables and markers present')
" && pass "inference.py has required env vars and stdout markers" || fail "inference.py missing required env vars or markers"

echo ""

# ── Summary ────────────────────────────────────────────────────────────────
echo "=============================="
echo "  VALIDATION SUMMARY"
echo "=============================="
echo -e "  ${GREEN}PASSED: $PASSED${NC}"
echo -e "  ${RED}FAILED: $FAILED${NC}"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED — Ready to submit!${NC}"
    exit 0
else
    echo -e "${RED}✗ $FAILED check(s) failed — Fix before submitting.${NC}"
    exit 1
fi
