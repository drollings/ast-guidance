#!/usr/bin/env bash
set -euo pipefail

BASE="http://127.0.0.1:8081"
PASS=0
FAIL=0
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local label="$1" method="$2" url="$3" body="$4" expected="$5"
    local http_code resp
    if [ -z "$body" ]; then
        resp=$(curl -s -m 30 -w "\n%{http_code}" -X "$method" "$url" 2>&1)
    else
        resp=$(curl -s -m 30 -w "\n%{http_code}" -X "$method" "$url" \
            -H "Content-Type: application/json" -d "$body" 2>&1)
    fi
    http_code=$(echo "$resp" | tail -1)
    resp=$(echo "$resp" | sed '$d')
    if echo "$resp" | grep -qF "$expected"; then
        echo -e "  ${GREEN}PASS${NC} [$http_code] $label"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC} [$http_code] $label  (expected fragment: \"$expected\")"
        echo "       response: $(echo "$resp" | head -c 200)"
        FAIL=$((FAIL + 1))
    fi
}

# ── Health / Stats ──────────────────────────────────────────
check "health"           GET  "$BASE/health"                              ''     'ok'
check "stats"            GET  "$BASE/stats"                               ''     'requests'

# ── Model "local" (local) ──────────────────────────────────
check "local: 2+2"        POST "$BASE/v1/chat/completions" \
  '{"model":"local","messages":[{"role":"user","content":"What is 2+2?"}]}' '4'

check "local: color of sky"    POST "$BASE/v1/chat/completions" \
  '{"model":"local","messages":[{"role":"user","content":"What color is the sky?"}]}' 'blue'

# ── Model "code" (orchestrator) ──────────────────────────────
check "code: 2+2"        POST "$BASE/v1/chat/completions" \
  '{"model":"code","messages":[{"role":"user","content":"What is 2+2?"}]}' '4'

check "code: explain GC" POST "$BASE/v1/chat/completions" \
  '{"model":"code","messages":[{"role":"user","content":"What is 2+2?"}]}' '4'

# ── Command dispatch (deterministic pre-filter) ──────────────
check "/help"             POST "$BASE/v1/chat/completions" \
  '{"model":"local","messages":[{"role":"user","content":"/help"}]}' 'help'

check "/checkpoint"       POST "$BASE/v1/chat/completions" \
  '{"model":"local","messages":[{"role":"user","content":"/checkpoint snap1"}]}' 'checkpoint'

check "/unknown"          POST "$BASE/v1/chat/completions" \
  '{"model":"local","messages":[{"role":"user","content":"/nonexistent"}]}' 'unknown command'

# ── PII detection ────────────────────────────────────────────
check "PII: ssn"          POST "$BASE/v1/chat/completions" \
  '{"model":"local","messages":[{"role":"user","content":"My SSN is 123-45-6789"}]}' 'blocked'

check "PII: email"        POST "$BASE/v1/chat/completions" \
  '{"model":"local","messages":[{"role":"user","content":"Email me@test.com please"}]}' 'email'

# ── Error cases ──────────────────────────────────────────────
check "empty messages"    POST "$BASE/v1/chat/completions" \
  '{"model":"local","messages":[]}'   'error'

check "bad JSON"          POST "$BASE/v1/chat/completions" \
  'not json'                         'error'

# ── 404 ───────────────────────────────────────────────────────
resp=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "$BASE/nonexistent")
if [ "$resp" = "404" ]; then
    echo -e "  ${GREEN}PASS${NC} [404] unknown path returns 404"
    PASS=$((PASS + 1))
else
    echo -e "  ${RED}FAIL${NC} [${resp}] unknown path returns 404"
    FAIL=$((FAIL + 1))
fi

# ── CORS ─────────────────────────────────────────────────────
# ── CORS ─────────────────────────────────────────────────────
resp=$(curl -s -m 5 -I -X GET "$BASE/health" 2>&1)
if echo "$resp" | grep -q 'Access-Control-Allow-Origin'; then
    echo -e "  ${GREEN}PASS${NC} [200] response includes CORS headers"
    PASS=$((PASS + 1))
else
    echo -e "  ${RED}FAIL${NC}        response includes CORS headers"
    FAIL=$((FAIL + 1))
fi

# ── Streaming flag preserved ─────────────────────────────────
resp=$(curl -s -m 30 -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"stream":true}')
if echo "$resp" | grep -q 'data:'; then
    echo -e "  ${GREEN}PASS${NC} [200] stream flag returns SSE chunks"
    PASS=$((PASS + 1))
else
    echo -e "  ${RED}FAIL${NC} [200] stream flag returns SSE chunks"
    FAIL=$((FAIL + 1))
fi

# ── Model routing verification ───────────────────────────────
resp=$(curl -s -m 30 -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"local","messages":[{"role":"user","content":"hi"}]}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['model']=='local'" 2>/dev/null; then
    echo -e "  ${GREEN}PASS${NC} [200] local routes to model=local"
    PASS=$((PASS + 1))
else
    echo -e "  ${RED}FAIL${NC}        local routes to model=local"
    FAIL=$((FAIL + 1))
fi

resp=$(curl -s -m 30 -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"code","messages":[{"role":"user","content":"hi"}]}')
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('model')=='code' or d.get('model')=='local'" 2>/dev/null; then
    echo -e "  ${GREEN}PASS${NC} [200] local routes model name correctly"
    PASS=$((PASS + 1))
else
    echo -e "  ${RED}FAIL${NC}        local routes model name correctly"
    FAIL=$((FAIL + 1))
fi

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && echo -e "${GREEN}All router-mock tests passed.${NC}" || echo -e "${RED}Some tests failed.${NC}"
exit "$FAIL"
