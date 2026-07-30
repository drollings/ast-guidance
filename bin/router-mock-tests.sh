#!/usr/bin/env bash
set -euo pipefail

# Resolve the router URL from the config file (source of truth).
# Override with ROUTER_BASE_URL env var if set.
CONFIG_FILE="${1:-env/coral-router.json}"
if [ -n "${ROUTER_BASE_URL:-}" ]; then
    BASE="$ROUTER_BASE_URL"
elif [ -f "$CONFIG_FILE" ]; then
    BIND_ADDR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['server']['bind_addr'])" 2>/dev/null || echo "127.0.0.1:8079")
    BASE="http://$BIND_ADDR"
else
    BASE="http://127.0.0.1:8079"
fi

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

check_model() {
    local label="$1" model="$2" content="$3" expected="$4"
    check "$label" POST "$BASE/v1/chat/completions" \
        "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$content\"}]}" \
        "$expected"
}

check_json_model() {
    local label="$1" model="$2" content="$3" expected="$4"
    local http_code resp
    resp=$(curl -s -m 30 -w "\n%{http_code}" -X POST "$BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$content\"}]}" 2>&1)
    http_code=$(echo "$resp" | tail -1)
    resp=$(echo "$resp" | sed '$d')
    if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); $expected" 2>/dev/null; then
        echo -e "  ${GREEN}PASS${NC} [$http_code] $label"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC} [$http_code] $label"
        echo "       response: $(echo "$resp" | head -c 200)"
        FAIL=$((FAIL + 1))
    fi
}

# ── Health / Stats ──────────────────────────────────────────────────

check "health"           GET  "$BASE/health"                              ''     'ok'
check "stats"            GET  "$BASE/stats"                               ''     'requests'

# ── Model-name coverage — every model in env/coral-router.json ──────

check_model "model: tiny"           "tiny"          "What is 2+2?"                      '4'
check_model "model: fast"           "fast"          "What is 2+2?"                      '4'
check_model "model: qwythos-9b"     "qwythos-9b"    "What is 2+2?"                      '4'
check_model "model: prose"          "prose"         "What is 2+2?"                      '4'
check_model "model: translation"    "translation"   "What is 2+2?"                      '4'

# ── Pipeline-routed model names (go through classifier -> dispatch) ─

check_model "route: local"          "local"         "What is 2+2?"                      '4'
check_model "route: code"           "code"          "Write a Rust function to compute Fibonacci numbers."  'fn fibonacci'

# ── Mock transcript coverage (all 8 transcript entries) ─────────────

check_model "transcript: 2+2"       "local"         "What is 2+2?"                      '4'
check_model "transcript: capital"   "local"         "What is the capital of France?"     'Paris'
check_model "transcript: sky color" "local"         "What color is the sky?"             'sky'
check_model "transcript: greeting"  "local"         "hi"                                'Hello'

# ── Command dispatch (deterministic pre-filter) ─────────────────────

check_model "cmd: /help"            "local"         "/help"                             'help'
check_model "cmd: /checkpoint"      "local"         "/checkpoint snap1"                 'checkpoint'
check_model "cmd: /stats"           "local"         "/stats"                            'stats'
check_model "cmd: unknown"          "local"         "/nonexistent"                      'unknown'

# ── PII detection ───────────────────────────────────────────────────

check_model "PII: ssn"              "local"         "My SSN is 123-45-6789"             'blocked'
check_model "PII: email"            "local"         "Email me@test.com please"          'email'
check_model "PII: api_key (hard)"   "local"         "api_key=sk-abcdefghijklmnop123456" 'api_key'

# ── Error cases ─────────────────────────────────────────────────────

check "empty messages"    POST "$BASE/v1/chat/completions" \
  '{"model":"local","messages":[]}'   'error'

check "empty request body" POST "$BASE/v1/chat/completions" \
  ''   'error'

check "bad JSON"          POST "$BASE/v1/chat/completions" \
  'not json'                         'error'

# ── 404 ─────────────────────────────────────────────────────────────

resp=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "$BASE/nonexistent")
if [ "$resp" = "404" ]; then
    echo -e "  ${GREEN}PASS${NC} [404] unknown path returns 404"
    PASS=$((PASS + 1))
else
    echo -e "  ${RED}FAIL${NC} [${resp}] unknown path returns 404"
    FAIL=$((FAIL + 1))
fi

# ── Streaming flag preserved ────────────────────────────────────────

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

# ── Model routing verification ──────────────────────────────────────

check_json_model "routing: model=local" "local" "hi" \
  "assert d['model']=='local'"

check_json_model "routing: model=code" "code" "Write a Rust function to compute Fibonacci numbers." \
  "assert d.get('model') in ('code','fast','local'), f'unexpected model: {d.get(\"model\")}'"

check_json_model "routing: model=tiny" "tiny" "What is 2+2?" \
  "assert d['model'] in ('tiny','local'), f'unexpected model: {d[\"model\"]}'"

check_json_model "routing: model=fast" "fast" "What is 2+2?" \
  "assert d['model'] in ('fast','local'), f'unexpected model: {d[\"model\"]}'"

# ── Summary ─────────────────────────────────────────────────────────

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && echo -e "${GREEN}All router-mock tests passed.${NC}" || echo -e "${RED}Some tests failed.${NC}"
exit "$FAIL"
