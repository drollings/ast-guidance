#!/usr/bin/env bash
# Poll a coral-router health URL until it answers 2xx or a timeout elapses.
# On timeout, tail the router's log and fail loudly (make aborts the target).
#
# Usage: router-wait-health.sh <health-url> [timeout_s] [logfile]
#   timeout_s  default 30
#   logfile    default /tmp/coral-router.out (tailed on failure)
set -euo pipefail

URL="${1:?usage: router-wait-health.sh <health-url> [timeout_s] [logfile]}"
TIMEOUT="${2:-30}"
LOG="${3:-/tmp/coral-router.out}"

if ! command -v curl > /dev/null 2>&1; then
    echo "ERROR: curl not found" >&2
    exit 1
fi

for ((i = 1; i <= TIMEOUT; i++)); do
    if curl -sf -m 1 "$URL" > /dev/null 2>&1; then
        echo "coral-router healthy at ${URL} (attempt ${i}/${TIMEOUT})"
        exit 0
    fi
    sleep 1
done

echo "ERROR: coral-router did not become healthy at ${URL} within ${TIMEOUT}s" >&2
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
    echo "--- tail of ${LOG} ---" >&2
    tail -40 "$LOG" >&2
fi
exit 1
