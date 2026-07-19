#!/usr/bin/env bash
# Lore soak + chaos test: periodic SIGKILL/restart under continuous load and
# a final data-integrity check.
#
# Usage:
#   cargo build --release
#   SOAK_SECS=300 KILL_EVERY=60 ./scripts/soak.sh
#
# Env:
#   SOAK_SECS   total duration (default 300)
#   KILL_EVERY  chaos period — SIGKILL + restart at this interval (default 60)
#   PORT        service port (default 13990)
#   BIN         binary path (default ./target/release/lore)
#
# Output: request counts, error count, ask p50/p95, restart count and an integrity
# check (agent list + exportability + /ready). Failure = exit 1.
set -euo pipefail

SOAK_SECS=${SOAK_SECS:-300}
KILL_EVERY=${KILL_EVERY:-60}
PORT=${PORT:-13990}
BIN=${BIN:-./target/release/lore}
BASE="http://127.0.0.1:$PORT"
KEY="soak-key"
DATA=$(mktemp -d)
LAT_FILE="$DATA/lat.txt"
ERR_FILE="$DATA/err.txt"
: >"$LAT_FILE"; : >"$ERR_FILE"
SRV_PID=""

log() { echo "[soak $(date +%H:%M:%S)] $*"; }

start_server() {
  LORE_DATA="$DATA" LORE_API_KEY="$KEY" LORE_LOG=warn LORE_CONSOLIDATE_SECS=20 \
    "$BIN" serve --addr "127.0.0.1:$PORT" >>"$DATA/server.log" 2>&1 &
  SRV_PID=$!
}

wait_ready() {
  for _ in $(seq 1 200); do
    if curl -sf "$BASE/ready" >/dev/null 2>&1; then return 0; fi
    sleep 0.05
  done
  log "ERROR: service did not become ready"; exit 1
}

cleanup() {
  [ -n "$SRV_PID" ] && kill -9 "$SRV_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

command -v curl >/dev/null || { echo "curl required"; exit 1; }
[ -x "$BIN" ] || { echo "binary missing: $BIN (first: cargo build --release)"; exit 1; }

log "starting: duration=${SOAK_SECS}s chaos=${KILL_EVERY}s data=$DATA"
start_server; wait_ready

# Team: 3 agents.
AGENT_IDS=()
for name in Aria Kai Sage; do
  id=$(curl -sf -H "x-api-key: $KEY" -H 'content-type: application/json' \
    -d "{\"name\":\"$name\",\"role\":\"soaker\"}" "$BASE/agents" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
  AGENT_IDS+=("$id")
done
log "team ready: ${AGENT_IDS[*]}"

END=$((SECONDS + SOAK_SECS))
NEXT_KILL=$((SECONDS + KILL_EVERY))
REQS=0; RESTARTS=0

while [ $SECONDS -lt $END ]; do
  ID=${AGENT_IDS[$((REQS % 3))]}
  # ask (latency-measured, session-based)
  T=$(curl -sf -o /dev/null -w '%{time_total}' -H "x-api-key: $KEY" -H 'content-type: application/json' \
    -d "{\"message\":\"soak note $REQS: blue door $((REQS % 50))\",\"session\":\"s$((REQS % 20))\"}" \
    "$BASE/agents/$ID/ask" 2>>"$ERR_FILE") && echo "$T" >>"$LAT_FILE" || echo "ask" >>"$ERR_FILE"
  # recall + board (every 5 requests)
  if [ $((REQS % 5)) -eq 0 ]; then
    curl -sf -H "x-api-key: $KEY" "$BASE/agents/$ID/recall?q=blue" >/dev/null 2>>"$ERR_FILE" || echo "recall" >>"$ERR_FILE"
    curl -sf -H "x-api-key: $KEY" "$BASE/board?limit=5" >/dev/null 2>>"$ERR_FILE" || echo "board" >>"$ERR_FILE"
  fi
  # deliberate (every 20 requests)
  if [ $((REQS % 20)) -eq 0 ]; then
    curl -sf -H "x-api-key: $KEY" -H 'content-type: application/json' \
      -d '{"question":"soak summary?"}' "$BASE/deliberate" >/dev/null 2>>"$ERR_FILE" || echo "deliberate" >>"$ERR_FILE"
  fi
  REQS=$((REQS + 1))

  # CHAOS: periodic hard kill + respawn.
  if [ $SECONDS -ge $NEXT_KILL ] && [ $SECONDS -lt $END ]; then
    log "CHAOS: SIGKILL + restart (#$((RESTARTS + 1)))"
    kill -9 "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    start_server; wait_ready
    RESTARTS=$((RESTARTS + 1))
    NEXT_KILL=$((SECONDS + KILL_EVERY))
  fi
done

log "load finished: requests=$REQS restarts=$RESTARTS"

# ── Integrity check ───────────────────────────────────────────────────────
FAIL=0

# 1) Service still ready.
curl -sf "$BASE/ready" >/dev/null || { log "ERROR: /ready failed"; FAIL=1; }

# 2) Agents (persona persistence) intact.
COUNT=$(curl -sf -H "x-api-key: $KEY" "$BASE/agents" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)))')
[ "$COUNT" -eq 3 ] || { log "ERROR: agent count $COUNT != 3"; FAIL=1; }

# 3) Memory is exportable + non-empty (SQLite integrity — despite the kills).
kill -9 "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; SRV_PID=""
LORE_DATA="$DATA" LORE_LOG=error "$BIN" export --out "$DATA/dump.json"
MEMS=$(python3 -c "import json;print(len(json.load(open('$DATA/dump.json'))))")
[ "$MEMS" -gt 0 ] || { log "ERROR: export empty"; FAIL=1; }

# 4) Error-rate report (drops during chaos restarts are expected —
#    threshold: ~2 dropped requests per restart, as a % of requests).
ERRS=$(wc -l <"$ERR_FILE")
ALLOW=$(( (RESTARTS + 1) * 4 ))
[ "$ERRS" -le "$ALLOW" ] || { log "ERROR: error count $ERRS > allowed $ALLOW"; FAIL=1; }

# 5) Latency report.
python3 - "$LAT_FILE" <<'EOF'
import sys
xs = sorted(float(l) for l in open(sys.argv[1]) if l.strip())
if xs:
    p = lambda q: xs[min(len(xs)-1, int(q*len(xs)))]
    print(f"[soak] ask latency: n={len(xs)} p50={p(0.5)*1000:.1f}ms p95={p(0.95)*1000:.1f}ms max={xs[-1]*1000:.1f}ms")
EOF

log "requests=$REQS restarts=$RESTARTS errors=$ERRS memories=$MEMS"
if [ "$FAIL" -eq 0 ]; then
  log "RESULT: INTEGRITY INTACT ✓"
else
  log "RESULT: FAILED ✗"; exit 1
fi
