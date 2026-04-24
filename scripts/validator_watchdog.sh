#!/usr/bin/env bash
# Validator wedge watchdog. Restarts pm2 'validator' if dashboard_data.json
# (the natural ~3s cadence heartbeat written by the publisher) has not been
# touched in WEDGE_THRESHOLD seconds. Default 1200s (20 min) -- normal
# duels can take ~10 min between dashboard publishes, so 20 min is a safe
# wedge signal.
set -euo pipefail
HEARTBEAT="${HEARTBEAT:-/home/const/subnet66/tau/workspace/validate/netuid-66/dashboard_data.json}"
WEDGE_THRESHOLD="${WEDGE_THRESHOLD:-2400}"
PM2_PROC="${PM2_PROC:-validator}"
LOG="${LOG:-/home/const/subnet66/tau/logs/watchdog.log}"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

while true; do
  if [[ -f "$HEARTBEAT" ]]; then
    mtime=$(stat -c '%Y' "$HEARTBEAT")
    now=$(date +%s)
    age=$(( now - mtime ))
    if (( age > WEDGE_THRESHOLD )); then
      echo "$(ts) WEDGE detected: $HEARTBEAT age=${age}s > ${WEDGE_THRESHOLD}s; restarting $PM2_PROC" >> "$LOG"
      pm2 restart "$PM2_PROC" --update-env >> "$LOG" 2>&1 || echo "$(ts) pm2 restart failed" >> "$LOG"
      sleep 120  # cooldown so we don't loop-restart while warming up
    fi
  fi
  sleep 60
done
