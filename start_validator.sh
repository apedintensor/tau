#!/bin/bash
exec /home/const/subnet66/.venv/bin/python -m cli validate \
  --workspace-root /home/const/subnet66/tau \
  --wallet-name sn66_owner \
  --wallet-hotkey default \
  --max-concurrency 1 \
  --round-concurrency 50 \
  --task-pool-target 50 \
  --task-pool-refresh-count 5 \
  --task-pool-refresh-interval-seconds 3600 \
  --duel-rounds 50 \
  --win-margin 0 \
  --min-commitment-block 7951985 \
  --pool-filler-concurrency 24 \
  --watch-github-prs \
  --github-pr-only \
  --github-pr-repo unarbos/ninja \
  --github-pr-base main
