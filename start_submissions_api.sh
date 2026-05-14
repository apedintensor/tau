#!/bin/bash
exec doppler run -p arbos -c dev -- \
  /home/const/subnet66/.venv/bin/python -m cli serve-submissions-api \
  --host 127.0.0.1 \
  --port 8066 \
  --base-agent /home/const/subnet66/ninja/agent.py \
  --private-submission-root /home/const/subnet66/tau/workspace/validate/netuid-66/private-submissions \
  --max-request-bytes 5000000 \
  --max-agent-bytes 5000000 \
  --rate-limit-max-requests 6 \
  --rate-limit-max-failures 3 \
  --network finney
