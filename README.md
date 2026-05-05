# tau

`tau` is a small CLI for running a staged SWE workflow:

1. `generate` mines a commit and creates a task.
2. `solve` runs a solver against that task.
3. `compare` scores two saved solutions by changed-line similarity.
4. `eval` compares multiple solutions with an LLM judge.
5. `delete` removes saved task artifacts.

## Miner Harness

The canonical miner-editable single-file harness now lives in the public
[`unarbos/ninja`](https://github.com/unarbos/ninja) repo. `tau` owns task
generation, Docker execution, validation, scoring, and managed inference; it no
longer tracks a root `agent.py` harness.

For local experiments, point `tau solve` at the public harness or at a local
checkout of `ninja`:

```bash
source .venv/bin/activate
tau solve --task my-task --solution ninja-main --agent unarbos/ninja
tau solve --task my-task --solution local-ninja --agent ../ninja
```

The file must define:

```python
def solve(repo_path: str, issue: str, model: str, api_base: str, api_key: str) -> dict:
    ...
```

It should return a dictionary with `patch`, `logs`, `steps`, `cost`, and
`success`. The validator owns the task repo, Docker sandbox, tests, scoring,
hidden tasks, and LLM routing; miners only patch `agent.py` in `ninja`.

The `model`, `api_base`, and `api_key` arguments are validator-managed. For Docker file solves, `api_base` points at the validator's OpenAI-compatible inference proxy, `api_key` is a per-run proxy token, and the proxy forwards to OpenRouter while enforcing request, token, cost, and model policy. Agents should not hardcode OpenRouter/OpenAI keys or call external LLM providers directly.

You can also pass any compatible GitHub repo for local/reproducible runs with
either a full GitHub URL or the `owner/repo` shorthand:

```bash
source .venv/bin/activate
tau solve --task my-task --solution shared --agent owner/repo
```

or:

```bash
source .venv/bin/activate
tau solve --task my-task --solution shared --agent https://github.com/owner/repo
```

Production miner submissions use the `ninja` PR flow described below instead of
direct `owner/repo@sha` commitments.

## Prerequisites

- Python 3.11+
- `uv`
- Docker
- A GitHub token for task generation
- An OpenRouter API key for Docker file solves and evaluation
- A Cursor API key for Cursor solves

## Setup

From the `tau/` directory:

```bash
source .venv/bin/activate
uv pip install -e .
```

Create a `.env` file in `tau/` if you do not already have one:

```bash
GITHUB_TOKEN=your_github_token
OPENROUTER_API_KEY=your_openrouter_api_key
CURSOR_API_KEY=your_cursor_api_key
```

`tau` loads `.env` automatically from the project root.

Optional environment defaults for centralized solver routing:

```bash
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
SOLVER_MAX_REQUESTS=40
SOLVER_MAX_TOTAL_TOKENS=200000
SOLVER_MAX_PROMPT_TOKENS=160000
SOLVER_MAX_COMPLETION_TOKENS=40000
SOLVER_MAX_TOKENS_PER_REQUEST=4096
SOLVER_MAX_COST=1.00
```

CLI flags still override these values for one-off runs.

## Validator GitHub PR Mode

The live validator can score miner edits from the public `unarbos/ninja` harness repo. In this mode, miners do not commit arbitrary GitHub repos directly. They open a PR against `unarbos/ninja`, then commit the PR head on-chain.

Miner commitment format:

```text
github-pr:unarbos/ninja#<pr-number>@<head-sha>
```

The PR title must start with the exact committing miner hotkey:

```text
<miner-hotkey> improve harness loop
```

The validator only queues the PR when all of these match:

- the commitment comes from a registered subnet hotkey
- the hotkey has not had another accepted commitment in the last 24h, measured as 7,200 chain blocks
- the watched repo is `unarbos/ninja` and the base branch is `main`
- the PR is open and not draft
- the PR title starts with the committing hotkey
- the committed SHA matches the current PR head SHA
- required GitHub checks are green: `PR Scope Guard` and `OpenRouter PR Judge`
- the PR head commit is publicly fetchable

GitHub PR mode uses 50 duel rounds minimum. If a run is configured lower, the
validator bumps it to 50 and raises the task pool target to match.

The production validator evaluates at most 10 queued candidates per epoch, in
queue order. Each duel can run up to 25 round workers with challenger agent
timeouts capped at 600 seconds. If a challenger hits 5 consecutive round
timeouts, the validator stops submitting new rounds for that challenger and
moves on after its already-running rounds finish.

When a PR challenger becomes king, the validator auto-merges that PR into the
watched `unarbos/ninja` base branch, records the king as the resulting base
repo commit while keeping the miner hotkey/PR metadata, flushes the old task
pool, and assigns all validator weight to the winning hotkey on the next
allowed weight-set epoch.

The background pool filler pre-solves tasks before challengers arrive. It caps
Cursor and king pool solves at 300 seconds, skips timed-out or empty Cursor
baselines, and the duel gatherer chooses the fastest eligible pool tasks first.
Once the pool is full, the production validator refreshes it by adding 5 new
valid tasks every hour; the normal prune step then removes the oldest 5 so the
pool stays at the configured target size.

`start_validator.sh` enables this production path with:

```bash
--round-concurrency 25 \
--candidates-per-epoch 10 \
--candidate-timeout-streak-limit 5 \
--watch-github-prs \
--github-pr-only \
--github-pr-repo unarbos/ninja \
--github-pr-base main
```

`--github-pr-only` means normal `unarbos/ninja@sha` commitments are ignored by the live validator. This keeps miner submissions tied to PR review, CI, and the committing hotkey.

## Validator Duel Scoring

Each validation task still starts from a mined GitHub commit: `task/original` is the repo before the commit, `task/reference` is the repo after it, and `task/reference.patch` is used to filter out tiny tasks.

For duels, the scoring target is the Cursor baseline solution, saved as `solutions/baseline`. The pool filler runs Cursor and the current king on the same task, then stores the king's similarity to `baseline`. During a duel, the challenger is also compared to `baseline`.

Round score is now blended: 1/2 Cursor-baseline similarity plus 1/2 LLM diff judgment. The diff judge uses `deepseek/deepseek-v4-flash` through OpenRouter at temperature 0 with medium reasoning effort and a 16000-token output cap, then scores the king and challenger patches against the task/reference context.

Cursor is only the measuring stick. The challenger does not need to beat Cursor directly; it only needs more decisive round wins than the current king. The live validator uses `--win-margin 0`, so one more challenger win than king win is enough.

The validator still compares `king` to `challenger` separately for copy detection, but that pairwise similarity does not replace the Cursor baseline scoring target.

## Managed Inference Policy

Docker file agents receive a validator-managed OpenAI-compatible endpoint through `solve(..., model, api_base, api_key)`. The upstream provider key is never passed into miner code.

The proxy forwards to OpenRouter and enforces:

- the validator-selected model, currently `deepseek/deepseek-v4-flash` for solver inference unless overridden by validator config
- `temperature=0.0`
- `top_p=1.0`
- removal of miner-controlled sampling fields such as `top_k`, `seed`, penalties, `logit_bias`, and `logprobs`
- request, token, and cost budgets

Miner agents should use only the supplied `api_base` and `api_key`. Attempts to choose another provider, model, sampling policy, or credential path are rejected by `ninja` CI and overwritten or stripped by the validator proxy.

## Basic Usage

Show top-level help:

```bash
source .venv/bin/activate
tau --help
```

All commands write their artifacts under:

```text
workspace/tasks/
```

You can override that with `--workspace-root /path/to/root`.

## Generate A Task

```bash
source .venv/bin/activate
tau generate --task my-task
```

Useful options:

- `--generator-model <model>`
- `--seed <int>`
- `--max-mining-attempts <int>`
- `--agent-timeout <seconds>`
- `--debug`

## Solve A Task

`solve` supports multiple backends. The `--agent` value can be:

- `cursor` to run the Cursor CLI in Docker
- `claude` to run the local Claude CLI on the host
- a local `agent.py` file for the Docker file solver
- a local repo root containing `agent.py` for the Docker file solver
- a GitHub repo URL or shorthand like `owner/repo` for the Docker file solver

Example using Cursor:

```bash
source .venv/bin/activate
tau solve --task my-task --solution cursor-run --agent cursor
```

Example using Claude:

```bash
source .venv/bin/activate
tau solve --task my-task --solution claude-run --agent claude
```

Example using the public `ninja` harness:

```bash
source .venv/bin/activate
tau solve --task my-task --solution baseline --agent unarbos/ninja
```

Example using a local checkout of `ninja`:

```bash
source .venv/bin/activate
tau solve --task my-task --solution baseline --agent ../ninja
```

Useful options:

- `--solver-model <model>`
- `--solver-max-requests <int>`
- `--solver-max-total-tokens <int>`
- `--solver-max-cost <float>`
- `--docker-solver-memory 2g`
- `--docker-solver-cpus 2`
- `--docker-solver-no-cache`
- `--agent-timeout <seconds>`
- `--debug`

## Compare Solutions

Compare two saved solutions using changed-lines-only similarity:

```bash
source .venv/bin/activate
tau compare --task my-task --solutions cursor-run baseline
```

Comma-separated values also work:

```bash
source .venv/bin/activate
tau compare --task my-task --solutions cursor-run,baseline
```

## Evaluate Solutions

Compare two or more solutions for the same task:

```bash
source .venv/bin/activate
tau eval --task my-task --solutions baseline candidate-a candidate-b
```

Comma-separated values also work:

```bash
source .venv/bin/activate
tau eval --task my-task --solutions baseline,candidate-a,candidate-b
```

Useful options:

- `--eval-model <model>`
- `--seed <int>`
- `--agent-timeout <seconds>`
- `--debug`

## Delete Saved Artifacts

Delete one task:

```bash
source .venv/bin/activate
tau delete --task my-task
```

Delete all saved tasks:

```bash
source .venv/bin/activate
tau delete task --all
```

## End-To-End Example

```bash
source .venv/bin/activate
tau generate --task demo-task
tau solve --task demo-task --solution run-1 --agent cursor
tau solve --task demo-task --solution run-2 --agent unarbos/ninja
tau compare --task demo-task --solutions run-1 run-2
tau eval --task demo-task --solutions run-1 run-2
```

## Single-File Agent In Docker

When you pass a local file, local repo directory, or GitHub repo to `--agent`, tau builds a small Python Docker image, imports `agent.py`, and calls its `solve(...)` function.

### What happens

1. A Docker image (`swe-eval/file-solver:<hash>`) is built from `python:3.11-slim`.
2. A container starts with resource limits (memory, CPU, pids, tmpfs).
3. The task repo is copied into the container at `/work/repo`.
4. The submitted `agent.py` is copied into the container and imported.
5. The validator calls `solve(repo_path="/work/repo", issue=..., model=..., api_base=..., api_key=...)` with the managed model id, local proxy URL, and per-run proxy token.
6. The diff is collected from the container and applied back to the host repo.
7. The container is torn down.

The submitted agent does not receive the upstream OpenRouter key. On Linux the solver container runs with Docker network disabled and reaches the validator proxy through a local socket bridge, so LLM calls flow through one managed endpoint.

## Cursor Agent In Docker

When you pass `--agent cursor`, tau builds a Docker image, runs the Cursor CLI inside it, and collects the resulting diff.

### What happens

1. A Docker image (`swe-eval/cursor-solver:<hash>`) is built from `python:3.11-slim` with the Cursor CLI installed via `curl https://cursor.com/install | bash`.
2. A container starts with resource limits (memory, CPU, pids, tmpfs).
3. The task repo is copied into the container at `/work/repo` and the prompt is written to `/work/task.txt`.
4. The Cursor `agent` CLI runs inside the container with `CURSOR_API_KEY` injected:

```bash
agent -p --force --trust --sandbox disabled --output-format stream-json \
    --workspace /work/repo "$PROMPT"
```

5. The diff is collected from the container and applied back to the host repo.
6. The container is torn down.

### Usage

```bash
source .venv/bin/activate
tau solve --task my-task --solution cursor-run --agent cursor
```

`CURSOR_API_KEY` must be set in your environment or in `tau/.env`.

### Docker options

| Flag | Purpose |
|------|---------|
| `--solver-model <model>` | Override the model used by Cursor |
| `--agent-timeout <seconds>` | Time limit for the solve |
| `--docker-solver-memory 2g` | Container memory limit |
| `--docker-solver-cpus 2` | Container CPU limit |
| `--docker-solver-no-cache` | Force rebuild the Docker image |
| `--debug` | Enable debug logging |

## Notes

- `generate` needs `GITHUB_TOKEN` or `GH_TOKEN`.
- `tau solve --agent cursor` needs `CURSOR_API_KEY` and Docker.
- `tau solve --agent claude` needs the `claude` CLI installed on the host.
- Docker file solves and `eval` need `OPENROUTER_API_KEY`.
- `compare` reads saved solution artifacts and does not call a model.
- Docker-backed solves use Docker, so Docker must be installed and running.
- Generated task, solution, and evaluation paths are printed by the CLI after each command finishes.
