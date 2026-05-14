from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

from config import RunConfig, SolverAgentSource
from pipeline import (
    compare_task_run,
    delete_task_run,
    evaluate_task_run,
    generate_task_run,
    solve_task_run,
)

_DEFAULT_CONCURRENCY = min(os.cpu_count() or 4, 8)
_DEFAULT_AGENT_FILE = "agent.py"
_PRIVATE_SUBMISSION_JUDGE_MODEL = "openai/gpt-5.4"
log = logging.getLogger("swe-eval.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, solve, compare, and evaluate SWE tasks as independent stages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Mine a commit and materialize a named task.")
    _add_shared_args(generate)
    generate.add_argument("--task", required=True, help="Unique name for the generated task.")
    generate.add_argument("--generator-model", help="Optional model override for task generation.")
    generate.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for repeatable mining order.",
    )
    generate.add_argument(
        "--max-mining-attempts",
        type=int,
        default=25,
        help="How many GitHub event/commit retries to attempt while mining a task.",
    )

    solve = subparsers.add_parser("solve", help="Run a named task with a named solver agent.")
    _add_shared_args(solve)
    solve.add_argument("--task", required=True, help="Existing generated task name.")
    solve.add_argument("--solution", required=True, help="Unique name for this solver run.")
    solve.add_argument(
        "--agent",
        required=True,
        help=(
            "Solver backend selector. Use 'cursor' for the Cursor CLI, "
            "'claude' for the host Claude CLI, "
            "'claw' for the host Claw CLI, "
            "or pass a local agent.py file / repo root / GitHub repo URL for the Docker file solver."
        ),
    )
    _add_solver_args(solve)

    evaluate = subparsers.add_parser("eval", help="Evaluate ordered solution pairs for one named task.")
    _add_shared_args(evaluate)
    evaluate.add_argument("--task", required=True, help="Existing generated task name.")
    evaluate.add_argument(
        "--solutions",
        required=True,
        nargs="+",
        help="Ordered solution names to compare. Supports '--solutions A B' and '--solutions A,B'.",
    )
    evaluate.add_argument("--eval-model", help="Optional model override for evaluation.")
    evaluate.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for deterministic blind-candidate ordering.",
    )

    compare = subparsers.add_parser("compare", help="Compare two saved solutions by changed-line similarity.")
    _add_shared_args(compare)
    compare.add_argument("--task", required=True, help="Existing generated task name.")
    compare.add_argument(
        "--solutions",
        required=True,
        nargs="+",
        help="Exactly two solution names to compare. Supports '--solutions A B' and '--solutions A,B'.",
    )

    delete = subparsers.add_parser("delete", help="Delete saved task workspaces and related artifacts.")
    _add_shared_args(delete)
    delete.add_argument(
        "resource",
        nargs="?",
        choices=["task"],
        help="Optional resource type. Use 'task' for forms like 'tau delete task --all'.",
    )
    delete_group = delete.add_mutually_exclusive_group(required=True)
    delete_group.add_argument("--task", help="Delete one saved task by name.")
    delete_group.add_argument("--all", action="store_true", help="Delete all saved task workspaces.")

    private_submit = subparsers.add_parser(
        "private-submit",
        help="Validate and store a signed private miner agent.py submission bundle.",
    )
    _add_shared_args(private_submit)
    private_submit.add_argument("--hotkey", required=True, help="Miner hotkey that signed this submission.")
    private_submit.add_argument("--agent", required=True, type=Path, help="Private submitted agent.py file.")
    private_submit.add_argument("--base-agent", required=True, type=Path, help="Current public base agent.py to diff against.")
    private_submit.add_argument("--signature", required=True, help="Hotkey signature over the printed signature payload.")
    private_submit.add_argument("--submission-id", help="Optional stable submission id. Defaults to hotkey/hash derived id.")
    private_submit.add_argument("--private-submission-root", type=Path, help="Directory where private bundles are stored.")
    private_submit.add_argument("--netuid", type=int, default=66, help="Netuid used only to derive the default private submission root.")
    private_submit.add_argument("--network", help="Optional Bittensor network name or websocket endpoint for registration lookup.")
    private_submit.add_argument(
        "--subtensor-endpoint",
        help="Optional websocket endpoint that overrides --network for private submission registration lookup.",
    )
    private_submit.add_argument(
        "--registration-block",
        type=int,
        help="Known current registration block for the hotkey. If omitted, private-submit looks it up on chain.",
    )
    private_submit.add_argument("--overwrite", action="store_true", help="Allow replacing an existing submission bundle id.")
    private_submit.add_argument("--skip-openrouter-judge", action="store_true", help="Run only local smoke/scope checks; accepted will remain false.")
    private_submit.add_argument("--judge-model", help="OpenRouter model for the private submission judge.")
    private_submit.add_argument("--judge-min-score", type=int, default=70, help="Minimum OpenRouter judge score required.")

    serve_submissions_api = subparsers.add_parser(
        "serve-submissions-api",
        help="Serve the private miner submissions API.",
    )
    _add_shared_args(serve_submissions_api)
    serve_submissions_api.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    serve_submissions_api.add_argument("--port", type=int, default=8066, help="Port to bind.")
    serve_submissions_api.add_argument("--base-agent", required=True, type=Path, help="Current public base agent.py.")
    serve_submissions_api.add_argument("--private-submission-root", type=Path, help="Directory where private bundles are stored.")
    serve_submissions_api.add_argument("--netuid", type=int, default=66, help="Subnet netuid.")
    serve_submissions_api.add_argument("--network", help="Optional Bittensor network name or websocket endpoint.")
    serve_submissions_api.add_argument(
        "--subtensor-endpoint",
        help="Optional websocket endpoint that overrides --network for registration lookup.",
    )
    serve_submissions_api.add_argument("--overwrite", action="store_true", help="Allow replacing an existing submission bundle id.")
    serve_submissions_api.add_argument("--skip-openrouter-judge", action="store_true", help="Run without the OpenRouter judge; submissions will not be accepted.")
    serve_submissions_api.add_argument("--judge-model", help="OpenRouter model for the private submission judge.")
    serve_submissions_api.add_argument("--judge-min-score", type=int, default=70, help="Minimum OpenRouter judge score required.")
    serve_submissions_api.add_argument("--max-request-bytes", type=int, default=5_000_000, help="Maximum POST body size.")
    serve_submissions_api.add_argument("--max-agent-bytes", type=int, default=5_000_000, help="Maximum submitted agent.py size.")
    serve_submissions_api.add_argument("--rate-limit-window-seconds", type=int, default=60, help="Per-IP rate-limit window.")
    serve_submissions_api.add_argument("--rate-limit-max-requests", type=int, default=6, help="Maximum submissions per IP per window.")
    serve_submissions_api.add_argument("--rate-limit-max-failures", type=int, default=3, help="Maximum failed submissions per IP per window.")

    validate = subparsers.add_parser(
        "validate",
        help="Run the live king-of-the-hill validator loop for accepted private submissions.",
    )
    _add_shared_args(validate)
    _add_solver_args(validate)
    validate.set_defaults(agent_timeout=1800, docker_solver_max_output_bytes=100000000)
    validate.add_argument("--netuid", type=int, default=66, help="Subnet netuid to validate.")
    validate.add_argument("--network", help="Optional Bittensor network name or websocket endpoint.")
    validate.add_argument(
        "--subtensor-endpoint",
        help="Optional websocket endpoint that overrides --network for chain access.",
    )
    validate.add_argument("-N", "--duel-rounds", type=int, default=50, help="Decisive rounds per duel.")
    validate.add_argument("-K", "--win-margin", type=int, default=0, help="Extra decisive round wins over the king required to dethrone.")
    validate.add_argument("--max-concurrency", type=int, default=1, help="Max parallel duels (1 = serialized).")
    validate.add_argument("--round-concurrency", type=int, default=25, help="Max parallel rounds within a single duel.")
    validate.add_argument("--candidate-timeout-streak-limit", type=int, default=5, help="Stop submitting new rounds for a challenger after this many consecutive round timeouts.")
    validate.add_argument("--task-pool-target", type=int, default=50, help="Pre-solved tasks to keep in pool.")
    validate.add_argument(
        "--task-pool-static",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep primary and retest pools as static fixed task sets for the current king; stale tasks are flushed instead of being refreshed in place.",
    )
    validate.add_argument("--pool-filler-concurrency", type=int, default=25, help="Parallel pool-filler threads.")
    validate.add_argument("--task-pool-refresh-count", type=int, default=0, help="Full-pool tasks to replace each refresh interval (0 disables refresh and keeps static pools).")
    validate.add_argument("--task-pool-refresh-interval-seconds", type=int, default=0, help="Seconds between full-pool refresh batches (0 disables refresh scheduling).")
    validate.add_argument(
        "--task-pool-fill-from-saved",
        action="store_true",
        help="Fill validator task pools by round-robin reusing saved task workspaces instead of fetching new tasks.",
    )
    validate.add_argument("--task-cleanup-min-age-seconds", type=int, default=3600, help="Minimum age before non-pool validate task dirs can be pruned.")
    validate.add_argument("--weight-interval-blocks", type=int, default=360, help="Blocks between weight sets.")
    validate.add_argument("--king-window-size", type=int, default=5, help="Number of recent kings to share emissions across (each gets 1/N).")
    validate.add_argument("--poll-interval-seconds", type=int, default=600, help="Seconds between chain submission refreshes.")
    validate.add_argument("--duel-timeout", type=int, default=7200, help="Max seconds a single duel may run before being cancelled.")
    validate.add_argument("--max-duels", type=int, help="Stop after this many completed duels.")
    validate.add_argument("--min-commitment-block", type=int, default=0, help="Ignore accepted submissions before this registration block (0 = auto-set to current block at startup).")
    validate.add_argument("--hotkey-spent-since-block", type=int, help="Block cutoff for hotkey-spent history.")
    validate.add_argument("--queue-size", type=int, help="Max queued challengers.")
    validate.add_argument("--publish-repo", help="Repository where winning private submissions are published, in owner/name form.")
    validate.add_argument("--publish-base", help="Base branch where winning private submissions are published.")
    validate.add_argument("--watch-private-submissions", action="store_true", default=None, help="Accept eligible private API submissions as validator challengers.")
    validate.add_argument("--private-submission-only", action="store_true", default=None, help="Use only private API submissions as miner submissions.")
    validate.add_argument("--private-submission-root", type=Path, help="Directory containing private submission bundles keyed by submission id.")
    validate.add_argument("--wallet-name", required=True, help="Wallet coldkey name.")
    validate.add_argument("--wallet-hotkey", required=True, help="Wallet hotkey name.")
    validate.add_argument("--wallet-path", help="Wallet path override.")

    restore_r2_kings = subparsers.add_parser(
        "restore-r2-kings",
        help="Republish the validator dashboard recent_kings window to R2 from saved state/duels without starting the validator loop.",
    )
    restore_r2_kings.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Root directory that contains workspace/validate/... artifacts.",
    )
    restore_r2_kings.add_argument("--netuid", type=int, default=66, help="Subnet netuid whose validator state should be read.")
    restore_r2_kings.add_argument("--count", type=int, default=5, help="How many recent real kings to publish into the dashboard window.")
    restore_r2_kings.add_argument(
        "--set-current-from-history",
        action="store_true",
        help="If state.json has no current king, publish the newest reconstructed king as current_king for this one-shot dashboard restore.",
    )
    restore_r2_kings.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging for the one-shot restore.",
    )
    return parser


def main() -> None:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "generate":
            result = generate_task_run(task_name=args.task, config=_build_generate_config(args))
            print(f"generated {result.task_name}: {result.repo}@{result.commit_sha[:12]}")
            print(result.task_root)
            return
        if args.command == "solve":
            result = solve_task_run(
                task_name=args.task,
                solution_name=args.solution,
                config=_build_solve_config(args),
            )
            status = "success" if result.success else "failed"
            print(
                f"solved {result.task_name}/{result.solution_name}: "
                f"{result.repo}@{result.commit_sha[:12]} -> {status}"
            )
            print(result.solution_root)
            return
        if args.command == "eval":
            result = evaluate_task_run(
                task_name=args.task,
                solution_names=_normalize_solution_names(args.solutions),
                config=_build_eval_config(args),
            )
            print(
                f"evaluated {result.task_name}/{result.eval_name}: "
                f"{result.repo}@{result.commit_sha[:12]} -> {result.comparison_count} comparisons"
            )
            print(result.eval_root)
            return
        if args.command == "compare":
            result = compare_task_run(
                task_name=args.task,
                solution_names=_normalize_compare_solution_names(args.solutions),
                config=_build_compare_config(args),
            )
            print(
                f"compared {result.task_name}/{result.comparison_name}: "
                f"{result.repo}@{result.commit_sha[:12]} -> "
                f"{result.matched_changed_lines}/{result.scored_positions} matching changed lines "
                f"({result.similarity_ratio:.2%})"
            )
            print(result.comparison_root)
            return
        if args.command == "delete":
            result = delete_task_run(
                task_name=getattr(args, "task", None),
                delete_all=getattr(args, "all", False),
                config=_build_delete_config(args),
            )
            if result.deleted_all:
                print(f"deleted {result.deleted_count} task workspace(s)")
            else:
                print(f"deleted task {result.deleted_tasks[0]}")
            return
        if args.command == "private-submit":
            _run_private_submit(args)
            return
        if args.command == "serve-submissions-api":
            _run_serve_submissions_api(args)
            return
        if args.command == "validate":
            from validate import validate_loop_run

            result = validate_loop_run(config=_build_validate_config(args))
            print(
                f"validate loop exited with king uid={result.king_uid} "
                f"hotkey={result.king_hotkey} repo={result.king_repo}"
            )
            print(result.validate_root)
            return
        if args.command == "restore-r2-kings":
            from validate import republish_recent_kings_dashboard_to_r2

            result = republish_recent_kings_dashboard_to_r2(
                config=_build_restore_r2_kings_config(args),
                count=args.count,
                set_current_from_history=args.set_current_from_history,
            )
            print(
                f"republished dashboard to R2 with {result['recent_king_count']} recent king(s); "
                f"current_king_uid={result['current_king_uid']}"
            )
            print(result["validate_root"])
            return
        parser.error(f"Unknown command: {args.command}")
    except Exception as exc:  # noqa: BLE001
        if getattr(args, "debug", False):
            raise
        parser.exit(1, f"error: {exc}\n")


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Root directory that will receive workspace/tasks/... artifacts.",
    )
    parser.add_argument(
        "--agent-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each model or solver invocation.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging for the selected stage.",
    )


def _build_generate_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        generator_model=_arg_or_env(args.generator_model, "GENERATOR_MODEL", "OPENROUTER_GENERATOR_MODEL"),
        agent_timeout=args.agent_timeout,
        random_seed=args.seed,
        max_mining_attempts=args.max_mining_attempts,
        debug=args.debug,
    )


def _build_solve_config(args: argparse.Namespace) -> RunConfig:
    solver_backend, agent_source = _resolve_solve_target(args.agent, cwd=Path.cwd())
    defaults = RunConfig()
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        solver_model=args.solver_model,
        baseline_model=_arg_or_env(args.baseline_model, "BASELINE_MODEL", "OPENROUTER_BASELINE_MODEL"),
        agent_timeout=args.agent_timeout,
        solver_max_requests=_arg_or_env_int(args.solver_max_requests, "SOLVER_MAX_REQUESTS"),
        solver_max_total_tokens=_arg_or_env_int(args.solver_max_total_tokens, "SOLVER_MAX_TOTAL_TOKENS"),
        solver_max_prompt_tokens=_arg_or_env_int(args.solver_max_prompt_tokens, "SOLVER_MAX_PROMPT_TOKENS"),
        solver_max_completion_tokens=_arg_or_env_int(args.solver_max_completion_tokens, "SOLVER_MAX_COMPLETION_TOKENS"),
        solver_max_cost=_arg_or_env_float(args.solver_max_cost, "SOLVER_MAX_COST"),
        solver_max_tokens_per_request=_arg_or_env_int(
            args.solver_max_tokens_per_request,
            "SOLVER_MAX_TOKENS_PER_REQUEST",
        ),
        solver_provider_sort=_arg_or_env(args.solver_provider_sort, "SOLVER_PROVIDER_SORT", "OPENROUTER_PROVIDER_SORT"),
        solver_provider_only=_arg_or_env(args.solver_provider_only, "SOLVER_PROVIDER_ONLY", "OPENROUTER_PROVIDER_ONLY"),
        solver_provider_allow_fallbacks=(
            False if args.solver_provider_disable_fallbacks else defaults.solver_provider_allow_fallbacks
        ),
        solver_provider_min_throughput_p50=_arg_or_env_float(
            args.solver_provider_min_throughput_p50,
            "SOLVER_PROVIDER_MIN_THROUGHPUT_P50",
            "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P50",
        ),
        solver_provider_min_throughput_p90=_arg_or_env_float(
            args.solver_provider_min_throughput_p90,
            "SOLVER_PROVIDER_MIN_THROUGHPUT_P90",
            "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P90",
        ),
        random_seed=args.seed,
        solver_backend=solver_backend,
        solve_agent=args.agent,
        docker_solver_image=args.docker_solver_image,
        solver_agent_source=agent_source,
        docker_solver_memory=args.docker_solver_memory,
        docker_solver_cpus=args.docker_solver_cpus,
        docker_solver_pids_limit=args.docker_solver_pids_limit,
        docker_solver_tmp_size=args.docker_solver_tmp_size,
        docker_solver_workdir_size=args.docker_solver_workdir_size,
        docker_solver_nofile_limit=args.docker_solver_nofile_limit,
        docker_solver_max_output_bytes=args.docker_solver_max_output_bytes,
        docker_solver_drop_caps=not args.docker_solver_keep_caps,
        docker_solver_no_new_privileges=not args.docker_solver_allow_privilege_escalation,
        docker_solver_read_only_rootfs=not args.docker_solver_writeable_rootfs,
        docker_solver_user=args.docker_solver_user,
        docker_solver_no_cache=args.docker_solver_no_cache,
        debug=args.debug,
    )


def _build_eval_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        eval_model=_arg_or_env(args.eval_model, "EVAL_MODEL", "OPENROUTER_EVAL_MODEL"),
        agent_timeout=args.agent_timeout,
        random_seed=args.seed,
        debug=args.debug,
    )


def _build_compare_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        agent_timeout=args.agent_timeout,
        debug=args.debug,
    )


def _build_delete_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        agent_timeout=args.agent_timeout,
        debug=args.debug,
    )


def _run_private_submit(args: argparse.Namespace) -> None:
    from private_submission import (
        SubmissionCheck,
        build_public_submissions_api_payload,
        derive_submission_id,
        private_submission_check_passed,
        private_submission_signature_payload,
        private_submission_registration_check,
        record_private_submission_acceptance,
        run_private_submission_checks,
        write_private_submission_bundle,
    )
    from validate import _verify_hotkey_signature

    agent_py = args.agent.expanduser().read_text(encoding="utf-8")
    base_agent_py = args.base_agent.expanduser().read_text(encoding="utf-8")
    judge = None if args.skip_openrouter_judge else _build_private_submission_openrouter_judge(args)
    result = run_private_submission_checks(
        hotkey=args.hotkey,
        submitted_agent_py=agent_py,
        base_agent_py=base_agent_py,
        openrouter_judge=judge,
        min_score=args.judge_min_score,
    )
    submission_id = args.submission_id or derive_submission_id(
        hotkey=args.hotkey,
        agent_sha256=result.agent_sha256,
    )
    root = (
        args.private_submission_root.expanduser()
        if args.private_submission_root is not None
        else RunConfig(
            workspace_root=args.workspace_root.resolve(),
            validate_netuid=args.netuid,
        ).validate_root
        / "private-submissions"
    )
    registration_block, uid, registration_error = _private_submit_registration_context(args)
    if registration_error is None:
        registration_check = private_submission_registration_check(
            root=root,
            hotkey=args.hotkey,
            submission_id=submission_id,
            agent_sha256=result.agent_sha256,
            registration_block=registration_block,
        )
    else:
        registration_check = SubmissionCheck(
            name="Registration Gate",
            status="failed",
            summary="Could not verify the hotkey's current registration.",
            findings=[registration_error],
            metadata={"registration_block": registration_block, "uid": uid},
        )
    result.checks["registration_gate"] = registration_check
    result.accepted = result.accepted and registration_check.status == "passed"
    signature_payload = private_submission_signature_payload(
        hotkey=args.hotkey,
        submission_id=submission_id,
        agent_sha256=result.agent_sha256,
    )
    signature_valid = _verify_hotkey_signature(args.hotkey, signature_payload, args.signature)
    bundle_path = None
    if signature_valid and result.accepted:
        existing_bundle = root / submission_id
        if (
            existing_bundle.exists()
            and not args.overwrite
            and private_submission_check_passed(
                root,
                submission_id,
                result.agent_sha256,
                hotkey=args.hotkey,
                signature_verifier=_verify_hotkey_signature,
            )
        ):
            bundle_path = existing_bundle
        else:
            bundle_path = write_private_submission_bundle(
                root=root,
                submission_id=submission_id,
                hotkey=args.hotkey,
                agent_py=agent_py,
                check_result=result,
                signature=args.signature,
                registration_block=registration_block,
                overwrite=args.overwrite,
            )
        if registration_block is not None:
            record_private_submission_acceptance(
                root=root,
                hotkey=args.hotkey,
                submission_id=submission_id,
                agent_sha256=result.agent_sha256,
                registration_block=registration_block,
            )
        try:
            from r2 import publish_submissions_api_data

            publish_submissions_api_data(build_public_submissions_api_payload(root=root))
        except Exception as exc:
            log.warning("Accepted private submission but failed to publish submissions API: %s", exc)

    ci_checks = {name: check.to_dict() for name, check in result.checks.items()}
    llm_judge = ci_checks.get("openrouter_judge")

    payload = {
        "accepted": bool(result.accepted and signature_valid),
        "signature_valid": signature_valid,
        "submission_id": submission_id,
        "agent_sha256": result.agent_sha256,
        "commitment": f"private-submission:{submission_id}:{result.agent_sha256}",
        "signature_payload": signature_payload.decode("utf-8"),
        "bundle_path": str(bundle_path) if bundle_path is not None else None,
        "registration": {
            "uid": uid,
            "registration_block": registration_block,
        },
        "ci_checks": ci_checks,
        "llm_judge": llm_judge,
        "checks": ci_checks,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["accepted"]:
        raise SystemExit(1)


def _private_submit_registration_context(args: argparse.Namespace) -> tuple[int | None, int | None, str | None]:
    if args.registration_block is not None:
        return int(args.registration_block), None, None

    from validate import _current_registration_block, _open_subtensor

    config = RunConfig(
        workspace_root=args.workspace_root.resolve(),
        validate_netuid=args.netuid,
        validate_network=args.network,
        validate_subtensor_endpoint=args.subtensor_endpoint,
    )
    try:
        with _open_subtensor(config) as subtensor:
            uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(args.hotkey, args.netuid)
            if uid is None:
                return None, None, f"Hotkey {args.hotkey} is not registered on netuid {args.netuid}."
            registration_block = _current_registration_block(
                subtensor=subtensor,
                config=config,
                hotkey=args.hotkey,
                uid=int(uid),
            )
    except Exception as exc:
        return None, None, f"Registration lookup failed: {exc}"

    if registration_block is None:
        return None, int(uid), f"Could not resolve registration block for hotkey {args.hotkey} on netuid {args.netuid}."
    return int(registration_block), int(uid), None


def _run_serve_submissions_api(args: argparse.Namespace) -> None:
    from submission_api import SubmissionApiConfig, serve_submissions_api

    root = (
        args.private_submission_root.expanduser()
        if args.private_submission_root is not None
        else RunConfig(
            workspace_root=args.workspace_root.resolve(),
            validate_netuid=args.netuid,
        ).validate_root
        / "private-submissions"
    )
    run_config = RunConfig(
        workspace_root=args.workspace_root.resolve(),
        validate_netuid=args.netuid,
        validate_network=args.network,
        validate_subtensor_endpoint=args.subtensor_endpoint,
    )
    config = SubmissionApiConfig(
        private_submission_root=root,
        base_agent=args.base_agent.expanduser(),
        run_config=run_config,
        judge=None if args.skip_openrouter_judge else _build_private_submission_openrouter_judge(args),
        judge_min_score=args.judge_min_score,
        overwrite=args.overwrite,
        max_request_bytes=args.max_request_bytes,
        max_agent_bytes=args.max_agent_bytes,
        rate_limit_window_seconds=args.rate_limit_window_seconds,
        rate_limit_max_requests=args.rate_limit_max_requests,
        rate_limit_max_failures=args.rate_limit_max_failures,
    )
    serve_submissions_api(host=args.host, port=args.port, config=config)


def _build_private_submission_openrouter_judge(args: argparse.Namespace):
    from openrouter_client import complete_text

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for private-submit unless --skip-openrouter-judge is used")

    def judge(payload: dict) -> dict:
        prompt_payload = dict(payload)
        prompt_payload["patch"] = str(prompt_payload.get("patch") or "")[:120_000]
        prompt_payload["base_agent_py"] = str(prompt_payload.get("base_agent_py") or "")[:80_000]
        prompt_payload["submitted_agent_py"] = str(prompt_payload.get("submitted_agent_py") or "")[:120_000]
        requested_model = args.judge_model or os.environ.get("PRIVATE_SUBMISSION_JUDGE_MODEL")
        response = complete_text(
            system_prompt=(
                "You are the private submission gatekeeper for unarbos/ninja agent.py. "
                "Judge whether the submitted agent.py is a real, safe miner harness improvement. "
                "Return only JSON with verdict pass|warn|fail, overall_score 0-100, summary, reasons."
            ),
            prompt=json.dumps(prompt_payload, sort_keys=True),
            model=requested_model or _PRIVATE_SUBMISSION_JUDGE_MODEL,
            timeout=args.agent_timeout,
            openrouter_api_key=api_key,
            max_tokens=16_000,
            reasoning={"effort": "medium", "exclude": True},
        )
        return _parse_json_object(response)

    return judge


def _parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("OpenRouter judge did not return a JSON object")
    return payload


def _build_validate_config(args: argparse.Namespace) -> RunConfig:
    defaults = RunConfig()
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        solver_model=args.solver_model,
        baseline_model=_arg_or_env(args.baseline_model, "BASELINE_MODEL", "OPENROUTER_BASELINE_MODEL"),
        agent_timeout=args.agent_timeout,
        solver_max_requests=_arg_or_env_int(args.solver_max_requests, "SOLVER_MAX_REQUESTS"),
        solver_max_total_tokens=_arg_or_env_int(args.solver_max_total_tokens, "SOLVER_MAX_TOTAL_TOKENS"),
        solver_max_prompt_tokens=_arg_or_env_int(args.solver_max_prompt_tokens, "SOLVER_MAX_PROMPT_TOKENS"),
        solver_max_completion_tokens=_arg_or_env_int(args.solver_max_completion_tokens, "SOLVER_MAX_COMPLETION_TOKENS"),
        solver_max_cost=_arg_or_env_float(args.solver_max_cost, "SOLVER_MAX_COST"),
        solver_max_tokens_per_request=_arg_or_env_int(
            args.solver_max_tokens_per_request,
            "SOLVER_MAX_TOKENS_PER_REQUEST",
        ),
        solver_provider_sort=_arg_or_env(args.solver_provider_sort, "SOLVER_PROVIDER_SORT", "OPENROUTER_PROVIDER_SORT"),
        solver_provider_only=_arg_or_env(args.solver_provider_only, "SOLVER_PROVIDER_ONLY", "OPENROUTER_PROVIDER_ONLY"),
        solver_provider_allow_fallbacks=(
            False if args.solver_provider_disable_fallbacks else defaults.solver_provider_allow_fallbacks
        ),
        solver_provider_min_throughput_p50=_arg_or_env_float(
            args.solver_provider_min_throughput_p50,
            "SOLVER_PROVIDER_MIN_THROUGHPUT_P50",
            "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P50",
        ),
        solver_provider_min_throughput_p90=_arg_or_env_float(
            args.solver_provider_min_throughput_p90,
            "SOLVER_PROVIDER_MIN_THROUGHPUT_P90",
            "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P90",
        ),
        random_seed=args.seed,
        docker_solver_image=args.docker_solver_image,
        docker_solver_memory=args.docker_solver_memory,
        docker_solver_cpus=args.docker_solver_cpus,
        docker_solver_pids_limit=args.docker_solver_pids_limit,
        docker_solver_tmp_size=args.docker_solver_tmp_size,
        docker_solver_workdir_size=args.docker_solver_workdir_size,
        docker_solver_nofile_limit=args.docker_solver_nofile_limit,
        docker_solver_max_output_bytes=args.docker_solver_max_output_bytes,
        docker_solver_drop_caps=not args.docker_solver_keep_caps,
        docker_solver_no_new_privileges=not args.docker_solver_allow_privilege_escalation,
        docker_solver_read_only_rootfs=not args.docker_solver_writeable_rootfs,
        docker_solver_user=args.docker_solver_user,
        docker_solver_no_cache=args.docker_solver_no_cache,
        validate_netuid=args.netuid,
        validate_network=args.network,
        validate_subtensor_endpoint=args.subtensor_endpoint,
        validate_duel_rounds=args.duel_rounds,
        validate_win_margin=args.win_margin,
        validate_max_concurrency=args.max_concurrency,
        validate_round_concurrency=args.round_concurrency,
        validate_candidate_timeout_streak_limit=args.candidate_timeout_streak_limit,
        validate_task_pool_target=args.task_pool_target,
        validate_task_pool_static=args.task_pool_static,
        validate_pool_filler_concurrency=args.pool_filler_concurrency,
        validate_task_pool_refresh_count=args.task_pool_refresh_count,
        validate_task_pool_refresh_interval_seconds=args.task_pool_refresh_interval_seconds,
        validate_task_pool_fill_from_saved=args.task_pool_fill_from_saved,
        validate_task_cleanup_min_age_seconds=args.task_cleanup_min_age_seconds,
        validate_weight_interval_blocks=args.weight_interval_blocks,
        validate_king_window_size=args.king_window_size,
        validate_poll_interval_seconds=args.poll_interval_seconds,
        validate_duel_timeout_seconds=args.duel_timeout,
        validate_max_duels=args.max_duels,
        validate_min_commitment_block=args.min_commitment_block,
        validate_hotkey_spent_since_block=(
            args.hotkey_spent_since_block
            if args.hotkey_spent_since_block is not None
            else defaults.validate_hotkey_spent_since_block
        ),
        validate_queue_size=args.queue_size,
        validate_wallet_name=args.wallet_name,
        validate_wallet_hotkey=args.wallet_hotkey,
        validate_wallet_path=args.wallet_path,
        validate_publish_repo=args.publish_repo or defaults.validate_publish_repo,
        validate_publish_base=args.publish_base or defaults.validate_publish_base,
        validate_private_submission_watch=(
            defaults.validate_private_submission_watch
            if args.watch_private_submissions is None
            else bool(args.watch_private_submissions)
        ),
        validate_private_submission_only=(
            defaults.validate_private_submission_only
            if args.private_submission_only is None
            else bool(args.private_submission_only)
        ),
        validate_private_submission_root=(
            args.private_submission_root
            if args.private_submission_root is not None
            else defaults.validate_private_submission_root
        ),
        debug=args.debug,
    )


def _build_restore_r2_kings_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        validate_netuid=args.netuid,
        validate_king_window_size=args.count,
        debug=args.debug,
    )


def _arg_or_env(value: str | None, *env_names: str) -> str | None:
    if value:
        return value
    for name in env_names:
        env_value = os.environ.get(name)
        if env_value:
            return env_value
    return None


def _arg_or_env_int(value: int | None, *env_names: str) -> int | None:
    if value is not None:
        return value
    env_value = _arg_or_env(None, *env_names)
    return int(env_value) if env_value is not None else None


def _arg_or_env_float(value: float | None, *env_names: str) -> float | None:
    if value is not None:
        return value
    env_value = _arg_or_env(None, *env_names)
    return float(env_value) if env_value is not None else None


def _normalize_solution_names(raw_values: list[str]) -> list[str]:
    names: list[str] = []
    for raw_value in raw_values:
        parts = [part.strip() for part in raw_value.split(",")]
        names.extend(part for part in parts if part)
    if len(names) < 2:
        raise ValueError("eval requires at least two solution names")
    return names


def _normalize_compare_solution_names(raw_values: list[str]) -> list[str]:
    names = _normalize_solution_names(raw_values)
    if len(names) != 2:
        raise ValueError("compare requires exactly two solution names")
    return names


def _resolve_solve_target(raw_value: str, *, cwd: Path) -> tuple[str, SolverAgentSource | None]:
    normalized = raw_value.strip().lower()
    if normalized == "cursor":
        return "cursor", None
    if normalized == "claude":
        return "claude", None
    if normalized == "claw":
        return "claw", None
    return "docker-file", _resolve_agent_source(raw_value, cwd=cwd)


def _resolve_agent_source(raw_value: str, *, cwd: Path) -> SolverAgentSource:
    value = raw_value.strip()
    if not value:
        raise ValueError("--agent cannot be empty")

    candidate_path = Path(value).expanduser()
    if candidate_path.exists():
        resolved = _resolve_local_agent_file(candidate_path.resolve())
        return SolverAgentSource(
            raw=value,
            kind="local_file",
            local_path=str(resolved),
            agent_file=resolved.name,
        )

    if candidate_path.is_absolute():
        raise ValueError(f"--agent local path does not exist: {candidate_path}")

    relative_candidate = (cwd / candidate_path).resolve()
    if relative_candidate.exists():
        resolved = _resolve_local_agent_file(relative_candidate)
        return SolverAgentSource(
            raw=value,
            kind="local_file",
            local_path=str(resolved),
            agent_file=resolved.name,
        )

    repo_url, agent_file, commit_sha = _normalize_github_agent_source(value)
    if repo_url is None:
        raise ValueError(
            "--agent must be an existing Python file, a directory containing agent.py, "
            "or a GitHub repo URL/shorthand like "
            "'github.com/org/repo', 'org/repo@commit', or "
            "'https://github.com/org/repo/commit/<sha>'"
        )

    return SolverAgentSource(
        raw=value,
        kind="github_repo",
        repo_url=repo_url,
        agent_file=agent_file,
        commit_sha=commit_sha,
    )


def _normalize_github_agent_source(raw_value: str) -> tuple[str | None, str, str | None]:
    cleaned = raw_value.strip().rstrip("/")
    pinned_match = _split_repo_commit_ref(cleaned)
    if pinned_match is not None:
        repo_path, commit_sha = pinned_match
        return f"https://github.com/{repo_path}.git", _DEFAULT_AGENT_FILE, commit_sha

    if "://" not in cleaned and cleaned.count("/") >= 1 and not cleaned.startswith("github.com/"):
        parts = [part for part in cleaned.split("/") if part]
        if len(parts) >= 2:
            repo_path = "/".join(parts[:2])
            return f"https://github.com/{repo_path}.git", _DEFAULT_AGENT_FILE, None
        return None, _DEFAULT_AGENT_FILE, None

    parsed = urlparse(cleaned if "://" in cleaned else f"https://{cleaned}")
    if parsed.netloc.lower() != "github.com":
        return None, _DEFAULT_AGENT_FILE, None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None, _DEFAULT_AGENT_FILE, None

    if len(parts) >= 4 and parts[2] == "commit":
        repo_path = "/".join(parts[:2])
        return f"https://github.com/{repo_path}.git", _DEFAULT_AGENT_FILE, parts[3]

    repo_parts = parts[:2]
    if len(parts) >= 5 and parts[2] == "blob":
        repo_path = "/".join(repo_parts)
        return f"https://github.com/{repo_path}.git", "/".join(parts[4:]), None

    repo_path = "/".join(repo_parts)
    return f"https://github.com/{repo_path}.git", _DEFAULT_AGENT_FILE, None


def _split_repo_commit_ref(raw_value: str) -> tuple[str, str] | None:
    if "@" not in raw_value or "://" in raw_value or raw_value.startswith("github.com/"):
        return None
    repo_path, commit_sha = raw_value.rsplit("@", 1)
    parts = [part for part in repo_path.split("/") if part]
    if len(parts) != 2 or not commit_sha:
        return None
    return "/".join(parts), commit_sha


def _resolve_local_agent_file(candidate: Path) -> Path:
    if candidate.is_file():
        if candidate.suffix != ".py":
            raise ValueError(f"--agent local file must be a Python file: {candidate}")
        return candidate
    if candidate.is_dir():
        agent_file = candidate / _DEFAULT_AGENT_FILE
        if agent_file.is_file():
            return agent_file
        raise ValueError(f"--agent local directory must contain {_DEFAULT_AGENT_FILE}: {candidate}")
    raise ValueError(f"--agent local path must be a Python file or directory: {candidate}")


def _add_solver_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--solver-model", help="Optional model override for solving.")
    parser.add_argument(
        "--baseline-model",
        help="Cursor model ID for the baseline comparison solver.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for deterministic solver-side choices.",
    )
    parser.add_argument(
        "--solver-max-requests",
        type=int,
        help="Maximum number of proxied OpenRouter requests allowed for a solve run.",
    )
    parser.add_argument(
        "--solver-max-total-tokens",
        type=int,
        help="Maximum total OpenRouter tokens allowed for a solve run.",
    )
    parser.add_argument(
        "--solver-max-prompt-tokens",
        type=int,
        help="Maximum prompt tokens allowed for a solve run.",
    )
    parser.add_argument(
        "--solver-max-completion-tokens",
        type=int,
        help="Maximum completion tokens allowed for a solve run.",
    )
    parser.add_argument(
        "--solver-max-cost",
        type=float,
        help="Maximum OpenRouter cost allowed for a solve run.",
    )
    parser.add_argument(
        "--solver-max-tokens-per-request",
        type=int,
        help="Maximum completion tokens to allow on any single proxied request.",
    )
    parser.add_argument(
        "--solver-provider-sort",
        choices=("price", "throughput", "latency"),
        help="OpenRouter provider sort policy for proxied solver requests.",
    )
    parser.add_argument(
        "--solver-provider-only",
        help="Comma-separated OpenRouter provider slugs to allow for proxied solver requests.",
    )
    parser.add_argument(
        "--solver-provider-disable-fallbacks",
        action="store_true",
        help="Disable OpenRouter fallbacks outside the ordered/allowed provider policy.",
    )
    parser.add_argument(
        "--solver-provider-min-throughput-p50",
        type=float,
        help="Prefer providers with at least this p50 throughput in tokens/sec.",
    )
    parser.add_argument(
        "--solver-provider-min-throughput-p90",
        type=float,
        help="Prefer providers with at least this p90 throughput in tokens/sec.",
    )
    parser.add_argument(
        "--docker-solver-image",
        help="Optional Docker image tag for the solver image. If omitted, one is derived.",
    )
    parser.add_argument(
        "--docker-solver-memory",
        default="2g",
        help="Docker memory limit for the solver container.",
    )
    parser.add_argument(
        "--docker-solver-cpus",
        default="2",
        help="Docker CPU limit for the solver container.",
    )
    parser.add_argument(
        "--docker-solver-pids-limit",
        type=int,
        default=256,
        help="Maximum number of processes allowed inside the solver container.",
    )
    parser.add_argument(
        "--docker-solver-tmp-size",
        default="128m",
        help="Maximum writable size of /tmp inside the solver container.",
    )
    parser.add_argument(
        "--docker-solver-workdir-size",
        default="2g",
        help="Maximum writable size of /work inside the solver container.",
    )
    parser.add_argument(
        "--docker-solver-nofile-limit",
        type=int,
        default=4096,
        help="Maximum number of open files allowed inside the solver container.",
    )
    parser.add_argument(
        "--docker-solver-max-output-bytes",
        type=int,
        default=1000000,
        help="Maximum combined stdout or stderr bytes allowed from the solver command before it is killed.",
    )
    parser.add_argument(
        "--docker-solver-user",
        help="Optional user to run the solver container as.",
    )
    parser.add_argument(
        "--docker-solver-keep-caps",
        action="store_true",
        help="Do not drop Linux capabilities in the solver container.",
    )
    parser.add_argument(
        "--docker-solver-allow-privilege-escalation",
        action="store_true",
        help="Do not set no-new-privileges on the solver container.",
    )
    parser.add_argument(
        "--docker-solver-writeable-rootfs",
        action="store_true",
        help="Do not force the solver container root filesystem to read-only mode.",
    )
    parser.add_argument(
        "--docker-solver-no-cache",
        action="store_true",
        help="Build the solver Docker image with --no-cache.",
    )


def _load_dotenv() -> None:
    dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


if __name__ == "__main__":
    main()
