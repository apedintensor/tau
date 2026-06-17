from __future__ import annotations

import hashlib
import os

# Greedy decoding uses temperature=0; low top_p is belt-and-suspenders for sampling APIs.
VALIDATOR_TOP_P = float(os.environ.get("TAU_TOP_P", "0.01"))


def deterministic_sampling_seed(*, configured: int | None, material: str) -> int:
    """Return a non-negative OpenRouter seed.

    When ``configured`` is set (via TAU_*_SEED env), that value is used for all
    calls. Otherwise derive a stable seed from ``material`` so identical inputs
    reuse the same sampling seed without a global constant.
    """
    if configured is not None:
        return int(configured) & 0x7FFFFFFF
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def judge_seed_material(
    *,
    task_name: str,
    model: str,
    king_patch: str,
    challenger_patch: str,
) -> str:
    patch_digest = hashlib.sha256(f"{king_patch}\0{challenger_patch}".encode("utf-8")).hexdigest()
    return f"judge:{task_name}:{model}:{patch_digest}"


def solver_seed_material(*, task_name: str, solution_name: str, agent_hash: str) -> str:
    return f"solver:{task_name}:{solution_name}:{agent_hash}"
