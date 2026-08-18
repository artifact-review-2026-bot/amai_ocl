from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from aimai_ocl.constraint_bank import ConstraintBank
from aimai_ocl.constraint_extractor import (
    extract_candidate_constraints,
)
from aimai_ocl.constraint_updater import (
    update_constraint_bank,
)


def update_online_constraint_bank(
    bank_path: str | Path,
    trajectory: list[dict[str, Any]],
    *,
    min_evidence_rounds: int = 2,
) -> dict[str, Any]:
    """
    Update a writable ConstraintBank from one completed trajectory.

    The bank is loaded from disk, candidate constraints are extracted
    from the trajectory, validated and deduplicated, then the updated
    bank is saved back to the same path.
    """

    path = Path(bank_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Online ConstraintBank does not exist: {path}"
        )

    bank = ConstraintBank.load_json(path)
    bank_size_before = len(bank)

    if not trajectory:
        return {
            "candidate_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "rejection_reasons": {},
            "bank_size_before": bank_size_before,
            "bank_size_after": bank_size_before,
            "accepted_ids": [],
        }

    candidates = extract_candidate_constraints(
        trajectory,
        min_repeats=min_evidence_rounds,
    )

    decisions = update_constraint_bank(
        bank,
        candidates,
        min_evidence_rounds=min_evidence_rounds,
    )

    accepted = [
        decision
        for decision in decisions
        if decision.accepted
    ]

    rejected = [
        decision
        for decision in decisions
        if not decision.accepted
    ]

    rejection_reasons = Counter(
        decision.reason
        for decision in rejected
    )

    bank.save_json(
        path,
        version="online-v1",
    )

    return {
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "rejection_reasons": dict(
            sorted(rejection_reasons.items())
        ),
        "bank_size_before": bank_size_before,
        "bank_size_after": len(bank),
        "accepted_ids": [
            decision.constraint_id
            for decision in accepted
        ],
    }
