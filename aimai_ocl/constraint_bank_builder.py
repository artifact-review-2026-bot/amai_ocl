from __future__ import annotations

from collections import Counter
from typing import Any

from aimai_ocl.constraint_bank import ConstraintBank
from aimai_ocl.constraint_extractor import (
    extract_candidate_constraints,
)
from aimai_ocl.constraint_updater import (
    update_constraint_bank,
)


def build_bank(
    records: list[dict[str, Any]],
    *,
    min_evidence_rounds: int = 2,
) -> tuple[ConstraintBank, dict[str, Any]]:
    """
    Build a ConstraintBank from structured benchmark records.

    Records without trajectories are skipped. Candidate constraints are
    extracted, validated, deduplicated, and added deterministically.
    """

    bank = ConstraintBank()

    candidate_count = 0
    trajectory_record_count = 0
    decisions = []

    for record in records:
        trajectory = record.get("trajectory") or []

        if not trajectory:
            continue

        trajectory_record_count += 1

        candidates = extract_candidate_constraints(
            trajectory,
            min_repeats=min_evidence_rounds,
        )

        candidate_count += len(candidates)

        decisions.extend(
            update_constraint_bank(
                bank,
                candidates,
                min_evidence_rounds=min_evidence_rounds,
            )
        )

    accepted_count = sum(
        decision.accepted
        for decision in decisions
    )

    rejected_count = (
        len(decisions) - accepted_count
    )

    rejection_reasons = Counter(
        decision.reason
        for decision in decisions
        if not decision.accepted
    )

    stats = {
        "record_count": len(records),
        "trajectory_record_count": trajectory_record_count,
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "rejection_reasons": dict(
            sorted(rejection_reasons.items())
        ),
        "bank_size": len(bank),
    }

    return bank, stats
