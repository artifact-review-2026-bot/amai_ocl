from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from aimai_ocl.constraint_bank import Constraint


@dataclass(frozen=True, slots=True)
class FeedbackSignal:
    evaluation: str
    directive: str
    source_hard_constraint: str
    evidence_rounds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    feedback: FeedbackSignal
    candidate: Constraint


def _stable_constraint_id(
    *,
    category: str,
    rule: str,
    when: str,
) -> str:
    payload = f"{category}\n{when}\n{rule}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"constraint_{digest}"


def _infer_final_status(
    trajectory: list[dict[str, Any]],
) -> str | None:
    if not trajectory:
        return None

    next_state = trajectory[-1].get("next_state") or {}
    status = next_state.get("status")

    if status is None:
        return None

    return str(status)


def extract_candidate_constraints(
    trajectory: list[dict[str, Any]],
    *,
    final_status: str | None = None,
    min_repeats: int = 2,
) -> list[ExtractionResult]:
    """
    Extract contextual candidate constraints from structured trajectories.

    This first version focuses on repeated hard-constraint triggers.

    The learned constraint must not duplicate the hard constraint itself.
    Instead, it captures how the agent should behave after the same
    execution boundary has already been enforced repeatedly.
    """

    if min_repeats < 2:
        raise ValueError("min_repeats must be at least 2.")

    rounds_by_constraint: dict[str, list[int]] = {}

    for step in trajectory:
        round_id = int(step.get("round_id", -1))

        for constraint_id in step.get("failed_hard_constraints") or []:
            constraint_id = str(constraint_id)

            rounds_by_constraint.setdefault(
                constraint_id,
                [],
            ).append(round_id)

    status = final_status or _infer_final_status(trajectory)

    results: list[ExtractionResult] = []

    for constraint_id, evidence_rounds in sorted(
        rounds_by_constraint.items()
    ):
        if len(evidence_rounds) < min_repeats:
            continue

        if status == "timeout":
            evaluation = "safe_but_inefficient"
        else:
            evaluation = "repeated_boundary_pressure"

        directive = (
            "After the same hard boundary has already been enforced, "
            "do not keep reconsidering the blocked action. "
            "Maintain the boundary and move toward a concise refusal, "
            "safe alternative, escalation, or termination."
        )

        when = (
            f"When hard constraint '{constraint_id}' has already been "
            "enforced and the same blocked behavior is proposed again "
            "in later rounds."
        )

        rule = (
            "Maintain the enforced boundary instead of repeatedly "
            "re-negotiating the blocked action; move to a concise "
            "refusal, safe alternative, escalation, or termination path."
        )

        category = "boundary"

        candidate = Constraint(
            id=_stable_constraint_id(
                category=category,
                rule=rule,
                when=when,
            ),
            category=category,
            rule=rule,
            when=when,
        )

        feedback = FeedbackSignal(
            evaluation=evaluation,
            directive=directive,
            source_hard_constraint=constraint_id,
            evidence_rounds=tuple(evidence_rounds),
        )

        results.append(
            ExtractionResult(
                feedback=feedback,
                candidate=candidate,
            )
        )

    return results
