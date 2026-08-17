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

    Two deterministic feedback patterns are currently supported:

    1. Repeated boundary failure:
       the same hard constraint is triggered repeatedly.

    2. Successful recovery:
       a hard constraint is triggered once, is not triggered again,
       and the episode ultimately succeeds.

    Learned constraints describe contextual handling strategies rather
    than duplicating the underlying hard constraint itself.
    """

    if min_repeats < 2:
        raise ValueError("min_repeats must be at least 2.")

    rounds_by_constraint: dict[str, list[int]] = {}

    for step in trajectory:
        round_id = int(step.get("round_id", -1))

        for constraint_id in step.get(
            "failed_hard_constraints"
        ) or []:
            constraint_id = str(constraint_id)

            rounds_by_constraint.setdefault(
                constraint_id,
                [],
            ).append(round_id)

    status = final_status or _infer_final_status(
        trajectory
    )

    results: list[ExtractionResult] = []

    # Pattern 1: repeated boundary failure.
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

    # Pattern 2: successful recovery after one intervention.
    if status == "agreed":
        for constraint_id, evidence_rounds in sorted(
            rounds_by_constraint.items()
        ):
            if len(evidence_rounds) != 1:
                continue

            trigger_round = evidence_rounds[0]

            directive = (
                "Continue from the corrected safe action after the "
                "hard intervention succeeds. Do not revert to the "
                "previously blocked behavior."
            )

            when = (
                f"When hard constraint '{constraint_id}' has been "
                "triggered once, the behavior has been corrected, "
                "and negotiation continues."
            )

            rule = (
                "Preserve the corrected safe behavior after the "
                "intervention and continue toward agreement without "
                "returning to the previously blocked action."
            )

            category = "recovery"

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
                evaluation="successful_recovery",
                directive=directive,
                source_hard_constraint=constraint_id,
                evidence_rounds=(trigger_round,),
            )

            results.append(
                ExtractionResult(
                    feedback=feedback,
                    candidate=candidate,
                )
            )

    return results
