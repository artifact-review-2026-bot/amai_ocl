from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from aimai_ocl.constraint_bank import Constraint, ConstraintBank
from aimai_ocl.constraint_extractor import ExtractionResult


@dataclass(frozen=True, slots=True)
class ConstraintUpdateDecision:
    constraint_id: str
    accepted: bool
    reason: str


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _constraint_signature(
    constraint: Constraint,
) -> tuple[str, str, str]:
    """
    Canonical content signature used for deterministic exact deduplication.
    """
    return (
        _normalize_text(constraint.category),
        _normalize_text(constraint.when),
        _normalize_text(constraint.rule),
    )


def validate_candidate(
    result: ExtractionResult,
    *,
    min_evidence_rounds: int = 2,
) -> tuple[bool, str]:
    """
    Apply deterministic validation before a candidate may enter
    ConstraintBank.

    This is intentionally conservative. It checks structural validity
    and evidence support, but does not yet attempt semantic conflict
    resolution.
    """

    if min_evidence_rounds < 1:
        raise ValueError("min_evidence_rounds must be at least 1.")

    candidate = result.candidate
    feedback = result.feedback

    if not candidate.id.strip():
        return False, "empty_id"

    if not candidate.category.strip():
        return False, "empty_category"

    if not candidate.when.strip():
        return False, "empty_when"

    if not candidate.rule.strip():
        return False, "empty_rule"

    if not feedback.evaluation.strip():
        return False, "empty_evaluation"

    if not feedback.directive.strip():
        return False, "empty_directive"

    if not feedback.source_hard_constraint.strip():
        return False, "missing_source_hard_constraint"

    evidence_rounds = tuple(dict.fromkeys(feedback.evidence_rounds))

    required_evidence = (
        1
        if feedback.evaluation == "successful_recovery"
        else min_evidence_rounds
    )

    if len(evidence_rounds) < required_evidence:
        return False, "insufficient_evidence"

    return True, "validated"


def update_constraint_bank(
    bank: ConstraintBank,
    results: Iterable[ExtractionResult],
    *,
    min_evidence_rounds: int = 2,
) -> list[ConstraintUpdateDecision]:
    """
    Validate candidate constraints, remove deterministic duplicates,
    and add accepted candidates to ConstraintBank.
    """

    existing_ids = {
        constraint.id
        for constraint in bank.constraints
    }

    existing_signatures = {
        _constraint_signature(constraint)
        for constraint in bank.constraints
    }

    decisions: list[ConstraintUpdateDecision] = []

    for result in results:
        candidate = result.candidate

        valid, reason = validate_candidate(
            result,
            min_evidence_rounds=min_evidence_rounds,
        )

        if not valid:
            decisions.append(
                ConstraintUpdateDecision(
                    constraint_id=candidate.id,
                    accepted=False,
                    reason=reason,
                )
            )
            continue

        if candidate.id in existing_ids:
            decisions.append(
                ConstraintUpdateDecision(
                    constraint_id=candidate.id,
                    accepted=False,
                    reason="duplicate_id",
                )
            )
            continue

        signature = _constraint_signature(candidate)

        if signature in existing_signatures:
            decisions.append(
                ConstraintUpdateDecision(
                    constraint_id=candidate.id,
                    accepted=False,
                    reason="duplicate_content",
                )
            )
            continue

        bank.add(candidate)

        existing_ids.add(candidate.id)
        existing_signatures.add(signature)

        decisions.append(
            ConstraintUpdateDecision(
                constraint_id=candidate.id,
                accepted=True,
                reason="added",
            )
        )

    return decisions
