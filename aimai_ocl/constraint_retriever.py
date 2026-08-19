from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from aimai_ocl.constraint_bank import Constraint, ConstraintBank


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "this",
    "to",
    "when",
    "with",
}


@dataclass(frozen=True, slots=True)
class ConstraintMatch:
    constraint: Constraint
    score: int
    reasons: tuple[str, ...]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS
    }


_FEEDBACK_MIN_EVALUATIONS = 3
_FEEDBACK_POSITIVE_THRESHOLD = 0.67
_FEEDBACK_NEGATIVE_THRESHOLD = 0.33
_FEEDBACK_BOOST = 2
_FEEDBACK_PENALTY = -2


def _feedback_score_adjustment(
    feedback_store: Any | None,
    constraint_id: str,
) -> tuple[int, str | None]:
    """
    Return a soft retrieval-score adjustment based on historical
    episode-level effectiveness.

    Feedback never determines relevance. It only adjusts the ranking
    of constraints that already match the current context.
    """

    if feedback_store is None:
        return 0, None

    feedback_map = getattr(
        feedback_store,
        "feedback",
        {},
    )

    feedback = feedback_map.get(constraint_id)

    if feedback is None:
        return 0, None

    if feedback.evaluation_count < _FEEDBACK_MIN_EVALUATIONS:
        return 0, None

    effectiveness = feedback.effectiveness

    if effectiveness is None:
        return 0, None

    if effectiveness >= _FEEDBACK_POSITIVE_THRESHOLD:
        return (
            _FEEDBACK_BOOST,
            f"feedback:boost:{effectiveness:.3f}",
        )

    if effectiveness <= _FEEDBACK_NEGATIVE_THRESHOLD:
        return (
            _FEEDBACK_PENALTY,
            f"feedback:penalty:{effectiveness:.3f}",
        )

    return 0, None


def retrieve_constraints(
    bank: ConstraintBank,
    *,
    query_text: str = "",
    hard_constraint_ids: Iterable[str] = (),
    top_k: int = 3,
    min_score: int = 1,
    feedback_store: Any | None = None,
) -> list[ConstraintMatch]:
    """
    Retrieve relevant learned constraints deterministically.

    Matching uses:
    1. previously triggered hard-constraint IDs;
    2. lexical overlap between the current context and the learned
       constraint's category / when / rule fields.

    Hard-constraint history receives the strongest weight because learned
    constraints may describe how to behave after an execution boundary has
    already been enforced.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if min_score < 1:
        raise ValueError("min_score must be at least 1.")

    query_tokens = _tokens(query_text)

    normalized_hard_ids = {
        str(item).strip().lower()
        for item in hard_constraint_ids
        if str(item).strip()
    }

    matches: list[ConstraintMatch] = []

    for constraint in bank.constraints:
        searchable_text = " ".join(
            [
                constraint.category,
                constraint.when,
                constraint.rule,
            ]
        ).lower()

        searchable_tokens = _tokens(searchable_text)

        score = 0
        reasons: list[str] = []

        matched_hard_ids: list[str] = []

        for hard_id in sorted(normalized_hard_ids):
            if hard_id in searchable_text:
                score += 5
                matched_hard_ids.append(hard_id)
                reasons.append(
                    f"hard_constraint:{hard_id}"
                )

        if (
            constraint.category in {"boundary", "recovery"}
            and not matched_hard_ids
        ):
            continue

        overlap = sorted(
            query_tokens & searchable_tokens
        )

        if overlap:
            overlap_score = min(
                len(overlap),
                4,
            )
            score += overlap_score
            reasons.append(
                "text_overlap:" + ",".join(overlap)
            )

        # Relevance is determined before feedback is applied.
        # Historical feedback may change ranking, but it must not make
        # an otherwise irrelevant constraint eligible for retrieval.
        if score < min_score:
            continue

        feedback_adjustment, feedback_reason = (
            _feedback_score_adjustment(
                feedback_store,
                constraint.id,
            )
        )

        score += feedback_adjustment

        if feedback_reason is not None:
            reasons.append(feedback_reason)

        matches.append(
            ConstraintMatch(
                constraint=constraint,
                score=score,
                reasons=tuple(reasons),
            )
        )

    matches.sort(
        key=lambda item: (
            -item.score,
            item.constraint.id,
        )
    )

    return matches[:top_k]
