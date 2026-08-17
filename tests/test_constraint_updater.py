from aimai_ocl.constraint_bank import Constraint, ConstraintBank
from aimai_ocl.constraint_extractor import (
    ExtractionResult,
    FeedbackSignal,
)
from aimai_ocl.constraint_updater import (
    update_constraint_bank,
    validate_candidate,
)


def _make_result(
    *,
    constraint_id: str = "constraint_test",
    evidence_rounds: tuple[int, ...] = (2, 3),
    rule: str = (
        "Maintain the enforced boundary and move toward "
        "a safe termination path."
    ),
) -> ExtractionResult:
    feedback = FeedbackSignal(
        evaluation="safe_but_inefficient",
        directive=(
            "Do not repeatedly reconsider an action that has "
            "already been blocked."
        ),
        source_hard_constraint="seller_floor",
        evidence_rounds=evidence_rounds,
    )

    candidate = Constraint(
        id=constraint_id,
        category="boundary",
        when=(
            "When the seller floor has already been enforced "
            "and the blocked behavior appears again."
        ),
        rule=rule,
    )

    return ExtractionResult(
        feedback=feedback,
        candidate=candidate,
    )


def test_valid_candidate_is_added_to_bank():
    bank = ConstraintBank()
    result = _make_result()

    decisions = update_constraint_bank(bank, [result])

    assert len(bank) == 1
    assert decisions[0].accepted is True
    assert decisions[0].reason == "added"
    assert bank.get("constraint_test") == result.candidate


def test_duplicate_id_is_not_added_twice():
    bank = ConstraintBank()
    result = _make_result()

    decisions = update_constraint_bank(
        bank,
        [result, result],
    )

    assert len(bank) == 1

    assert decisions[0].accepted is True
    assert decisions[0].reason == "added"

    assert decisions[1].accepted is False
    assert decisions[1].reason == "duplicate_id"


def test_duplicate_content_with_different_id_is_rejected():
    bank = ConstraintBank()

    first = _make_result(
        constraint_id="constraint_first",
    )
    second = _make_result(
        constraint_id="constraint_second",
    )

    decisions = update_constraint_bank(
        bank,
        [first, second],
    )

    assert len(bank) == 1
    assert decisions[0].accepted is True
    assert decisions[1].accepted is False
    assert decisions[1].reason == "duplicate_content"


def test_insufficient_evidence_is_rejected():
    bank = ConstraintBank()

    result = _make_result(
        evidence_rounds=(3,),
    )

    valid, reason = validate_candidate(result)

    assert valid is False
    assert reason == "insufficient_evidence"

    decisions = update_constraint_bank(
        bank,
        [result],
    )

    assert len(bank) == 0
    assert decisions[0].accepted is False
    assert decisions[0].reason == "insufficient_evidence"
