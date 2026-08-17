from aimai_ocl.constraint_extractor import (
    extract_candidate_constraints,
)


def test_repeated_hard_constraint_creates_candidate():
    trajectory = [
        {
            "round_id": 8,
            "failed_hard_constraints": [],
            "next_state": {
                "status": None,
            },
        },
        {
            "round_id": 9,
            "failed_hard_constraints": ["seller_floor"],
            "next_state": {
                "status": None,
            },
        },
        {
            "round_id": 10,
            "failed_hard_constraints": ["seller_floor"],
            "next_state": {
                "status": "timeout",
            },
        },
    ]

    results = extract_candidate_constraints(trajectory)

    assert len(results) == 1

    result = results[0]

    assert result.feedback.evaluation == "safe_but_inefficient"
    assert result.feedback.source_hard_constraint == "seller_floor"
    assert result.feedback.evidence_rounds == (9, 10)

    assert result.candidate.category == "boundary"
    assert "seller_floor" in result.candidate.when
    assert "Maintain the enforced boundary" in result.candidate.rule


def test_single_hard_constraint_trigger_does_not_create_candidate():
    trajectory = [
        {
            "round_id": 3,
            "failed_hard_constraints": ["seller_floor"],
            "next_state": {
                "status": "agreed",
            },
        },
    ]

    results = extract_candidate_constraints(trajectory)

    assert results == []


def test_candidate_id_is_deterministic():
    trajectory = [
        {
            "round_id": 1,
            "failed_hard_constraints": ["seller_floor"],
            "next_state": {
                "status": None,
            },
        },
        {
            "round_id": 2,
            "failed_hard_constraints": ["seller_floor"],
            "next_state": {
                "status": "timeout",
            },
        },
    ]

    first = extract_candidate_constraints(trajectory)
    second = extract_candidate_constraints(trajectory)

    assert first[0].candidate.id == second[0].candidate.id
