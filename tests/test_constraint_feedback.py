from aimai_ocl.constraint_bank import Constraint, ConstraintBank
from aimai_ocl.constraint_feedback import (
    ConstraintFeedbackStore,
    update_feedback_from_trajectory,
)
from aimai_ocl.constraint_retriever import retrieve_constraints


def _make_budget_bank():
    bank = ConstraintBank()

    bank.add(
        Constraint(
            id="good_constraint",
            category="boundary",
            when=(
                "When hard constraint 'budget_cap' "
                "has already been enforced."
            ),
            rule="Maintain the budget boundary.",
        )
    )

    bank.add(
        Constraint(
            id="bad_constraint",
            category="boundary",
            when=(
                "When hard constraint 'budget_cap' "
                "has already been enforced."
            ),
            rule="Maintain the budget boundary.",
        )
    )

    return bank


def test_feedback_persistence(tmp_path):
    path = tmp_path / "feedback.json"

    store = ConstraintFeedbackStore()

    item = store.get_or_create("constraint_a")
    item.evaluation_count = 3
    item.retrieval_event_count = 4
    item.success_count = 2
    item.failure_count = 1

    store.save_json(path)

    loaded = ConstraintFeedbackStore.load_json(path)
    result = loaded.feedback["constraint_a"]

    assert result.evaluation_count == 3
    assert result.retrieval_event_count == 4
    assert result.success_count == 2
    assert result.failure_count == 1
    assert result.effectiveness == 2 / 3


def test_feedback_changes_retrieval_ranking():
    bank = _make_budget_bank()
    store = ConstraintFeedbackStore()

    good = store.get_or_create("good_constraint")
    good.evaluation_count = 3
    good.success_count = 3

    bad = store.get_or_create("bad_constraint")
    bad.evaluation_count = 3
    bad.failure_count = 3

    matches = retrieve_constraints(
        bank,
        hard_constraint_ids=["budget_cap"],
        feedback_store=store,
    )

    scores = {
        match.constraint.id: match.score
        for match in matches
    }

    assert scores["good_constraint"] == 7
    assert scores["bad_constraint"] == 3


def test_insufficient_feedback_does_not_change_score():
    bank = _make_budget_bank()
    store = ConstraintFeedbackStore()

    good = store.get_or_create("good_constraint")
    good.evaluation_count = 2
    good.success_count = 2

    matches = retrieve_constraints(
        bank,
        hard_constraint_ids=["budget_cap"],
        feedback_store=store,
    )

    scores = {
        match.constraint.id: match.score
        for match in matches
    }

    assert scores["good_constraint"] == 5


def test_feedback_from_trajectory_is_episode_level():
    store = ConstraintFeedbackStore()

    trajectory = [
        {
            "round_id": 0,
            "failed_hard_constraints": [],
            "retrieved_constraints": [],
        },
        {
            "round_id": 1,
            "failed_hard_constraints": [],
            "retrieved_constraints": [
                {
                    "id": "constraint_a",
                    "reasons": [
                        "hard_constraint:budget_cap"
                    ],
                }
            ],
        },
        {
            "round_id": 2,
            "failed_hard_constraints": [],
            "retrieved_constraints": [
                {
                    "id": "constraint_a",
                    "reasons": [
                        "hard_constraint:budget_cap"
                    ],
                }
            ],
        },
    ]

    result = update_feedback_from_trajectory(
        store,
        trajectory,
    )

    feedback = store.feedback["constraint_a"]

    assert result["evaluated_constraints"] == 1
    assert feedback.evaluation_count == 1
    assert feedback.retrieval_event_count == 2
    assert feedback.success_count == 1
    assert feedback.failure_count == 0


def test_matching_failure_produces_negative_feedback():
    store = ConstraintFeedbackStore()

    trajectory = [
        {
            "round_id": 1,
            "failed_hard_constraints": [],
            "retrieved_constraints": [
                {
                    "id": "constraint_a",
                    "reasons": [
                        "hard_constraint:budget_cap"
                    ],
                }
            ],
        },
        {
            "round_id": 2,
            "failed_hard_constraints": [
                "budget_cap"
            ],
            "retrieved_constraints": [],
        },
    ]

    update_feedback_from_trajectory(
        store,
        trajectory,
    )

    feedback = store.feedback["constraint_a"]

    assert feedback.evaluation_count == 1
    assert feedback.success_count == 0
    assert feedback.failure_count == 1
    assert feedback.effectiveness == 0.0
