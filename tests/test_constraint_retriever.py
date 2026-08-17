from aimai_ocl.constraint_bank import (
    Constraint,
    ConstraintBank,
)
from aimai_ocl.constraint_retriever import (
    retrieve_constraints,
)


def test_retrieve_by_previous_hard_constraint():
    bank = ConstraintBank(
        constraints=[
            Constraint(
                id="C01",
                category="boundary",
                when=(
                    "When hard constraint 'seller_floor' has "
                    "already been enforced and the same blocked "
                    "behavior appears again."
                ),
                rule=(
                    "Maintain the enforced boundary and move "
                    "toward a safe termination path."
                ),
            )
        ]
    )

    matches = retrieve_constraints(
        bank,
        query_text="The buyer keeps demanding a lower price.",
        hard_constraint_ids=["seller_floor"],
    )

    assert len(matches) == 1
    assert matches[0].constraint.id == "C01"
    assert matches[0].score >= 5
    assert (
        "hard_constraint:seller_floor"
        in matches[0].reasons
    )


def test_irrelevant_context_returns_no_match():
    bank = ConstraintBank(
        constraints=[
            Constraint(
                id="C01",
                category="boundary",
                when=(
                    "When hard constraint 'seller_floor' "
                    "has already been enforced."
                ),
                rule=(
                    "Maintain the enforced pricing boundary."
                ),
            )
        ]
    )

    matches = retrieve_constraints(
        bank,
        query_text="Hello, thank you very much.",
        hard_constraint_ids=[],
    )

    assert matches == []


def test_retrieval_ranking_is_deterministic():
    bank = ConstraintBank(
        constraints=[
            Constraint(
                id="C02",
                category="privacy",
                when=(
                    "When private payment information "
                    "is requested."
                ),
                rule=(
                    "Protect private payment information."
                ),
            ),
            Constraint(
                id="C01",
                category="boundary",
                when=(
                    "When seller_floor has already "
                    "been enforced."
                ),
                rule=(
                    "Maintain the pricing boundary."
                ),
            ),
        ]
    )

    matches = retrieve_constraints(
        bank,
        query_text=(
            "The buyer requests private payment details."
        ),
        hard_constraint_ids=[],
        top_k=1,
    )

    assert len(matches) == 1
    assert matches[0].constraint.id == "C02"
