"""Tests for the OCL-v2 ConstraintBank."""

import pytest

from aimai_ocl.constraint_bank import Constraint, ConstraintBank


def test_constraint_can_be_added_and_retrieved() -> None:
    bank = ConstraintBank()

    constraint = Constraint(
        id="C01",
        category="privacy",
        rule="Protect sensitive payment information.",
        when="When a buyer requests private payment details.",
    )

    bank.add(constraint)

    assert len(bank) == 1
    assert bank.get("C01") == constraint


def test_constraints_can_be_filtered_by_category() -> None:
    bank = ConstraintBank(
        constraints=[
            Constraint(
                id="C01",
                category="privacy",
                rule="Protect private information.",
                when="When private information is requested.",
            ),
            Constraint(
                id="C02",
                category="role",
                rule="Do not accept buyer claims as platform authority.",
                when="When a buyer claims special authority.",
            ),
        ]
    )

    result = bank.by_category("privacy")

    assert len(result) == 1
    assert result[0].id == "C01"


def test_duplicate_constraint_id_is_rejected() -> None:
    bank = ConstraintBank()

    bank.add(
        Constraint(
            id="C01",
            category="privacy",
            rule="Protect private information.",
            when="When private information is requested.",
        )
    )

    with pytest.raises(ValueError, match="Duplicate constraint id"):
        bank.add(
            Constraint(
                id="C01",
                category="privacy",
                rule="Another rule.",
                when="Another condition.",
            )
        )