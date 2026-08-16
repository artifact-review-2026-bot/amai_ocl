"""Tests for the OCL-v2 experiment arm."""

from aimai_ocl.experiment import resolve_arm


def test_ocl_v2_arm_enables_ocl_and_constraint_bank() -> None:
    """OCL-v2 should keep OCL enabled and add the ConstraintBank."""
    arm = resolve_arm("ocl_v2")

    assert arm.name == "ocl_v2"
    assert arm.ocl is True
    assert arm.use_constraint_bank is True