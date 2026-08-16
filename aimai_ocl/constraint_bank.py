"""ConstraintBank for OCL-v2.

The ConstraintBank stores reusable governance constraints learned from
interaction experience. Existing hard constraints remain in control.py
and always take precedence.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Constraint:
    """A reusable governance constraint."""

    id: str
    category: str
    rule: str
    when: str


@dataclass(slots=True)
class ConstraintBank:
    """A collection of reusable constraints for OCL-v2."""

    constraints: list[Constraint] = field(default_factory=list)

    def add(self, constraint: Constraint) -> None:
        """Add a constraint to the bank."""
        if any(item.id == constraint.id for item in self.constraints):
            raise ValueError(f"Duplicate constraint id: {constraint.id}")

        self.constraints.append(constraint)

    def get(self, constraint_id: str) -> Constraint | None:
        """Get one constraint by id."""
        for constraint in self.constraints:
            if constraint.id == constraint_id:
                return constraint

        return None

    def by_category(self, category: str) -> list[Constraint]:
        """Return constraints from one category."""
        return [
            constraint
            for constraint in self.constraints
            if constraint.category == category
        ]

    def __len__(self) -> int:
        return len(self.constraints)