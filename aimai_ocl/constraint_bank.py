from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Constraint:
    id: str
    category: str
    rule: str
    when: str


@dataclass(slots=True)
class ConstraintBank:
    constraints: list[Constraint] = field(default_factory=list)

    def add(self, constraint: Constraint) -> None:
        if self.get(constraint.id) is not None:
            raise ValueError(
                f"Duplicate constraint id: {constraint.id}"
            )

        self.constraints.append(constraint)

    def get(self, constraint_id: str) -> Constraint | None:
        for constraint in self.constraints:
            if constraint.id == constraint_id:
                return constraint

        return None

    def by_category(self, category: str) -> list[Constraint]:
        return [
            constraint
            for constraint in self.constraints
            if constraint.category == category
        ]

    def __len__(self) -> int:
        return len(self.constraints)

    def to_dict(
        self,
        *,
        version: str = "v1",
    ) -> dict:
        return {
            "schema_version": 1,
            "bank_version": version,
            "constraints": [
                asdict(constraint)
                for constraint in sorted(
                    self.constraints,
                    key=lambda item: item.id,
                )
            ],
        }

    def save_json(
        self,
        path: str | Path,
        *,
        version: str = "v1",
    ) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.to_dict(version=version),
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ) -> "ConstraintBank":
        input_path = Path(path)

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if data.get("schema_version") != 1:
            raise ValueError(
                "Unsupported ConstraintBank schema_version."
            )

        raw_constraints = data.get("constraints")

        if not isinstance(raw_constraints, list):
            raise ValueError(
                "ConstraintBank JSON must contain a constraints list."
            )

        bank = cls()

        for item in raw_constraints:
            constraint = Constraint(
                id=str(item["id"]),
                category=str(item["category"]),
                rule=str(item["rule"]),
                when=str(item["when"]),
            )
            bank.add(constraint)

        return bank
