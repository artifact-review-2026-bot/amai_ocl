from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_HARD_REASON_PREFIX = "hard_constraint:"


@dataclass(slots=True)
class ConstraintFeedback:
    constraint_id: str
    evaluation_count: int = 0
    retrieval_event_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    @property
    def effectiveness(self) -> float | None:
        if self.evaluation_count == 0:
            return None

        return self.success_count / self.evaluation_count


@dataclass(slots=True)
class ConstraintFeedbackStore:
    feedback: dict[str, ConstraintFeedback] = field(
        default_factory=dict
    )

    def get_or_create(
        self,
        constraint_id: str,
    ) -> ConstraintFeedback:
        if constraint_id not in self.feedback:
            self.feedback[constraint_id] = ConstraintFeedback(
                constraint_id=constraint_id
            )

        return self.feedback[constraint_id]

    def to_dict(self) -> dict[str, Any]:
        items = {}

        for constraint_id in sorted(self.feedback):
            item = self.feedback[constraint_id]

            data = asdict(item)
            data["effectiveness"] = item.effectiveness

            items[constraint_id] = data

        return {
            "schema_version": 1,
            "feedback": items,
        }

    def save_json(
        self,
        path: str | Path,
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
                self.to_dict(),
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ) -> "ConstraintFeedbackStore":
        input_path = Path(path)

        if not input_path.exists():
            return cls()

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if data.get("schema_version") != 1:
            raise ValueError(
                "Unsupported ConstraintFeedbackStore schema_version."
            )

        store = cls()

        for constraint_id, raw in (
            data.get("feedback") or {}
        ).items():
            store.feedback[constraint_id] = ConstraintFeedback(
                constraint_id=constraint_id,
                evaluation_count=int(
                    raw.get("evaluation_count", 0)
                ),
                retrieval_event_count=int(
                    raw.get("retrieval_event_count", 0)
                ),
                success_count=int(
                    raw.get("success_count", 0)
                ),
                failure_count=int(
                    raw.get("failure_count", 0)
                ),
            )

        return store


def _hard_constraint_ids(
    retrieved_constraint: dict[str, Any],
) -> set[str]:
    hard_ids = set()

    for reason in retrieved_constraint.get(
        "reasons",
        [],
    ):
        reason = str(reason)

        if reason.startswith(_HARD_REASON_PREFIX):
            hard_id = reason[
                len(_HARD_REASON_PREFIX):
            ].strip()

            if hard_id:
                hard_ids.add(hard_id)

    return hard_ids


def update_feedback_from_trajectory(
    store: ConstraintFeedbackStore,
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate retrieved learned constraints using downstream hard-constraint
    outcomes from one completed episode.

    A constraint is evaluated once per episode for each constraint id.
    Repeated retrieval events are counted separately but do not create
    duplicate effectiveness evaluations.

    Success:
        None of the hard constraints that caused retrieval are triggered
        from the first retrieval round onward.

    Failure:
        At least one corresponding hard constraint is triggered from the
        first retrieval round onward.
    """

    exposures: dict[str, dict[str, Any]] = {}

    for index, step in enumerate(trajectory):
        for retrieved in step.get(
            "retrieved_constraints",
            [],
        ):
            constraint_id = str(
                retrieved.get("id", "")
            ).strip()

            if not constraint_id:
                continue

            hard_ids = _hard_constraint_ids(retrieved)

            if not hard_ids:
                continue

            feedback = store.get_or_create(
                constraint_id
            )
            feedback.retrieval_event_count += 1

            if constraint_id not in exposures:
                exposures[constraint_id] = {
                    "first_index": index,
                    "hard_ids": set(hard_ids),
                }
            else:
                exposures[constraint_id][
                    "hard_ids"
                ].update(hard_ids)

    evaluations = []

    for constraint_id, exposure in exposures.items():
        first_index = exposure["first_index"]
        hard_ids = exposure["hard_ids"]

        later_failures = set()

        for step in trajectory[first_index:]:
            later_failures.update(
                step.get(
                    "failed_hard_constraints",
                    [],
                )
                or []
            )

        matched_failures = sorted(
            hard_ids & later_failures
        )

        feedback = store.get_or_create(
            constraint_id
        )
        feedback.evaluation_count += 1

        if matched_failures:
            feedback.failure_count += 1
            outcome = "failure"
        else:
            feedback.success_count += 1
            outcome = "success"

        evaluations.append(
            {
                "constraint_id": constraint_id,
                "hard_constraint_ids": sorted(
                    hard_ids
                ),
                "first_retrieval_round": trajectory[
                    first_index
                ].get("round_id"),
                "outcome": outcome,
                "matched_failures": matched_failures,
            }
        )

    return {
        "evaluated_constraints": len(evaluations),
        "evaluations": evaluations,
    }
