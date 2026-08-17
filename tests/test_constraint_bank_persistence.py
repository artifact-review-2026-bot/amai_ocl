import json

from aimai_ocl.constraint_bank import (
    Constraint,
    ConstraintBank,
)


def test_constraint_bank_save_and_load(tmp_path):
    bank = ConstraintBank()

    bank.add(
        Constraint(
            id="constraint_b",
            category="boundary",
            when="When B happens.",
            rule="Do B safely.",
        )
    )

    bank.add(
        Constraint(
            id="constraint_a",
            category="boundary",
            when="When A happens.",
            rule="Do A safely.",
        )
    )

    output = tmp_path / "constraint_bank.json"

    bank.save_json(
        output,
        version="v1",
    )

    loaded = ConstraintBank.load_json(output)

    assert len(loaded) == 2
    assert loaded.get("constraint_a") is not None
    assert loaded.get("constraint_b") is not None

    with output.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    assert data["schema_version"] == 1
    assert data["bank_version"] == "v1"

    ids = [
        item["id"]
        for item in data["constraints"]
    ]

    assert ids == [
        "constraint_a",
        "constraint_b",
    ]


def test_constraint_bank_rejects_unsupported_schema(tmp_path):
    output = tmp_path / "constraint_bank.json"

    output.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "bank_version": "v1",
                "constraints": [],
            }
        ),
        encoding="utf-8",
    )

    try:
        ConstraintBank.load_json(output)
    except ValueError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError(
            "Expected unsupported schema to fail."
        )
