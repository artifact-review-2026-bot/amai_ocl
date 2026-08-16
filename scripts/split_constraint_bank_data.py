"""Create reproducible build/evaluation splits for OCL-v2 ConstraintBank.

The first run creates a stratified deterministic split and records every
assignment in a manifest. Later runs preserve all existing assignments and
only assign newly added samples, so expanding the dataset does not reshuffle
old samples between build and evaluation sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = ROOT / "configs" / "adversarial_buyers.json"
DEFAULT_BUILD_OUTPUT = ROOT / "configs" / "constraint_bank_build.json"
DEFAULT_EVAL_OUTPUT = ROOT / "configs" / "constraint_bank_eval.json"
DEFAULT_MANIFEST = ROOT / "configs" / "constraint_bank_split_manifest.json"

SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        help="JSON list to include. Repeat --input for additional datasets.",
    )
    parser.add_argument("--build-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-field", default="persona_type")

    parser.add_argument(
        "--build-output",
        type=Path,
        default=DEFAULT_BUILD_OUTPUT,
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=DEFAULT_EVAL_OUTPUT,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    return parser.parse_args()


def _stable_id(record: dict[str, Any]) -> str:
    """Return a stable sample id, preferring explicit ids over names."""

    for field in ("id", "buyer_id", "sample_id", "name"):
        value = record.get(field)

        if value is not None and str(value).strip():
            return str(value).strip()

    canonical = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        "content:"
        + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    )


def _rank(seed: int, category: str, sample_id: str) -> str:
    """Return a deterministic ranking value for one sample."""

    payload = f"{seed}|{category}|{sample_id}".encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the SHA256 fingerprint of an input file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_records(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load one or more JSON datasets."""

    records: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    seen_ids: set[str] = set()

    for raw_path in paths:
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")

        fingerprints[str(path.relative_to(ROOT))] = _file_sha256(path)

        for record in data:
            if not isinstance(record, dict):
                raise ValueError(
                    f"Every sample must be a JSON object in {path}"
                )

            sample_id = _stable_id(record)

            if sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate stable sample id {sample_id!r}. "
                    "Add an explicit unique id/buyer_id to the data."
                )

            seen_ids.add(sample_id)
            records.append(record)

    return records, fingerprints


def _load_manifest(
    path: Path,
    *,
    seed: int,
    build_ratio: float,
    stratify_field: str,
) -> dict[str, Any]:
    """Load an existing split manifest or create a new one."""

    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "build_ratio": build_ratio,
            "stratify_field": stratify_field,
            "assignments": {},
        }

    manifest = json.loads(path.read_text(encoding="utf-8"))

    expected = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "build_ratio": build_ratio,
        "stratify_field": stratify_field,
    }

    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Existing manifest has {key}="
                f"{manifest.get(key)!r}, expected {value!r}. "
                "Use the original split settings or create "
                "a new versioned manifest."
            )

    manifest.setdefault("assignments", {})

    return manifest


def create_split(
    records: list[dict[str, Any]],
    *,
    seed: int,
    build_ratio: float,
    stratify_field: str,
    manifest: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Create a stable build/evaluation split."""

    if not 0.0 < build_ratio < 1.0:
        raise ValueError("build_ratio must be between 0 and 1")

    by_category: dict[
        str,
        list[tuple[str, dict[str, Any]]],
    ] = defaultdict(list)

    current_ids: set[str] = set()

    for record in records:
        sample_id = _stable_id(record)
        category = str(
            record.get(stratify_field, "__all__")
        )

        current_ids.add(sample_id)
        by_category[category].append(
            (sample_id, record)
        )

    assignments: dict[
        str,
        dict[str, str],
    ] = manifest["assignments"]

    for category in sorted(by_category):
        items = by_category[category]

        existing_build = sum(
            1
            for sample_id, _ in items
            if assignments.get(
                sample_id,
                {},
            ).get("split") == "build"
        )

        target_build = int(
            len(items) * build_ratio + 0.5
        )

        new_items = [
            (sample_id, record)
            for sample_id, record in items
            if sample_id not in assignments
        ]

        new_items.sort(
            key=lambda item: _rank(
                seed,
                category,
                item[0],
            )
        )

        build_needed = max(
            0,
            target_build - existing_build,
        )

        for index, (sample_id, _) in enumerate(
            new_items
        ):
            split = (
                "build"
                if index < build_needed
                else "eval"
            )

            assignments[sample_id] = {
                "split": split,
                "category": category,
            }

    build_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []

    category_counts: dict[
        str,
        dict[str, int],
    ] = defaultdict(
        lambda: {
            "build": 0,
            "eval": 0,
        }
    )

    for record in sorted(
        records,
        key=_stable_id,
    ):
        sample_id = _stable_id(record)
        assignment = assignments[sample_id]

        split = assignment["split"]
        category = str(
            record.get(
                stratify_field,
                "__all__",
            )
        )

        category_counts[category][split] += 1

        if split == "build":
            build_records.append(record)

        elif split == "eval":
            eval_records.append(record)

        else:
            raise ValueError(
                f"Unknown split {split!r} "
                f"for sample {sample_id!r}"
            )

    manifest["current_sample_ids"] = sorted(
        current_ids
    )
    manifest["build_count"] = len(
        build_records
    )
    manifest["eval_count"] = len(
        eval_records
    )
    manifest["category_counts"] = dict(
        sorted(category_counts.items())
    )

    return (
        build_records,
        eval_records,
        manifest,
    )


def main() -> None:
    args = parse_args()

    inputs = args.inputs or [DEFAULT_INPUT]

    records, fingerprints = _load_records(
        inputs
    )

    manifest_path = (
        args.manifest
        if args.manifest.is_absolute()
        else ROOT / args.manifest
    )

    manifest = _load_manifest(
        manifest_path,
        seed=args.seed,
        build_ratio=args.build_ratio,
        stratify_field=args.stratify_field,
    )

    build_records, eval_records, manifest = (
        create_split(
            records,
            seed=args.seed,
            build_ratio=args.build_ratio,
            stratify_field=args.stratify_field,
            manifest=manifest,
        )
    )

    manifest["sources"] = fingerprints

    build_path = (
        args.build_output
        if args.build_output.is_absolute()
        else ROOT / args.build_output
    )

    eval_path = (
        args.eval_output
        if args.eval_output.is_absolute()
        else ROOT / args.eval_output
    )

    build_path.write_text(
        json.dumps(
            build_records,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    eval_path.write_text(
        json.dumps(
            eval_records,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Build set: {len(build_records)}"
    )
    print(
        f"Evaluation set: {len(eval_records)}"
    )

    for category, counts in (
        manifest["category_counts"].items()
    ):
        print(
            f"{category}: "
            f"build={counts['build']}, "
            f"eval={counts['eval']}"
        )

    print(
        "Manifest: "
        f"{manifest_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()