from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aimai_ocl.constraint_bank_builder import build_bank


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ConstraintBank from structured "
            "benchmark trajectories."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="benchmark_results.json",
    )

    parser.add_argument(
        "--bank-output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--manifest-output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--version",
        default="v1",
    )

    parser.add_argument(
        "--min-evidence-rounds",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    records = data.get("records")

    if not isinstance(records, list):
        raise ValueError(
            "Input JSON must contain a records list."
        )

    bank, stats = build_bank(
        records,
        min_evidence_rounds=args.min_evidence_rounds,
    )

    bank.save_json(
        args.bank_output,
        version=args.version,
    )

    manifest = {
        "schema_version": 1,
        "bank_version": args.version,
        "input_file": str(args.input),
        "input_sha256": _sha256_file(args.input),
        "min_evidence_rounds": args.min_evidence_rounds,
        **stats,
    }

    args.manifest_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.manifest_output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    print(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
