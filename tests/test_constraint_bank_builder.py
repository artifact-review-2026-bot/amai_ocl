from aimai_ocl.constraint_bank_builder import build_bank


def test_build_bank_from_multiple_trajectories():
    records = [
        {
            "trajectory": [
                {
                    "round_id": 1,
                    "failed_hard_constraints": [
                        "seller_floor"
                    ],
                    "next_state": {
                        "status": None,
                    },
                },
                {
                    "round_id": 2,
                    "failed_hard_constraints": [
                        "seller_floor"
                    ],
                    "next_state": {
                        "status": "timeout",
                    },
                },
            ]
        },
        {
            "trajectory": [
                {
                    "round_id": 4,
                    "failed_hard_constraints": [
                        "seller_floor"
                    ],
                    "next_state": {
                        "status": None,
                    },
                },
                {
                    "round_id": 5,
                    "failed_hard_constraints": [
                        "seller_floor"
                    ],
                    "next_state": {
                        "status": "timeout",
                    },
                },
            ]
        },
    ]

    bank, stats = build_bank(records)

    assert stats["record_count"] == 2
    assert stats["trajectory_record_count"] == 2
    assert stats["candidate_count"] == 2
    assert stats["accepted_count"] == 1
    assert stats["rejected_count"] == 1

    assert stats["rejection_reasons"] == {
        "duplicate_id": 1
    }

    assert stats["bank_size"] == 1
    assert len(bank) == 1


def test_build_bank_skips_records_without_trajectory():
    records = [
        {
            "arm": "ocl_full",
            "trajectory": [],
        },
        {
            "arm": "ocl_full",
        },
    ]

    bank, stats = build_bank(records)

    assert len(bank) == 0
    assert stats["record_count"] == 2
    assert stats["trajectory_record_count"] == 0
    assert stats["candidate_count"] == 0
    assert stats["bank_size"] == 0
