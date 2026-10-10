from scripts.extract_on_policy_repairs import extract_later_successes


def test_extracts_only_later_success_with_feedback() -> None:
    failed_step = {
        "repair_index": 1,
        "prompt": "Public-test execution feedback:\nfirst",
        "candidate_code": "def f():\n    return 0\n",
        "public_after": {"all_passed": False},
    }
    successful_step = {
        "repair_index": 2,
        "prompt": "Public-test execution feedback:\nsecond",
        "candidate_code": "def f():\n    return 1\n",
        "public_after": {"all_passed": True},
    }
    rows = [
        {
            "task_id": "t1",
            "bug_id": "b1",
            "bug_type": "logic_error",
            "repair_at_1": False,
            "repair_at_k": True,
            "trajectory": [failed_step, successful_step],
        },
        {
            "task_id": "t2",
            "bug_id": "b2",
            "bug_type": "logic_error",
            "repair_at_1": True,
            "repair_at_k": True,
            "trajectory": [successful_step],
        },
    ]

    samples = extract_later_successes(rows)

    assert len(samples) == 1
    assert samples[0]["bug_id"] == "b1"
    assert samples[0]["source_repair_index"] == 2
    assert samples[0]["completion"] == "def f():\n    return 1\n"
