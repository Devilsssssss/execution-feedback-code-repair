import json

from scripts.compare_summaries import load_summary, main


def test_load_summary_and_compare_cli(tmp_path, monkeypatch) -> None:
    source = tmp_path / "summary.json"
    output = tmp_path / "comparison.json"
    source.write_text(json.dumps({"cases": 2, "repair_at_1": 0.5}), encoding="utf-8")

    assert load_summary(f"run={source}")[0] == "run"
    monkeypatch.setattr(
        "sys.argv",
        ["compare_summaries", f"run={source}", "--output", str(output)],
    )
    assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))[0]["repair_at_1"] == 0.5
