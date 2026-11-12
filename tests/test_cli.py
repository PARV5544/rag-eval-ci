from pathlib import Path

from rag_eval_ci.cli import main


def test_cli_pass_and_markdown_output(tmp_path: Path):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        '{"id":"1","question":"What is Redis?","answer":"Redis is an in-memory cache.","contexts":["Redis is an in-memory cache used for low-latency lookups."]}\n',
        encoding="utf-8",
    )
    report_path = tmp_path / "report.md"
    exit_code = main(
        [
            "run",
            "--dataset",
            str(dataset),
            "--min-faithfulness",
            "0.8",
            "--max-hallucination-rate",
            "0.2",
            "--markdown-out",
            str(report_path),
        ]
    )
    assert exit_code == 0
    assert "**PASS**" in report_path.read_text(encoding="utf-8")


def test_cli_fails_when_hallucination_exceeds_threshold(tmp_path: Path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        '{"id":"1","question":"What is X?","answer":"Completely fabricated claim about Mars.","contexts":["PostgreSQL audit table indexing."]}\n',
        encoding="utf-8",
    )
    exit_code = main(
        [
            "run",
            "--dataset",
            str(dataset),
            "--min-faithfulness",
            "0.8",
            "--max-hallucination-rate",
            "0.1",
        ]
    )
    assert exit_code == 1
