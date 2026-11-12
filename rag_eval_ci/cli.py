import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from rag_eval_ci.scorer import SampleResult, aggregate_results, score_sample


def load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            row.setdefault("id", f"sample-{idx}")
            records.append(row)
    return records


def format_markdown_report(
    summary: dict,
    results: List[SampleResult],
    min_faithfulness: float,
    max_hallucination_rate: float,
    passed: bool,
) -> str:
    status_badge = "PASS" if passed else "FAIL"
    lines = [
        f"## RAG Evaluation Gate: **{status_badge}**",
        "",
        "| Metric | Value | Threshold |",
        "|---|---|---|",
        f"| Samples Evaluated | {summary['total_samples']} | - |",
        f"| Avg Faithfulness | {summary['avg_faithfulness']:.2%} | >= {min_faithfulness:.2%} |",
        f"| Avg Answer Relevance | {summary['avg_answer_relevance']:.2%} | - |",
        f"| Hallucination Rate | {summary['hallucination_rate']:.2%} | <= {max_hallucination_rate:.2%} |",
        "",
    ]

    flagged = [r for r in results if r.hallucinated]
    if flagged:
        lines.append("### Flagged Samples")
        for item in flagged:
            lines.append(
                f"- `{item.sample_id}` (faithfulness: {item.faithfulness:.2f}) — unsupported: {item.unsupported_sentences}"
            )
    return "\n".join(lines) + "\n"


def run_evaluation(
    dataset_path: Path,
    min_faithfulness: float = 0.75,
    max_hallucination_rate: float = 0.25,
    markdown_out: Optional[Path] = None,
) -> int:
    records = load_jsonl(dataset_path)
    results = [
        score_sample(
            sample_id=str(row["id"]),
            question=row["question"],
            answer=row["answer"],
            contexts=row.get("contexts", []),
        )
        for row in records
    ]
    summary = aggregate_results(results)
    passed = (
        summary["avg_faithfulness"] >= min_faithfulness
        and summary["hallucination_rate"] <= max_hallucination_rate
    )

    report = format_markdown_report(
        summary=summary,
        results=results,
        min_faithfulness=min_faithfulness,
        max_hallucination_rate=max_hallucination_rate,
        passed=passed,
    )
    print(report)

    if markdown_out is not None:
        markdown_out.write_text(report, encoding="utf-8")

    return 0 if passed else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-eval-ci",
        description="Run deterministic RAG faithfulness & hallucination checks in CI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Evaluate a JSONL RAG dataset")
    run_parser.add_argument("--dataset", required=True, type=Path, help="Path to JSONL dataset")
    run_parser.add_argument("--min-faithfulness", type=float, default=0.75)
    run_parser.add_argument("--max-hallucination-rate", type=float, default=0.25)
    run_parser.add_argument("--markdown-out", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "run":
        return run_evaluation(
            dataset_path=args.dataset,
            min_faithfulness=args.min_faithfulness,
            max_hallucination_rate=args.max_hallucination_rate,
            markdown_out=args.markdown_out,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
