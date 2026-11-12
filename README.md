# rag-eval-ci

A zero-bloat, deterministic **RAG faithfulness & hallucination regression gate** for CI/CD pipelines and GitHub Actions.

## Why `rag-eval-ci`?
LLM-based evaluation suites can be slow, expensive, and non-deterministic in pull request checks. `rag-eval-ci` provides a fast, deterministic sentence-level claim-grounding check that runs in milliseconds with **zero external API keys**, failing PRs when:
- Average **Faithfulness** drops below `--min-faithfulness`
- **Hallucination Rate** exceeds `--max-hallucination-rate`

## CLI Quickstart

```bash
pip install -e .
rag-eval-ci run \
  --dataset examples/sample_eval.jsonl \
  --min-faithfulness 0.80 \
  --max-hallucination-rate 0.10 \
  --markdown-out report.md
```

## Use as a GitHub Action

```yaml
- uses: PARV5544/rag-eval-ci@main
  with:
    dataset: evals/rag_golden.jsonl
    min-faithfulness: "0.80"
    max-hallucination-rate: "0.15"
```

## Sample Output

| Metric | Value | Threshold |
|---|---|---|
| Samples Evaluated | 3 | - |
| Avg Faithfulness | 100.00% | >= 80.00% |
| Avg Answer Relevance | 75.00% | - |
| Hallucination Rate | 0.00% | <= 10.00% |
