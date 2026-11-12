from rag_eval_ci.scorer import aggregate_results, score_sample


def test_grounded_answer_scores_high():
    res = score_sample(
        sample_id="1",
        question="What does Kafka handle?",
        answer="Kafka handles event streaming with dead-letter retry topics.",
        contexts=["Apache Kafka handles event streaming and dead-letter retry topics."],
    )
    assert res.faithfulness == 1.0
    assert res.hallucinated is False
    assert res.unsupported_sentences == []


def test_hallucinated_answer_is_flagged():
    res = score_sample(
        sample_id="2",
        question="What database is used?",
        answer="The system stores all graphs in Neo4j AuraDB. It also uses Cassandra for logs.",
        contexts=["PostgreSQL stores audit tables with optimized B-tree indexes."],
    )
    assert res.faithfulness == 0.0
    assert res.hallucinated is True
    assert len(res.unsupported_sentences) == 2


def test_aggregate_results():
    r1 = score_sample("1", "Q?", "Redis caches results.", ["Redis caches results."])
    r2 = score_sample("2", "Q?", "Quantum teleportation.", ["PostgreSQL indexes."])
    summary = aggregate_results([r1, r2])
    assert summary["total_samples"] == 2
    assert summary["avg_faithfulness"] == 0.5
    assert summary["hallucination_rate"] == 0.5
