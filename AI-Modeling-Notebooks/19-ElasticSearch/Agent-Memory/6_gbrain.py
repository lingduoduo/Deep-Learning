from __future__ import annotations

import argparse
from typing import Any

from memory_store import (
    EPISODIC_INDEX,
    PROCEDURAL_INDEX,
    SEMANTIC_INDEX,
    create_client,
    ensure_memory_indices,
    recall_memory,
    supersede_fact,
    write_memory,
)


def seed_deployment_memories(client: Any, user_id: str = "bob") -> dict[str, str]:
    old_fact_id = write_memory(
        client,
        {
            "user_id": user_id,
            "text": "Bob prefers weekly deployment summaries from Grafana.",
            "memory_type": "semantic",
            "title": "Outdated deployment summary preference",
            "confidence": 0.7,
        },
        SEMANTIC_INDEX,
    )
    concise_fact_id = write_memory(
        client,
        {
            "user_id": user_id,
            "text": "Bob prefers concise deployment summaries with action items first.",
            "memory_type": "semantic",
            "title": "Deployment summary style",
            "confidence": 0.95,
        },
        SEMANTIC_INDEX,
    )
    episode_id = write_memory(
        client,
        {
            "user_id": user_id,
            "text": "Bob asked to check failed deployment logs for the payments API last Friday.",
            "memory_type": "message",
            "title": "Payments API deployment log request",
        },
        EPISODIC_INDEX,
    )
    procedure_id = write_memory(
        client,
        {
            "user_id": user_id,
            "text": "Deployment log investigation workflow",
            "memory_type": "procedural",
            "title": "Investigate failed deployment",
            "trigger_text": "when Bob asks about failed deployment logs",
            "steps": [
                "Check failed pods and recent error spikes.",
                "Compare the release diff for the failing service.",
                "Summarize impact, likely cause, and action items.",
            ],
            "confidence": 0.9,
        },
        PROCEDURAL_INDEX,
    )
    return {
        "old_fact_id": old_fact_id,
        "concise_fact_id": concise_fact_id,
        "episode_id": episode_id,
        "procedure_id": procedure_id,
    }


def answer_with_memory(user_message: str, memories: list[dict[str, Any]]) -> str:
    memory_text = " ".join(memory["text"] for memory in memories)
    source = "Kibana" if "Kibana" in memory_text else "the deployment logs"
    style = "concise" if "concise" in memory_text else "short"
    return (
        f"I will check {source} for the failed deployment and keep the summary {style}: "
        "impact first, likely cause second, and action items last."
    )


def run_deployment_memory_use_case(client: Any | None = None) -> dict[str, Any]:
    client = client or create_client()
    user_id = "bob"
    seeded = seed_deployment_memories(client, user_id=user_id)

    new_fact_id = supersede_fact(
        client,
        user_id=user_id,
        old_fact_id=seeded["old_fact_id"],
        new_text="Bob prefers daily deployment summaries from Kibana.",
        contradiction_type="harsh",
    )

    user_message = "Can you check the failed deployment logs and summarize what changed?"
    recalled = recall_memory(client, user_id, user_message, k=5)
    assistant_response = answer_with_memory(user_message, recalled)
    return {
        "user_message": user_message,
        "recalled_memories": recalled,
        "assistant_response": assistant_response,
        "new_fact_id": new_fact_id,
    }


def run_live_elasticsearch_demo(es_url: str = "http://localhost:9200", recreate: bool = False) -> dict[str, Any]:
    client = create_client(es_url)
    ensure_memory_indices(client, recreate=recreate)
    return run_deployment_memory_use_case(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Agent Memory demo against Elasticsearch.")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the atlas-* demo indices before seeding memories.",
    )
    args = parser.parse_args()

    result = run_live_elasticsearch_demo(es_url=args.es_url, recreate=args.recreate)
    print(result["assistant_response"])
    print("\nRecalled memories:")
    for memory in result["recalled_memories"]:
        print(f"- [{memory['memory_type']}] {memory['text']}")
