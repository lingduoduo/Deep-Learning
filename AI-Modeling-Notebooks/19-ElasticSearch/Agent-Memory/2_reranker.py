from __future__ import annotations

from typing import Any

from memory_store import MEMORY_INDICES, build_recall_query, recall_memory


def recall_memory_with_optional_rerank(
    client: Any,
    user_id: str,
    query: str,
    k: int = 5,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """
    Recall different memory types with one Elasticsearch query.

    The important design point is that recall searches across the memory index
    list in a single request. One user question can therefore return:

    - episodic memory: raw conversation events and user turns
    - semantic memory: stable facts, preferences, and profile knowledge
    - procedural memory: reusable steps for tasks the agent has learned

    Local Elasticsearch installs usually do not have Jina embedding and
    reranker inference endpoints configured. The code first tries a lightweight
    rescore query shape, then degrades to the BM25 recall helper.
    """
    try:
        overfetch_k = max(k * 8, 40)
        first_pass = client.search(
            index=MEMORY_INDICES,
            body=build_recall_query(user_id, query, include_superseded=include_superseded),
            size=overfetch_k,
        )
        ids = [hit["_id"] for hit in first_pass.get("hits", {}).get("hits", [])]
        if not ids:
            return []

        reranked = client.search(
            index=MEMORY_INDICES,
            body={
                "query": {"ids": {"values": ids}},
                "rescore": {
                    "window_size": overfetch_k,
                    "query": {
                        "rescore_query": {
                            "multi_match": {
                                "query": query,
                                "fields": ["text^4", "title^2", "trigger_text"],
                            }
                        },
                        "query_weight": 0.7,
                        "rescore_query_weight": 1.3,
                    },
                },
            },
            size=k,
        )
        return [
            {
                "id": hit["_id"],
                "index": hit["_index"],
                "text": hit["_source"].get("text", ""),
                "score": hit.get("_score", 0.0),
                "source": "rescore",
            }
            for hit in reranked.get("hits", {}).get("hits", [])[:k]
        ]
    except Exception as exc:
        print(f"[WARN] Rerank path failed, falling back to BM25 recall: {exc}")
        return recall_memory(
            client,
            user_id,
            query,
            k=k,
            include_superseded=include_superseded,
        )
