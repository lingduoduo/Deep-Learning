from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


EPISODIC_INDEX = "atlas-episodic"
SEMANTIC_INDEX = "atlas-semantic"
PROCEDURAL_INDEX = "atlas-procedural"
CATALOG_INDEX = "atlas-catalog"
MEMORY_INDICES = [EPISODIC_INDEX, SEMANTIC_INDEX, PROCEDURAL_INDEX, CATALOG_INDEX]
CATALOG_SOURCE_PRIOR = 0.85

INDEX_SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
}

COMMON_PROPERTIES = {
    "user_id": {"type": "keyword"},
    "text": {"type": "text"},
    "memory_type": {"type": "keyword"},
    "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
    "timestamp": {"type": "date", "ignore_malformed": True},
    "metadata": {"type": "object", "enabled": True},
    "index_name": {"type": "keyword"},
    "confidence": {"type": "float"},
    "supersedes": {"type": "keyword"},
    "superseded_by": {"type": "keyword"},
    "superseded_at": {"type": "date", "ignore_malformed": True},
    "supporting_episode_ids": {"type": "keyword"},
    "last_used_at": {"type": "date", "ignore_malformed": True},
    "use_count": {"type": "integer"},
}

INDEX_MAPPINGS = {
    EPISODIC_INDEX: {"properties": COMMON_PROPERTIES},
    SEMANTIC_INDEX: {"properties": COMMON_PROPERTIES},
    PROCEDURAL_INDEX: {
        "properties": {
            **COMMON_PROPERTIES,
            "steps": {"type": "text"},
            "trigger_text": {"type": "text"},
            "success_count": {"type": "integer"},
            "failure_count": {"type": "integer"},
        }
    },
    CATALOG_INDEX: {
        "properties": {
            **COMMON_PROPERTIES,
            "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "description": {"type": "text"},
        }
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_client(url: str = "http://localhost:9200", api_key: Any | None = None) -> Any:
    from elasticsearch import Elasticsearch

    if api_key is None:
        return Elasticsearch(url)
    return Elasticsearch(url, api_key=api_key)


def build_dls_query(user_id: str) -> dict[str, Any]:
    """
    Cluster-level DLS query for one user's API key.

    User-owned memories must match user_id. Catalog documents are shared by
    omitting user_id, so they remain visible to every tenant.
    """
    return {
        "bool": {
            "should": [
                {"term": {"user_id": user_id}},
                {"bool": {"must_not": {"exists": {"field": "user_id"}}}},
            ],
            "minimum_should_match": 1,
        }
    }


def build_user_api_key_role_descriptor(user_id: str) -> dict[str, Any]:
    return {
        f"atlas_memory_{user_id}": {
            "indices": [
                {
                    "names": MEMORY_INDICES,
                    "privileges": ["read", "view_index_metadata"],
                    "query": build_dls_query(user_id),
                }
            ]
        }
    }


def create_user_api_key(client: Any, user_id: str) -> dict[str, Any]:
    """
    Mint one Elasticsearch API key scoped by Document-Level Security.

    Call this from an admin/bootstrap client. The returned API key should be
    used by the application when serving that specific user, so Elasticsearch
    enforces tenant isolation before application code receives any hits.
    """
    return client.security.create_api_key(
        name=f"atlas-memory-{user_id}",
        role_descriptors=build_user_api_key_role_descriptor(user_id),
        metadata={"user_id": user_id, "purpose": "atlas-memory"},
    )


def ensure_memory_indices(client: Any, recreate: bool = False) -> list[str]:
    """
    Create the Elasticsearch indices used by the memory sample.

    Set recreate=True for local demos when you want deterministic output and do
    not mind deleting the four atlas-* sample indices first.
    """
    created = []
    for index in MEMORY_INDICES:
        if recreate:
            client.indices.delete(index=index, ignore_unavailable=True)

        if _index_exists(client, index):
            continue

        client.indices.create(
            index=index,
            settings=INDEX_SETTINGS,
            mappings=INDEX_MAPPINGS[index],
        )
        created.append(index)
    return created


def build_memory_document(doc: dict[str, Any], index: str) -> dict[str, Any]:
    timestamp = doc.get("timestamp", utc_now_iso())
    memory_type = doc.get("memory_type") or _memory_type_from_index(index)
    stored = {
        "user_id": doc.get("user_id"),
        "text": doc["text"],
        "memory_type": memory_type,
        "title": doc.get("title", ""),
        "timestamp": timestamp,
        "metadata": doc.get("metadata", {}),
        "index_name": _index_alias(index),
    }

    for field in (
        "confidence",
        "supersedes",
        "superseded_by",
        "superseded_at",
        "supporting_episode_ids",
        "steps",
        "trigger_text",
        "success_count",
        "failure_count",
        "last_used_at",
        "use_count",
    ):
        if field in doc:
            stored[field] = doc[field]

    return {key: value for key, value in stored.items() if value is not None}


def write_memory(client: Any, doc: dict[str, Any], index: str) -> str:
    """
    Write one memory document and refresh immediately for same-turn recall.

    The sample uses Elasticsearch as a search-optimized memory layer, so writes
    preserve optional semantic/procedural fields instead of flattening every
    memory into only text and metadata.
    """
    body = build_memory_document(doc, index)
    try:
        resp = client.index(index=index, document=body, refresh=True)
    except TypeError:
        resp = client.index(index=index, body=body, refresh=True)
    return resp["_id"]


def build_recall_query(
    user_id: str,
    query: str,
    include_superseded: bool = False,
) -> dict[str, Any]:
    bool_query: dict[str, Any] = {
        "should": [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["text^3", "title^2", "name", "description", "trigger_text"],
                    "type": "best_fields",
                }
            }
        ],
        "filter": [build_user_or_catalog_filter(user_id)],
    }

    if not include_superseded:
        bool_query["must_not"] = [{"exists": {"field": "superseded_by"}}]

    return {
        "query": {
            "function_score": {
                "query": {"bool": bool_query},
                "functions": [
                    {
                        "filter": {"term": {"_index": CATALOG_INDEX}},
                        "weight": CATALOG_SOURCE_PRIOR,
                    }
                ],
                "score_mode": "multiply",
                "boost_mode": "multiply",
            }
        }
    }


def build_user_or_catalog_filter(user_id: str) -> dict[str, Any]:
    """
    Redundant app-layer tenant filter.

    DLS should enforce isolation in the cluster itself. This filter repeats the
    same user-or-catalog rule in application queries as defense in depth.
    """
    return {
        "bool": {
            "should": [
                {"term": {"user_id": user_id}},
                {"bool": {"must_not": [{"exists": {"field": "user_id"}}]}},
            ],
            "minimum_should_match": 1,
        }
    }


def recall_memory(
    client: Any,
    user_id: str,
    query: str,
    k: int = 5,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """
    Recall memories with a robust local-first search path.

    A production cluster can extend this query with RRF, dense retrieval, and a
    cross-encoder reranker. This sample keeps the default path runnable on a
    plain Elasticsearch install by using BM25 and metadata filters.
    """
    body = build_recall_query(user_id, query, include_superseded=include_superseded)
    resp = client.search(index=MEMORY_INDICES, body=body, size=k)
    return [_format_hit(hit, source="bm25") for hit in resp.get("hits", {}).get("hits", [])[:k]]


def supersede_fact(
    client: Any,
    user_id: str,
    old_fact_id: str,
    new_text: str,
    contradiction_type: str = "natural",
) -> str:
    """
    Add a new semantic fact without deleting the old one.

    Supersession forms an audit chain of arbitrary length:
    old -> newer -> newest. The old document receives superseded_by, and the
    new document receives supersedes. Use recall_memory(...,
    include_superseded=True) when you need historical preferences or the full
    chain for debugging.

    A harsh contradiction is an explicit user correction, such as "I never said
    that." The new fact gets a small confidence penalty until future
    interactions confirm it.
    """
    confidence = 1.0
    if contradiction_type == "harsh":
        confidence -= 0.1

    new_id = write_memory(
        client,
        {
            "user_id": user_id,
            "text": new_text,
            "memory_type": "semantic",
            "confidence": confidence,
            "supersedes": old_fact_id,
            "timestamp": "now",
        },
        index=SEMANTIC_INDEX,
    )

    client.update(
        index=SEMANTIC_INDEX,
        id=old_fact_id,
        body={"doc": {"superseded_by": new_id, "superseded_at": "now"}},
    )
    return new_id


def _format_hit(hit: dict[str, Any], source: str) -> dict[str, Any]:
    doc = hit.get("_source", {})
    return {
        "id": hit.get("_id"),
        "index": hit.get("_index"),
        "text": doc.get("text", ""),
        "title": doc.get("title", ""),
        "memory_type": doc.get("memory_type", ""),
        "score": hit.get("_score", 0.0),
        "source": source,
        "metadata": doc.get("metadata", {}),
        "steps": doc.get("steps", []),
        "trigger_text": doc.get("trigger_text", ""),
    }


def _index_alias(index: str) -> str:
    return index.replace("atlas-", "")


def _memory_type_from_index(index: str) -> str:
    return _index_alias(index)


def _index_exists(client: Any, index: str) -> bool:
    response = client.indices.exists(index=index)
    if isinstance(response, bool):
        return response
    return bool(response)
