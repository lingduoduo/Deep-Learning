import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch


AGENT_MEMORY_DIR = Path(__file__).resolve().parent


def load_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, AGENT_MEMORY_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


memory_store = load_module("memory_store", "memory_store.py")
bootstrap_users_module = load_module("bootstrap_users", "bootstrap_users.py")
gbrain = load_module("gbrain", "6_gbrain.py")
pre_recall = load_module("pre_recall", "3_verbatim_pre_recall.py")


def recall_bool_query(search_body):
    query = search_body["query"]
    if "function_score" in query:
        return query["function_score"]["query"]["bool"]
    return query["bool"]


class FakeElasticsearch:
    def __init__(self):
        self.documents = {}
        self.updated = []
        self.index_calls = []
        self.search_calls = []
        self.indices = FakeIndices()
        self.counter = 0

    def index(self, index, document=None, body=None, refresh=False):
        self.counter += 1
        doc_id = f"{index}-{self.counter}"
        source = document if document is not None else body
        source = dict(source)
        self.documents[(index, doc_id)] = source
        self.index_calls.append({"index": index, "id": doc_id, "document": source, "refresh": refresh})
        return {"_id": doc_id}

    def update(self, index, id, body):
        self.updated.append({"index": index, "id": id, "body": body})
        current = self.documents.get((index, id), {})
        current.update(body["doc"])
        self.documents[(index, id)] = current
        return {"result": "updated"}

    def search(self, index, body, size=10):
        self.search_calls.append({"index": index, "body": body, "size": size})
        indices = index if isinstance(index, list) else [index]
        hits = []
        include_superseded = "must_not" not in recall_bool_query(body)
        for (doc_index, doc_id), source in self.documents.items():
            if doc_index not in indices:
                continue
            if source.get("superseded_by") and not include_superseded:
                continue
            hits.append({"_id": doc_id, "_index": doc_index, "_source": source, "_score": 1.0})
        return {"hits": {"hits": hits[:size]}}


class FakeIndices:
    def __init__(self):
        self.created = {}
        self.deleted = []

    def exists(self, index):
        return index in self.created

    def create(self, index, mappings=None, settings=None):
        self.created[index] = {"mappings": mappings, "settings": settings}
        return {"acknowledged": True}

    def delete(self, index, ignore_unavailable=False):
        self.deleted.append({"index": index, "ignore_unavailable": ignore_unavailable})
        self.created.pop(index, None)
        return {"acknowledged": True}


class FakeSecurity:
    def __init__(self):
        self.api_key_requests = []

    def create_api_key(self, name, role_descriptors, metadata=None):
        self.api_key_requests.append(
            {"name": name, "role_descriptors": role_descriptors, "metadata": metadata}
        )
        return {"id": "api-key-id", "api_key": "secret"}


class FakeSecureElasticsearch(FakeElasticsearch):
    def __init__(self):
        super().__init__()
        self.security = FakeSecurity()


def test_build_dls_query_allows_only_user_or_catalog_docs():
    dls_query = memory_store.build_dls_query("bob")

    assert dls_query == {
        "bool": {
            "should": [
                {"term": {"user_id": "bob"}},
                {"bool": {"must_not": {"exists": {"field": "user_id"}}}},
            ],
            "minimum_should_match": 1,
        }
    }


def test_build_api_key_role_descriptor_embeds_dls_query():
    descriptor = memory_store.build_user_api_key_role_descriptor("bob")

    indices = descriptor["atlas_memory_bob"]["indices"][0]
    assert indices["names"] == memory_store.MEMORY_INDICES
    assert indices["privileges"] == ["read", "view_index_metadata"]
    assert indices["query"] == memory_store.build_dls_query("bob")


def test_create_user_api_key_uses_role_descriptor_and_metadata():
    client = FakeSecureElasticsearch()

    response = memory_store.create_user_api_key(client, "bob")

    assert response == {"id": "api-key-id", "api_key": "secret"}
    assert client.security.api_key_requests == [
        {
            "name": "atlas-memory-bob",
            "role_descriptors": memory_store.build_user_api_key_role_descriptor("bob"),
            "metadata": {"user_id": "bob", "purpose": "atlas-memory"},
        }
    ]


def test_bootstrap_users_creates_indices_and_dls_keys_for_each_user():
    client = FakeSecureElasticsearch()

    with patch.object(bootstrap_users_module, "create_client", return_value=client):
        result = bootstrap_users_module.bootstrap_users("http://localhost:9200", ["bob", "alice"])

    assert set(result) == {"bob", "alice"}
    assert set(client.indices.created) == set(memory_store.MEMORY_INDICES)
    assert [request["name"] for request in client.security.api_key_requests] == [
        "atlas-memory-bob",
        "atlas-memory-alice",
    ]
    assert client.security.api_key_requests[1]["role_descriptors"] == (
        memory_store.build_user_api_key_role_descriptor("alice")
    )


def test_bootstrap_users_requires_at_least_one_user():
    try:
        bootstrap_users_module.bootstrap_users("http://localhost:9200", [])
    except ValueError as exc:
        assert str(exc) == "At least one user_id is required."
    else:
        raise AssertionError("Expected ValueError")


def test_ensure_memory_indices_creates_real_elasticsearch_mappings():
    client = FakeElasticsearch()

    created = memory_store.ensure_memory_indices(client)

    assert created == memory_store.MEMORY_INDICES
    semantic_mapping = client.indices.created["atlas-semantic"]["mappings"]["properties"]
    assert semantic_mapping["text"]["type"] == "text"
    assert semantic_mapping["user_id"]["type"] == "keyword"
    assert semantic_mapping["confidence"]["type"] == "float"
    assert semantic_mapping["superseded_by"]["type"] == "keyword"


def test_ensure_memory_indices_can_recreate_demo_indices():
    client = FakeElasticsearch()
    memory_store.ensure_memory_indices(client)

    memory_store.ensure_memory_indices(client, recreate=True)

    assert client.indices.deleted == [
        {"index": "atlas-episodic", "ignore_unavailable": True},
        {"index": "atlas-semantic", "ignore_unavailable": True},
        {"index": "atlas-procedural", "ignore_unavailable": True},
        {"index": "atlas-catalog", "ignore_unavailable": True},
    ]


def test_write_memory_preserves_extended_fields_and_refreshes_immediately():
    client = FakeElasticsearch()

    doc_id = memory_store.write_memory(
        client,
        {
            "user_id": "bob",
            "text": "Bob prefers concise deployment summaries.",
            "memory_type": "semantic",
            "confidence": 0.9,
            "supersedes": "old_fact",
            "metadata": {"source": "conversation"},
        },
        "atlas-semantic",
    )

    stored = client.documents[("atlas-semantic", doc_id)]
    assert stored["confidence"] == 0.9
    assert stored["supersedes"] == "old_fact"
    assert stored["metadata"] == {"source": "conversation"}
    assert client.index_calls[0]["refresh"] is True


def test_recall_memory_filters_superseded_docs_by_default():
    client = FakeElasticsearch()
    old_id = memory_store.write_memory(
        client,
        {"user_id": "bob", "text": "Bob uses Grafana.", "superseded_by": "new"},
        "atlas-semantic",
    )
    new_id = memory_store.write_memory(
        client,
        {"user_id": "bob", "text": "Bob uses Kibana."},
        "atlas-semantic",
    )

    results = memory_store.recall_memory(client, "bob", "observability dashboard", k=5)

    assert [result["id"] for result in results] == [new_id]
    assert old_id not in [result["id"] for result in results]
    assert recall_bool_query(client.search_calls[0]["body"])["must_not"] == [
        {"exists": {"field": "superseded_by"}}
    ]


def test_recall_memory_can_include_superseded_docs_for_history():
    client = FakeElasticsearch()
    old_id = memory_store.write_memory(
        client,
        {"user_id": "bob", "text": "Bob used MySQL in 2023.", "superseded_by": "new"},
        "atlas-semantic",
    )
    new_id = memory_store.write_memory(
        client,
        {"user_id": "bob", "text": "Bob uses PostgreSQL in 2026."},
        "atlas-semantic",
    )

    results = memory_store.recall_memory(
        client,
        "bob",
        "database preference history",
        k=5,
        include_superseded=True,
    )

    assert {result["id"] for result in results} == {old_id, new_id}
    assert "must_not" not in recall_bool_query(client.search_calls[0]["body"])


def test_recall_query_uses_app_layer_user_or_catalog_filter_and_catalog_prior():
    body = memory_store.build_recall_query("bob", "deployment logs")

    bool_query = recall_bool_query(body)
    assert bool_query["filter"] == [memory_store.build_user_or_catalog_filter("bob")]
    assert body["query"]["function_score"]["functions"] == [
        {
            "filter": {"term": {"_index": memory_store.CATALOG_INDEX}},
            "weight": memory_store.CATALOG_SOURCE_PRIOR,
        }
    ]
    assert body["query"]["function_score"]["boost_mode"] == "multiply"


def test_supersede_fact_creates_new_fact_and_marks_old_fact():
    client = FakeElasticsearch()
    old_id = memory_store.write_memory(
        client,
        {"user_id": "bob", "text": "Bob prefers weekly deployment summaries."},
        "atlas-semantic",
    )

    new_id = memory_store.supersede_fact(
        client,
        user_id="bob",
        old_fact_id=old_id,
        new_text="Bob prefers daily deployment summaries.",
        contradiction_type="harsh",
    )

    new_doc = client.documents[("atlas-semantic", new_id)]
    assert new_doc["confidence"] == 0.9
    assert new_doc["supersedes"] == old_id
    assert client.updated == [
        {
            "index": "atlas-semantic",
            "id": old_id,
            "body": {"doc": {"superseded_by": new_id, "superseded_at": "now"}},
        }
    ]


def test_supersede_fact_supports_chained_history():
    client = FakeElasticsearch()
    first_id = memory_store.write_memory(
        client,
        {"user_id": "bob", "text": "Bob prefers MySQL."},
        "atlas-semantic",
    )

    second_id = memory_store.supersede_fact(
        client,
        user_id="bob",
        old_fact_id=first_id,
        new_text="Bob prefers PostgreSQL.",
    )
    third_id = memory_store.supersede_fact(
        client,
        user_id="bob",
        old_fact_id=second_id,
        new_text="Bob prefers PostgreSQL with pgvector.",
    )

    first_doc = client.documents[("atlas-semantic", first_id)]
    second_doc = client.documents[("atlas-semantic", second_id)]
    third_doc = client.documents[("atlas-semantic", third_id)]
    assert first_doc["superseded_by"] == second_id
    assert second_doc["supersedes"] == first_id
    assert second_doc["superseded_by"] == third_id
    assert third_doc["supersedes"] == second_id


def test_verbatim_pre_recall_uses_original_exact_tokens_before_llm_rewrite():
    client = FakeElasticsearch()
    memory_store.write_memory(
        client,
        {
            "user_id": "bob",
            "text": "Fix postgres v15.3 + pgvector 0.5.1 index recall issue.",
            "memory_type": "message",
        },
        "atlas-episodic",
    )

    class RewritingLLM:
        def __init__(self):
            self.messages = []

        def chat(self, messages):
            self.messages = messages
            return "ok"

    llm = RewritingLLM()
    agent = pre_recall.MemoryAwareAgent(client, llm, "bob")

    agent.run_turn("postgres v15.3 + pgvector 0.5.1 failed again", "turn-123456")

    search_query = recall_bool_query(client.search_calls[0]["body"])["should"][0]["multi_match"]["query"]
    tool_arguments = llm.messages[0]["tool_calls"][0]["function"]["arguments"]
    assert search_query == "postgres v15.3 + pgvector 0.5.1 failed again"
    assert "postgres v15.3 + pgvector 0.5.1 failed again" in tool_arguments


def test_gbrain_use_case_returns_response_and_supersedes_stale_fact():
    client = FakeElasticsearch()

    result = gbrain.run_deployment_memory_use_case(client)

    assert "Kibana" in result["assistant_response"]
    assert "concise" in result["assistant_response"]
    assert result["new_fact_id"] is not None
    memory_types = {memory["memory_type"] for memory in result["recalled_memories"]}
    assert {"message", "semantic", "procedural"}.issubset(memory_types)
    superseded_updates = [update for update in client.updated if update["index"] == "atlas-semantic"]
    assert len(superseded_updates) == 1
