#!/usr/bin/env python3
"""
elasticsearch_vector_demo_books.py

Optimized version:
- Synthetic in-script dataset
- Faster bulk indexing
- Safer embedding batching + auto embedding dims detection
- Modern ES knn query style
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


# -------------------------
# Configuration
# -------------------------

@dataclass(frozen=True)
class Settings:
    # Elasticsearch
    es_host: str = os.getenv("ES_HOST", "http://127.0.0.1:9200")
    es_user: Optional[str] = os.getenv("ES_USER")
    es_password: Optional[str] = os.getenv("ES_PASSWORD")
    verify_certs: bool = os.getenv("ES_VERIFY_CERTS", "false").lower() == "true"
    index_name: str = os.getenv("ES_INDEX", "books_demo")

    # HF embedding model (auto-download)
    hf_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    hf_token: Optional[str] = os.getenv("HF_TOKEN")

    # If set, we will validate against model output dims
    embedding_dims: int = int(os.getenv("EMBED_DIMS", "1536"))

    # Embedding settings
    embedding_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "16"))
    embedding_max_length: int = int(os.getenv("EMBED_MAX_LENGTH", "512"))
    device: str = os.getenv("DEVICE", "auto")  # auto | cpu | cuda | mps


# -------------------------
# Synthetic Dataset
# -------------------------

def make_synthetic_books() -> List[Dict[str, Any]]:
    return [
        {
            "title": "Elasticsearch Basics",
            "author": "Emily",
            "published_date": "2024-01-01",
            "tags": ["search", "elasticsearch"],
            "views": 1024,
        },
        {
            "title": "Lucene Internals: Inverted Index and Doc Values",
            "author": "Raj",
            "published_date": "2023-11-15",
            "tags": ["lucene", "search", "indexing"],
            "views": 870,
        },
        {
            "title": "Hybrid Search: BM25 + Vector Retrieval",
            "author": "Mina",
            "published_date": "2024-06-18",
            "tags": ["hybrid-search", "bm25", "vector-search"],
            "views": 1540,
        },
        {
            "title": "Vector Databases vs Elasticsearch for RAG",
            "author": "Chris",
            "published_date": "2024-09-02",
            "tags": ["rag", "vector-search", "elasticsearch"],
            "views": 2210,
        },
        {
            "title": "Scaling Elasticsearch Clusters: Shards, Replicas, and Nodes",
            "author": "Ava",
            "published_date": "2024-02-12",
            "tags": ["clusters", "shards", "performance"],
            "views": 1310,
        },
        {
            "title": "Practical Aggregations: Facets for E-commerce Search",
            "author": "Noah",
            "published_date": "2024-04-20",
            "tags": ["aggregations", "facets", "ecommerce"],
            "views": 980,
        },
        {
            "title": "Search Relevance Tuning with Query DSL",
            "author": "Emily",
            "published_date": "2024-08-08",
            "tags": ["relevance", "query-dsl", "boosting"],
            "views": 1675,
        },
        {
            "title": "Observability Logs in Elasticsearch: Index Lifecycle and Tiers",
            "author": "Sam",
            "published_date": "2024-03-30",
            "tags": ["observability", "ilm", "hot-warm-cold"],
            "views": 1135,
        },
    ]


def doc_to_embedding_text(doc: Dict[str, Any]) -> str:
    title = doc.get("title", "")
    author = doc.get("author", "")
    tags = doc.get("tags", [])
    tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
    return f"Title: {title}\nAuthor: {author}\nTags: {tags_str}"


def build_query_text(user_query: str) -> str:
    task = "Given a library search query, retrieve the most relevant book entries."
    return f"Instruct: {task}\nQuery: {user_query}"


# -------------------------
# Elasticsearch: Mapping + Client
# -------------------------

def build_index_mapping(dims: int) -> Dict[str, Any]:
    return {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "author": {"type": "keyword"},
                "published_date": {"type": "date"},
                "tags": {"type": "keyword"},
                "views": {"type": "integer"},
                "docEmbedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        }
    }


def connect_es(cfg: Settings) -> Elasticsearch:
    kwargs = {
        "hosts": [cfg.es_host],
        "verify_certs": cfg.verify_certs,
        "request_timeout": 10,
    }
    if cfg.es_user and cfg.es_password:
        kwargs["basic_auth"] = (cfg.es_user, cfg.es_password)

    es = Elasticsearch(**kwargs)

    # Prefer info() over ping()
    info = es.info()
    print(f"[OK] Connected: {info['cluster_name']} (v{info['version']['number']}) @ {cfg.es_host}")
    return es


def recreate_index(es: Elasticsearch, index_name: str, index_mapping: Dict[str, Any]) -> None:
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"[OK] Index '{index_name}' DELETED.")
    es.indices.create(index=index_name, body=index_mapping)
    print(f"[OK] Index '{index_name}' CREATED.")


def bulk_index_books(
    es: Elasticsearch,
    index_name: str,
    books: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> int:
    if len(books) != len(embeddings):
        raise ValueError("books and embeddings length mismatch")

    actions = []
    for i, (b, emb) in enumerate(zip(books, embeddings)):
        doc = dict(b)
        doc["docEmbedding"] = emb
        actions.append(
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": str(i),
                "_source": doc,
            }
        )

    # bulk returns (success_count, errors)
    success, errors = bulk(es, actions, raise_on_error=False, request_timeout=120)
    if errors:
        # errors is a list of per-item failures; print a short summary
        print(f"[WARN] Bulk had {len(errors)} errors (showing up to 2): {errors[:2]}")
    # Ensure docs are searchable immediately in a demo context
    es.indices.refresh(index=index_name)
    return int(success)


# -------------------------
# Embeddings (HF auto-download)
# -------------------------

def pick_device(cfg: Settings) -> str:
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def last_token_pool(last_hidden_states, attention_mask):
    """
    Pooling: take the last non-padding token representation.
    Works for both left- and right-padding.
    """
    import torch

    # attention_mask: 1 for real tokens, 0 for pads
    # For left-padding, last position is always real tokens => sum equals batch size
    left_padding = (attention_mask[:, -1].sum().item() == attention_mask.shape[0])

    if left_padding:
        return last_hidden_states[:, -1]

    seq_lens = attention_mask.sum(dim=1) - 1  # index of last real token
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lens]


def l2_normalize(x, eps: float = 1e-12):
    import torch

    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps)


def load_hf_embedder(cfg: Settings):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = pick_device(cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.hf_model_id,
        padding_side="left",
        trust_remote_code=True,
        token=cfg.hf_token,
    )
    model = AutoModel.from_pretrained(
        cfg.hf_model_id,
        trust_remote_code=True,
        token=cfg.hf_token,
    )

    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return tokenizer, model, device


def _batched(iterable: List[str], batch_size: int) -> Iterable[Tuple[int, int, List[str]]]:
    n = len(iterable)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end, iterable[start:end]


def batch_embedding_hf(
    tokenizer,
    model,
    device: str,
    texts: List[str],
    batch_size: int,
    max_length: int,
) -> List[List[float]]:
    import torch

    results: List[List[float]] = []
    print(f"[EMB] Embedding {len(texts)} texts on {device} using {getattr(model, 'name_or_path', 'hf-model')}")

    for _, _, batch_texts in _batched(texts, batch_size):
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        emb = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        emb = l2_normalize(emb).to("cpu", dtype=torch.float32)
        results.extend(emb.numpy().tolist())


    return results


# -------------------------
# Search Queries
# -------------------------

def keyword_search(es: Elasticsearch, index_name: str, user_query: str, size: int = 5):
    dsl = {
        "query": {
            "multi_match": {
                "query": user_query,
                "fields": ["title^2", "author"],
                "type": "best_fields",
            }
        },
        "sort": [{"views": "desc"}],
        "size": size,
    }
    return es.search(index=index_name, body=dsl)["hits"]["hits"]


def facet_by_tags(es: Elasticsearch, index_name: str, size: int = 10):
    dsl = {
        "size": 0,
        "aggs": {"top_tags": {"terms": {"field": "tags", "size": size}}},
    }
    return es.search(index=index_name, body=dsl)["aggregations"]["top_tags"]["buckets"]


def knn_search(es: Elasticsearch, index_name: str, query_vector: List[float], k: int = 50, size: int = 5):
    """
    Modern knn style:
    - k controls candidate count (recall). size controls returned hits.
    """
    dsl = {
        "knn": {
            "field": "docEmbedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": max(k, 100),
        },
        "size": size,
    }
    return es.search(index=index_name, body=dsl)["hits"]["hits"]


# -------------------------
# Main
# -------------------------

def main():
    cfg = Settings()
    es = connect_es(cfg)

    tokenizer, model, device = load_hf_embedder(cfg)

    books = make_synthetic_books()

    # Embed documents
    doc_texts = [doc_to_embedding_text(b) for b in books]
    doc_embeddings = batch_embedding_hf(
        tokenizer,
        model,
        device,
        texts=doc_texts,
        batch_size=cfg.embedding_batch_size,
        max_length=cfg.embedding_max_length,
    )

    # Auto-detect dims from actual model output (and validate EMBED_DIMS if user set it)
    actual_dims = len(doc_embeddings[0]) if doc_embeddings else cfg.embedding_dims
    if actual_dims != cfg.embedding_dims:
        print(f"[WARN] EMBED_DIMS={cfg.embedding_dims} but model produced dims={actual_dims}. Using dims={actual_dims} for mapping.")

    mapping = build_index_mapping(actual_dims)
    recreate_index(es, cfg.index_name, mapping)

    # Bulk index
    ok = bulk_index_books(es, cfg.index_name, books, doc_embeddings)
    print(f"[OK] Indexed {ok}/{len(books)} synthetic books into '{cfg.index_name}'")

    # Keyword search
    user_query_kw = "Elasticsearch basics for search"
    hits_kw = keyword_search(es, cfg.index_name, user_query_kw, size=5)
    print(f"\n[Keyword Search] Query: {user_query_kw}")
    for h in hits_kw:
        src = h.get("_source") or {}
        print("-", src.get("title"), "| author:", src.get("author"), "| views:", src.get("views"), "| score:", h.get("_score"))

    # Facets
    buckets = facet_by_tags(es, cfg.index_name, size=10)
    print("\n[Facets] Top tags:")
    for b in buckets:
        print("-", b["key"], "=>", b["doc_count"])

    # Vector KNN search
    user_query_vec = "Explain inverted index and doc values in Lucene"
    query_text = build_query_text(user_query_vec)
    q_emb = batch_embedding_hf(
        tokenizer,
        model,
        device,
        texts=[query_text],
        batch_size=1,
        max_length=cfg.embedding_max_length,
    )[0]

    hits_vec = knn_search(es, cfg.index_name, q_emb, k=50, size=5)
    print(f"\n[KNN Search] Query: {user_query_vec}")
    for h in hits_vec:
        src = h.get("_source") or {}
        print("-", src.get("title"), "| tags:", src.get("tags"), "| views:", src.get("views"), "| score:", h.get("_score"))


if __name__ == "__main__":
    main()

'''
(llm)  ✘  🐍 llm  linghuang@Mac  ~/Git/Deep-Learning   local-ai-model ±  /Users/linghuang/miniconda3/envs/llm/bin/python /Users/linghuang/Git/Deep-Learning/AI-Modelin
g/19-Elastic/es.py
[OK] Connected: docker-cluster (v8.11.3) @ http://127.0.0.1:9200
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████| 310/310 [00:00<00:00, 3790.56it/s, Materializing param=norm.weight]
[EMB] Embedding 8 texts on mps using Qwen/Qwen3-Embedding-0.6B
[WARN] EMBED_DIMS=1536 but model produced dims=1024. Using dims=1024 for mapping.
[OK] Index 'books_demo' CREATED.
/Users/linghuang/Git/Deep-Learning/AI-Modeling/19-Elastic/es.py:198: DeprecationWarning: Passing transport options in the API method is deprecated. Use 'Elasticsearch.options()' instead.
  success, errors = bulk(es, actions, raise_on_error=False, request_timeout=120)
[OK] Indexed 8/8 synthetic books into 'books_demo'

[Keyword Search] Query: Elasticsearch basics for search
- Vector Databases vs Elasticsearch for RAG | author: Chris | views: 2210 | score: None
- Search Relevance Tuning with Query DSL | author: Emily | views: 1675 | score: None
- Hybrid Search: BM25 + Vector Retrieval | author: Mina | views: 1540 | score: None
- Scaling Elasticsearch Clusters: Shards, Replicas, and Nodes | author: Ava | views: 1310 | score: None
- Observability Logs in Elasticsearch: Index Lifecycle and Tiers | author: Sam | views: 1135 | score: None

[Facets] Top tags:
- elasticsearch => 2
- search => 2
- vector-search => 2
- aggregations => 1
- bm25 => 1
- boosting => 1
- clusters => 1
- ecommerce => 1
- facets => 1
- hot-warm-cold => 1
[EMB] Embedding 1 texts on mps using Qwen/Qwen3-Embedding-0.6B

[KNN Search] Query: Explain inverted index and doc values in Lucene
- Lucene Internals: Inverted Index and Doc Values | tags: ['lucene', 'search', 'indexing'] | views: 870 | score: 0.94278723
- Observability Logs in Elasticsearch: Index Lifecycle and Tiers | tags: ['observability', 'ilm', 'hot-warm-cold'] | views: 1135 | score: 0.71357036
- Elasticsearch Basics | tags: ['search', 'elasticsearch'] | views: 1024 | score: 0.70136124
- Hybrid Search: BM25 + Vector Retrieval | tags: ['hybrid-search', 'bm25', 'vector-search'] | views: 1540 | score: 0.6974207
- Search Relevance Tuning with Query DSL | tags: ['relevance', 'query-dsl', 'boosting'] | views: 1675 | score: 0.6927766
'''