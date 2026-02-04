#!/usr/bin/env python3
"""
elasticsearch_vector_demo_books.py

This version:
- DROPS external data loading (no doc1.json / JSONL needed).
- Uses a small in-script synthetic dataset like:
  {
    "title": "Elasticsearch Basics",
    "author": "Emily",
    "published_date": "2024-01-01",
    "tags": ["search", "elasticsearch"],
    "views": 1024
  }
- Updates the ES mapping accordingly.
- Updates keyword query + vector KNN query to match the new fields.

Embedding model (Option 3):
- Auto-downloads from Hugging Face Hub: Qwen/Qwen3-Embedding-0.6B
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch


# -------------------------
# Configuration
# -------------------------

@dataclass
class Settings:
    # Elasticsearch
    es_host: str = os.getenv("ES_HOST", "http://localhost:9200")
    es_user: Optional[str] = os.getenv("ES_USER")  # optional if security disabled
    es_password: Optional[str] = os.getenv("ES_PASSWORD")
    verify_certs: bool = os.getenv("ES_VERIFY_CERTS", "false").lower() == "true"
    index_name: str = os.getenv("ES_INDEX", "books_demo")

    # HF embedding model (auto-download)
    hf_model_id: str = os.getenv("HF_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
    hf_token: Optional[str] = os.getenv("HF_TOKEN")

    # Dense vector dims must match embedding output dims
    embedding_dims: int = int(os.getenv("EMBED_DIMS", "1536"))

    # Embedding settings
    embedding_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "16"))
    embedding_max_length: int = int(os.getenv("EMBED_MAX_LENGTH", "512"))  # titles/tags are short
    device: str = os.getenv("DEVICE", "auto")  # auto | cpu | cuda | mps


# -------------------------
# Synthetic Dataset
# -------------------------

def make_synthetic_books() -> List[Dict[str, Any]]:
    """
    Small synthetic dataset suitable for demoing:
    - keyword search (match/match_phrase)
    - sort by numeric fields
    - terms aggregation on tags
    - vector search using embeddings generated from title+tags text
    """
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
    """
    Create the text we embed into docEmbedding.
    For small book metadata, title + tags is a simple, effective choice.
    """
    title = doc.get("title", "")
    tags = doc.get("tags", [])
    tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
    author = doc.get("author", "")
    return f"Title: {title}\nAuthor: {author}\nTags: {tags_str}"


def build_query_text(user_query: str) -> str:
    """
    Instruct-style wrapper to match the notebook's idea, but tailored to this dataset.
    """
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
                "author": {"type": "keyword"},  # exact match + aggregation
                "published_date": {"type": "date"},
                "tags": {"type": "keyword"},  # facet-friendly
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
    if cfg.es_user and cfg.es_password:
        es = Elasticsearch(
            hosts=[cfg.es_host],
            basic_auth=(cfg.es_user, cfg.es_password),
            verify_certs=cfg.verify_certs,
        )
    else:
        es = Elasticsearch(hosts=[cfg.es_host], verify_certs=cfg.verify_certs)

    if not es.ping():
        raise RuntimeError(f"Elasticsearch not reachable at {cfg.es_host}")
    return es


def recreate_index(es: Elasticsearch, index_name: str, index_mapping: Dict[str, Any]) -> None:
    """
    For a demo: delete + recreate so mapping is always correct.
    If you prefer "create if not exists", change this behavior.
    """
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"[OK] Index '{index_name}' DELETED.")
    es.indices.create(index=index_name, body=index_mapping)
    print(f"[OK] Index '{index_name}' CREATED.")


def add_doc(es: Elasticsearch, index_name: str, document: Dict[str, Any], doc_id: str) -> bool:
    try:
        es.index(index=index_name, id=doc_id, document=document)
        return True
    except Exception as e:
        print(f"[ERR] Index '{index_name}', Doc Id '{doc_id}' ADD failed: {e}")
        return False


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
    Notebook-style pooling: last non-padding token.
    """
    import torch
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lens]


def l2_normalize(x, eps: float = 1e-12):
    import torch
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps)


def load_hf_embedder(cfg: Settings):
    import torch
    from transformers import AutoTokenizer, AutoModel

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


def batch_embedding_hf(tokenizer, model, device: str, texts: List[str], batch_size: int, max_length: int) -> List[List[float]]:
    import torch

    results: List[List[float]] = []
    n = len(texts)
    print(f"[EMB] Embedding {n} texts on {device} using {getattr(model, 'name_or_path', 'hf-model')}")

    for i in range(math.ceil(n / batch_size)):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch_texts = texts[start:end]

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
        emb = l2_normalize(emb).detach().to("cpu")
        results.extend(emb.numpy().tolist())

    return results


# -------------------------
# Search Queries (Updated)
# -------------------------

def keyword_search(es: Elasticsearch, index_name: str, user_query: str, size: int = 5):
    """
    Updated keyword search for the new schema:
    - searches title (text) and tags (keyword via terms query fallback)
    For simplicity, we do multi_match on title + author.
    """
    dsl = {
        "query": {
            "multi_match": {
                "query": user_query,
                "fields": ["title^2", "author"],
                "type": "best_fields",
            }
        },
        "sort": [{"views": "desc"}],  # example: sort by popularity
        "size": size,
    }
    return es.search(index=index_name, body=dsl)["hits"]["hits"]


def facet_by_tags(es: Elasticsearch, index_name: str, size: int = 10):
    """
    Simple facet example: top tags by count.
    """
    dsl = {
        "size": 0,
        "aggs": {
            "top_tags": {
                "terms": {"field": "tags", "size": size}
            }
        },
    }
    return es.search(index=index_name, body=dsl)["aggregations"]["top_tags"]["buckets"]


def knn_search(es: Elasticsearch, index_name: str, query_vector: List[float], size: int = 5):
    dsl = {
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "field": "docEmbedding",
                            "query_vector": query_vector,
                        }
                    }
                ],
                "minimum_should_match": 0,
            }
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

    mapping = build_index_mapping(cfg.embedding_dims)
    recreate_index(es, cfg.index_name, mapping)

    books = make_synthetic_books()

    tokenizer, model, device = load_hf_embedder(cfg)

    # Embed documents (title + author + tags)
    doc_texts = [doc_to_embedding_text(b) for b in books]
    doc_embeddings = batch_embedding_hf(
        tokenizer, model, device,
        texts=doc_texts,
        batch_size=cfg.embedding_batch_size,
        max_length=cfg.embedding_max_length,
    )

    # Index docs
    ok = 0
    for i, b in enumerate(books):
        doc = dict(b)
        doc["docEmbedding"] = doc_embeddings[i]
        if add_doc(es, cfg.index_name, doc, doc_id=str(i)):
            ok += 1

    print(f"[OK] Indexed {ok}/{len(books)} synthetic books into '{cfg.index_name}'")

    # ---------- Updated user queries ----------
    # Keyword query now targets title/author
    user_query_kw = "Elasticsearch basics for search"
    hits_kw = keyword_search(es, cfg.index_name, user_query_kw, size=5)
    print(f"\n[Keyword Search] Query: {user_query_kw}")
    for h in hits_kw:
        src = h.get("_source") or {}
        print("-", src.get("title"), "| author:", src.get("author"), "| views:", src.get("views"), "| score:", h.get("_score"))

    # Facets (aggregations) on tags
    buckets = facet_by_tags(es, cfg.index_name, size=10)
    print("\n[Facets] Top tags:")
    for b in buckets:
        print("-", b["key"], "=>", b["doc_count"])

    # KNN query now embeds a library-style query and searches docEmbedding
    user_query_vec = "Explain inverted index and doc values in Lucene"
    query_text = build_query_text(user_query_vec)
    q_emb = batch_embedding_hf(
        tokenizer, model, device,
        texts=[query_text],
        batch_size=1,
        max_length=cfg.embedding_max_length,
    )[0]

    hits_vec = knn_search(es, cfg.index_name, q_emb, size=5)
    print(f"\n[KNN Search] Query: {user_query_vec}")
    for h in hits_vec:
        src = h.get("_source") or {}
        print("-", src.get("title"), "| tags:", src.get("tags"), "| views:", src.get("views"), "| score:", h.get("_score"))


if __name__ == "__main__":
    main()
