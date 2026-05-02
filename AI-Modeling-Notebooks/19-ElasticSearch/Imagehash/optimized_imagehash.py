#!/usr/bin/env python3
"""
Image similarity recommendations with CLIP embeddings and Elasticsearch.

Typical usage:

1. Index the sample images:
   python Imagehash/optimized_imagehash.py index --image-dir Imagehash/test_images

2. Recommend images similar to a query image:
   python Imagehash/optimized_imagehash.py recommend --query-image Imagehash/test_images/1.png

3. Recommend images similar to one already indexed by filename:
   python Imagehash/optimized_imagehash.py recommend --image-id 1.png
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ImageDocument:
    filename: str
    path: str
    category: str
    description: str
    metadata: Dict[str, Any]
    image_vector: List[float]

    def to_es_source(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "path": self.path,
            "category": self.category,
            "description": self.description,
            "metadata": self.metadata,
            "image_vector": self.image_vector,
        }


class ImageEmbeddingModel:
    """Lazy CLIP wrapper so simple operations do not require the full stack."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        embedding_dims: int = 512,
    ) -> None:
        self.model_name = model_name
        self.embedding_dims = embedding_dims
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "CLIP dependencies are unavailable. Install working `torch` and "
                "`transformers` packages before generating embeddings."
            ) from exc

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)

    def _normalize_dims(self, vector: np.ndarray) -> np.ndarray:
        if len(vector) > self.embedding_dims:
            return vector[: self.embedding_dims]
        if len(vector) < self.embedding_dims:
            padding = np.zeros(self.embedding_dims - len(vector), dtype=np.float32)
            return np.concatenate([vector, padding])
        return vector

    def _extract_tensor(self, output):
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0, :]
        return output

    def embed_image(self, image_path: str) -> np.ndarray:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)

        with self._torch.no_grad():
            features = self._model.get_image_features(**inputs)
            features = self._extract_tensor(features)
            features = features / features.norm(dim=-1, keepdim=True)

        vector = features.cpu().numpy().astype(np.float32).flatten()
        return self._normalize_dims(vector)

    def embed_text(self, text: str) -> np.ndarray:
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._device is not None

        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(
            self._device
        )

        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = self._extract_tensor(features)
            features = features / features.norm(dim=-1, keepdim=True)

        vector = features.cpu().numpy().astype(np.float32).flatten()
        return self._normalize_dims(vector)


class ImageSimilarityElasticsearch:
    def __init__(
        self,
        host: str = "http://localhost:9200",
        index_name: str = "image_similarity",
        embedding_dims: int = 512,
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        self.host = host
        self.index_name = index_name
        self.embedding_dims = embedding_dims
        self.model_name = model_name
        self.embedding_model = ImageEmbeddingModel(
            model_name=model_name,
            embedding_dims=embedding_dims,
        )
        self._es = None

    @property
    def es(self):
        if self._es is None:
            try:
                from elasticsearch import Elasticsearch
            except Exception as exc:  # pragma: no cover - depends on local env
                raise RuntimeError(
                    "Elasticsearch client is unavailable. Install the "
                    "`elasticsearch` package before indexing or searching."
                ) from exc

            self._es = Elasticsearch(hosts=[self.host])
        return self._es

    def ping(self) -> bool:
        return bool(self.es.ping())

    def create_index(self, recreate: bool = False) -> None:
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index.refresh_interval": "1s",
            },
            "mappings": {
                "properties": {
                    "filename": {"type": "keyword"},
                    "path": {"type": "keyword"},
                    "category": {"type": "keyword"},
                    "description": {"type": "text"},
                    "metadata": {
                        "properties": {
                            "width": {"type": "integer"},
                            "height": {"type": "integer"},
                            "format": {"type": "keyword"},
                            "mode": {"type": "keyword"},
                            "size_bytes": {"type": "long"},
                            "indexed_at": {"type": "date"},
                        }
                    },
                    "image_vector": {
                        "type": "dense_vector",
                        "dims": self.embedding_dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        }

        if recreate and self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=mapping)

    def iter_image_paths(self, image_directory: str) -> Iterable[Path]:
        for root, _, files in os.walk(image_directory):
            for name in sorted(files):
                path = Path(root) / name
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield path

    def build_document(self, image_path: str) -> ImageDocument:
        path = Path(image_path).resolve()
        image = Image.open(path)
        vector = self.embedding_model.embed_image(str(path))

        metadata = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "size_bytes": path.stat().st_size,
            "indexed_at": utc_timestamp(),
        }

        category = path.parent.name
        description = (
            f"Sample image {path.name} from {category} with size "
            f"{image.width}x{image.height}"
        )

        return ImageDocument(
            filename=path.name,
            path=str(path),
            category=category,
            description=description,
            metadata=metadata,
            image_vector=vector.tolist(),
        )

    def index_images(self, image_directory: str, recreate: bool = False) -> int:
        self.create_index(recreate=recreate)

        try:
            from elasticsearch.helpers import bulk
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "Elasticsearch bulk helper is unavailable. Install the "
                "`elasticsearch` package before indexing."
            ) from exc

        actions = []
        count = 0

        for image_path in self.iter_image_paths(image_directory):
            doc = self.build_document(str(image_path))
            actions.append(
                {
                    "_index": self.index_name,
                    "_id": doc.filename,
                    "_source": doc.to_es_source(),
                }
            )
            count += 1

        if not actions:
            return 0

        bulk(self.es, actions, refresh="wait_for")
        return count

    def get_by_id(self, image_id: str) -> Dict[str, Any]:
        response = self.es.get(index=self.index_name, id=image_id)
        return response["_source"]

    def similarity_search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        exclude_filename: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []
        if exclude_filename:
            filters.append({"bool": {"must_not": {"term": {"filename": exclude_filename}}}})
        if category:
            filters.append({"term": {"category": category}})

        query: Dict[str, Any] = {
            "knn": {
                "field": "image_vector",
                "query_vector": query_vector.tolist(),
                "k": k,
                "num_candidates": max(k * 3, 10),
            },
            "_source": ["filename", "path", "category", "description", "metadata"],
        }

        if filters:
            query["knn"]["filter"] = filters

        response = self.es.search(index=self.index_name, body=query)
        return response["hits"]["hits"]

    def recommend_from_image(
        self,
        image_path: str,
        k: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.embed_image(image_path)
        return self.similarity_search(
            query_vector=query_vector,
            k=k,
            exclude_filename=Path(image_path).name,
            category=category,
        )

    def recommend_from_indexed_image(
        self,
        image_id: str,
        k: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        source = self.get_by_id(image_id)
        vector = np.array(source["image_vector"], dtype=np.float32)
        return self.similarity_search(
            query_vector=vector,
            k=k,
            exclude_filename=source["filename"],
            category=category,
        )

    def text_to_image_search(
        self,
        text_query: str,
        k: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.embed_text(text_query)
        return self.similarity_search(
            query_vector=query_vector,
            k=k,
            category=category,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create CLIP embeddings and use Elasticsearch for image similarity."
    )
    parser.add_argument("--host", default="http://localhost:9200")
    parser.add_argument("--index-name", default="image_similarity")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--embedding-dims", type=int, default=512)

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index images into Elasticsearch")
    index_parser.add_argument(
        "--image-dir",
        default="Imagehash/test_images",
        help="Directory of source images to embed and index.",
    )
    index_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the index before indexing.",
    )

    recommend_parser = subparsers.add_parser(
        "recommend",
        help="Recommend similar images from a query image or indexed image id.",
    )
    recommend_parser.add_argument("--query-image", help="Path to the query image.")
    recommend_parser.add_argument("--image-id", help="Indexed image filename to use as query.")
    recommend_parser.add_argument("--k", type=int, default=5)
    recommend_parser.add_argument("--category")

    text_parser = subparsers.add_parser(
        "text-search",
        help="Search indexed images using a text prompt in CLIP space.",
    )
    text_parser.add_argument("text_query")
    text_parser.add_argument("--k", type=int, default=5)
    text_parser.add_argument("--category")

    return parser


def format_hits(hits: List[Dict[str, Any]]) -> str:
    rows = []
    for hit in hits:
        source = hit["_source"]
        rows.append(
            {
                "filename": source["filename"],
                "score": round(hit.get("_score", 0.0), 4),
                "category": source.get("category"),
                "path": source.get("path"),
            }
        )
    return json.dumps(rows, indent=2)


def main() -> None:
    args = build_parser().parse_args()
    image_es = ImageSimilarityElasticsearch(
        host=args.host,
        index_name=args.index_name,
        embedding_dims=args.embedding_dims,
        model_name=args.model_name,
    )

    if not image_es.ping():
        raise RuntimeError(
            f"Unable to connect to Elasticsearch at {args.host}. "
            "Start Elasticsearch before running this command."
        )

    if args.command == "index":
        count = image_es.index_images(args.image_dir, recreate=args.recreate)
        print(f"Indexed {count} images into '{args.index_name}'.")
        return

    if args.command == "recommend":
        if not args.query_image and not args.image_id:
            raise ValueError("Pass either --query-image or --image-id.")
        if args.query_image and args.image_id:
            raise ValueError("Use only one of --query-image or --image-id.")

        if args.query_image:
            hits = image_es.recommend_from_image(
                image_path=args.query_image,
                k=args.k,
                category=args.category,
            )
        else:
            hits = image_es.recommend_from_indexed_image(
                image_id=args.image_id,
                k=args.k,
                category=args.category,
            )
        print(format_hits(hits))
        return

    if args.command == "text-search":
        hits = image_es.text_to_image_search(
            text_query=args.text_query,
            k=args.k,
            category=args.category,
        )
        print(format_hits(hits))
        return


if __name__ == "__main__":
    main()
