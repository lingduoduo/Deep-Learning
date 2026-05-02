#!/usr/bin/env python3
"""
Generate CLIP embeddings for images and optionally write them to JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Imagehash.optimized_imagehash import IMAGE_EXTENSIONS, ImageEmbeddingModel, utc_timestamp


def collect_image_embeddings(
    image_dir: str,
    model_name: str = "openai/clip-vit-base-patch32",
    embedding_dims: int = 512,
) -> List[Dict[str, Any]]:
    embedder = ImageEmbeddingModel(
        model_name=model_name,
        embedding_dims=embedding_dims,
    )

    image_paths = sorted(
        path for path in Path(image_dir).rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    records: List[Dict[str, Any]] = []
    for path in image_paths:
        image = Image.open(path)
        vector = embedder.embed_image(str(path))
        records.append(
            {
                "filename": path.name,
                "path": str(path.resolve()),
                "category": path.parent.name,
                "metadata": {
                    "width": image.width,
                    "height": image.height,
                    "format": image.format,
                    "mode": image.mode,
                    "indexed_at": utc_timestamp(),
                },
                "image_vector": vector.tolist(),
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reusable CLIP embeddings for a directory of images."
    )
    parser.add_argument(
        "--image-dir",
        default="Imagehash/test_images",
        help="Directory containing images to embed.",
    )
    parser.add_argument(
        "--output",
        default="Imagehash/image_embeddings.json",
        help="Path to the JSON file that will store embeddings.",
    )
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--embedding-dims", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = collect_image_embeddings(
        image_dir=args.image_dir,
        model_name=args.model_name,
        embedding_dims=args.embedding_dims,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} embeddings to {output_path}")


if __name__ == "__main__":
    main()
