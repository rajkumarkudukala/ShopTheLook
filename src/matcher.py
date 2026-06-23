"""
src/matcher.py — FAISS similarity search + exact match threshold logic.

Loads pre-built artifacts at startup:
- catalog_ids.json — ordered list of product IDs (same order as embeddings)
- catalog.index — FAISS IndexFlatIP index
- config.json — tuned exact-match threshold (from tune_threshold.py)

All searches are cosine similarity (IndexFlatIP on L2-normalized vectors).
"""

import json
import os
import numpy as np
import faiss

ARTIFACTS_DIR = "artifacts"


def _load_artifacts():
    """Load FAISS index, catalog IDs, and config at module level."""
    ids_path = os.path.join(ARTIFACTS_DIR, "catalog_ids.json")
    index_path = os.path.join(ARTIFACTS_DIR, "catalog.index")
    config_path = os.path.join(ARTIFACTS_DIR, "config.json")

    if not os.path.exists(ids_path) or not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Artifacts not found in {ARTIFACTS_DIR}/. "
            "Run offline_indexer.py first to generate catalog_ids.json and catalog.index."
        )

    with open(ids_path) as f:
        catalog_ids = json.load(f)
    index = faiss.read_index(index_path)

    # Load threshold from tune_threshold.py output
    threshold = 0.92  # default fallback
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        threshold = config.get("exact_match_threshold", 0.92)
    else:
        print("Warning: artifacts/config.json not found. Using default threshold 0.92.")
        print("Run tune_threshold.py to find the optimal threshold for this dataset.")

    return catalog_ids, index, threshold


# Load all artifacts at startup — never reload during a request
catalog_ids, index, EXACT_MATCH_THRESHOLD = _load_artifacts()


def match_crop(crop_embedding: np.ndarray, top_k: int = 5) -> list:
    """
    Find top-K catalog matches for an L2-normalized crop embedding.

    Args:
        crop_embedding: shape (512,) float32, L2-normalized
        top_k: number of results to return

    Returns:
        List of dicts, each containing:
        - "product_id": str — catalog product signature hash
        - "similarity_score": float — cosine similarity in [0, 1]
        - "is_exact_match": bool — True if score >= EXACT_MATCH_THRESHOLD
        - "rank": int — 1-indexed rank position
    """
    query = crop_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue
        results.append({
            "product_id": catalog_ids[idx],
            "similarity_score": float(score),
            "is_exact_match": float(score) >= EXACT_MATCH_THRESHOLD,
            "rank": rank,
        })
    return results
