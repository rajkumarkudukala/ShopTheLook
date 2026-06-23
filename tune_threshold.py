"""
tune_threshold.py — Find optimal exact-match cosine similarity threshold.

Run AFTER offline_indexer.py completes, BEFORE evaluate.py.

Sweeps cosine similarity thresholds (0.75 to 0.99) on the 288 in-catalog
validation pairs and finds the threshold that maximizes F1 score for
the exact-match classification decision.

The tuned threshold is saved to artifacts/config.json:
  {"exact_match_threshold": 0.XX}

This file is loaded by src/matcher.py at startup.
"""

import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from src.fetcher import fetch_image
from src.detector import detect_fashion_crops
from src.embedder import embed_single_image

ARTIFACTS_DIR = "artifacts"
VAL_FILE = "data/validation.jsonl"


def tune():
    """Sweep thresholds and save optimal value to config.json."""
    # Load artifacts
    with open(os.path.join(ARTIFACTS_DIR, "catalog_ids.json")) as f:
        catalog_ids = json.load(f)
    catalog_id_set = set(catalog_ids)
    index = faiss.read_index(os.path.join(ARTIFACTS_DIR, "catalog.index"))

    # Load validation data
    with open(VAL_FILE) as f:
        val_records = [json.loads(l) for l in f if l.strip()]

    # Only tune on pairs where the product IS in the catalog (expected: 288)
    in_catalog_pairs = [r for r in val_records if r["product"] in catalog_id_set]
    print(f"Tuning on {len(in_catalog_pairs)} in-catalog validation pairs "
          f"(out of {len(val_records)} total)")

    # For each pair: embed scene image, find top-1 match, record score + correctness
    results = []
    skipped = 0

    for pair in tqdm(in_catalog_pairs, desc="Processing scenes"):
        scene_img = fetch_image(pair["scene"])
        if scene_img is None:
            skipped += 1
            continue

        # Use the detector to get garment crops, same as the real pipeline
        crops = detect_fashion_crops(scene_img)
        best_score = 0.0
        best_id = None
        for crop_info in crops:
            emb = embed_single_image(crop_info["crop"]).reshape(1, -1)
            scores, indices = index.search(emb, 1)
            score = float(scores[0][0])
            if score > best_score:
                best_score = score
                best_id = catalog_ids[indices[0][0]]

        is_correct = (best_id == pair["product"])
        results.append({"score": best_score, "correct": is_correct})

    print(f"\nProcessed: {len(results)}, Skipped (dead URL): {skipped}")

    if not results:
        print("No results to tune on! Check that scene images are accessible.")
        return

    # Sweep thresholds from 0.75 to 0.99 in steps of 0.01
    thresholds = np.arange(0.75, 1.00, 0.01)
    best_threshold = 0.92
    best_f1 = 0.0

    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 42)

    for thresh in thresholds:
        tp = sum(1 for r in results if r["score"] >= thresh and r["correct"])
        fp = sum(1 for r in results if r["score"] >= thresh and not r["correct"])
        fn = sum(1 for r in results if r["score"] < thresh and r["correct"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        print(f"{thresh:>10.2f} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thresh)

    # Save best threshold to config.json
    config = {"exact_match_threshold": round(best_threshold, 2)}
    config_path = os.path.join(ARTIFACTS_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*42}")
    print(f"Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    print(f"Saved to {config_path}")
    print(f"{'='*42}")


if __name__ == "__main__":
    tune()
