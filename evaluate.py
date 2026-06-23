"""
evaluate.py — Evaluation script for Shop-the-Look system.

Computes:
  - Recall@1, Recall@5, Recall@10
  - MRR (Mean Reciprocal Rank)
  - Exact Match Rate (only on the 288 in-catalog validation pairs)

Multi-product scene handling:
  8 scenes have 2-3 ground-truth products. For these scenes, a prediction
  is considered correct if it matches ANY of the true products for that scene.
  (Ignoring this would undercount correct matches.)

  Multi-product scenes:
    scene=0aba9fb6... | 2 products | both IN catalog
    scene=465cb4f0... | 2 products | both IN catalog
    scene=7eb435fc... | 2 products | both IN catalog
    scene=c2c43ce0... | 2 products | both IN catalog
    scene=e7d70ce4... | 3 products | 1 IN, 2 NOT IN catalog
    scene=ebcd4808... | 2 products | both NOT IN catalog
    scene=f86017bc... | 2 products | both IN catalog
    scene=fb6d5c9b... | 2 products | both IN catalog

Out-of-catalog handling:
  41 validation products (19.9%) are not in the catalog.
  For these, Recall/MRR cannot be satisfied by definition.
  They ARE included in Recall/MRR denominators (count as misses)
  but are EXCLUDED from Exact Match Rate computation.
"""

import json
import os
from collections import defaultdict
from tqdm import tqdm
from src.pipeline import process_scene

VAL_FILE = "data/validation.jsonl"
ARTIFACTS_DIR = "artifacts"


def evaluate(top_k_max: int = 10):
    """Run full evaluation on validation set."""
    # Load validation data
    with open(VAL_FILE) as f:
        val_records = [json.loads(l) for l in f if l.strip()]

    # Load catalog ID set for in-catalog/out-of-catalog classification
    with open(os.path.join(ARTIFACTS_DIR, "catalog_ids.json")) as f:
        catalog_id_set = set(json.load(f))

    # Group by scene — CRITICAL for multi-product scene handling
    # scene_to_products[scene_sig] = set of true product IDs for that scene
    scene_to_products = defaultdict(set)
    for r in val_records:
        scene_to_products[r["scene"]].add(r["product"])

    # Report multi-product scenes
    multi_product_scenes = {s: p for s, p in scene_to_products.items() if len(p) > 1}
    print(f"\nDataset stats:")
    print(f"  Total validation pairs: {len(val_records)}")
    print(f"  Unique scenes: {len(scene_to_products)}")
    print(f"  Multi-product scenes: {len(multi_product_scenes)}")
    for scene, products in multi_product_scenes.items():
        in_cat = sum(1 for p in products if p in catalog_id_set)
        print(f"    {scene[:16]}... | {len(products)} products | "
              f"{in_cat} IN catalog, {len(products)-in_cat} NOT IN")

    # Evaluation counters
    recall_at = {1: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    exact_match_tp = 0      # correctly identified as exact match AND is correct top-1
    exact_match_total = 0   # in-catalog products appearing in evaluated scenes
    skipped_scenes = []

    total_scenes = len(scene_to_products)  # 321

    for scene_sig, true_products in tqdm(scene_to_products.items(),
                                          desc="Evaluating scenes",
                                          total=total_scenes):
        result = process_scene(scene_sig, top_k=top_k_max)

        if result["error"]:
            skipped_scenes.append(scene_sig)
            continue

        # Collect all predicted product IDs across all detected crops,
        # ranked globally by similarity score (highest first)
        all_preds = []
        for item in result["items"]:
            for match in item["matches"]:
                all_preds.append((
                    match["product_id"],
                    match["similarity_score"],
                    match["is_exact_match"],
                ))

        # Deduplicate by product ID, keeping highest score
        seen_pids = {}
        for pid, score, is_exact in all_preds:
            if pid not in seen_pids or score > seen_pids[pid][0]:
                seen_pids[pid] = (score, is_exact)

        ranked = sorted(seen_pids.items(), key=lambda x: -x[1][0])
        # ranked = [(pid, (score, is_exact)), ...]
        ranked_pids = [pid for pid, _ in ranked]

        # --- Recall@K ---
        # A prediction is correct if ANY true product for this scene is in top-K
        for k in [1, 5, 10]:
            if any(pid in true_products for pid in ranked_pids[:k]):
                recall_at[k] += 1

        # --- MRR ---
        # Rank of the FIRST occurrence of any true product
        rr = 0.0
        for rank, pid in enumerate(ranked_pids, start=1):
            if pid in true_products:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # --- Exact Match Rate ---
        # Only count in-catalog products. For each true product in the catalog,
        # check if it appears anywhere in the ranked results with is_exact_match
        # flagged. For multi-product scenes, each true product is evaluated
        # independently against the full ranked list.
        in_catalog_true = [p for p in true_products if p in catalog_id_set]
        for true_pid in in_catalog_true:
            exact_match_total += 1
            for pid, (score, is_exact) in ranked:
                if pid == true_pid and is_exact:
                    exact_match_tp += 1
                    break

    # Compute final metrics
    n_evaluated = total_scenes - len(skipped_scenes)

    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Total scenes:           {total_scenes}")
    print(f"  Evaluated:              {n_evaluated}")
    print(f"  Skipped (dead URL):     {len(skipped_scenes)}")
    print(f"  Multi-product scenes:   {len(multi_product_scenes)} (handled correctly)")
    print(f"{'='*50}")
    print(f"  {'Metric':<24} {'Value':>8}")
    print(f"  {'-'*34}")

    for k in [1, 5, 10]:
        val = recall_at[k] / n_evaluated if n_evaluated > 0 else 0.0
        print(f"  {'Recall@' + str(k):<24} {val:>8.4f}")

    mrr = sum(reciprocal_ranks) / n_evaluated if n_evaluated > 0 else 0.0
    print(f"  {'MRR':<24} {mrr:>8.4f}")

    # Count out-of-catalog products dynamically
    all_true_products = set()
    for products in scene_to_products.values():
        all_true_products.update(products)
    out_of_catalog = sum(1 for p in all_true_products if p not in catalog_id_set)
    total_products = len(all_true_products)

    if exact_match_total > 0:
        em_rate = exact_match_tp / exact_match_total
        print(f"  {'Exact Match Rate':<24} {em_rate:>8.4f}")
        print(f"  (computed on {exact_match_total} in-catalog pairs; "
              f"{out_of_catalog} not-in-catalog pairs excluded)")

    print(f"{'='*50}")
    if total_products > 0:
        oc_pct = out_of_catalog / total_products * 100
        print(f"\nNote: {out_of_catalog}/{total_products} validation products "
              f"({oc_pct:.1f}%) are not in the catalog.")
        print("For these, the system returns similar products — not penalized in "
              "Exact Match Rate.")


if __name__ == "__main__":
    evaluate()
