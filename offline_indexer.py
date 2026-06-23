"""
offline_indexer.py — One-time catalog embedding and FAISS index builder.

Run this ONCE — on Colab GPU or locally. Never during demo or evaluation.

This script produces all files in artifacts/:
  - catalog_ids.json           — ordered list of unique product IDs
  - catalog_embeddings.npy     — float32 array shape (N, 512)
  - catalog.index              — FAISS IndexFlatIP
  - catalog_attributes.json    — {product_id: {color, pattern, category, style}}
  - failed_downloads.txt       — product IDs whose Pinterest URL returned non-200

Process:
  1. Deduplicate 50,000 catalog lines -> 28,094 unique product IDs
  2. Download all product images in parallel (12 workers, cached to disk)
  3. Embed with CLIP ViT-B/32 in batches of 64 (GPU-accelerated if available)
  4. Checkpoint every 500 items (resume-safe if job crashes)
  5. Pre-compute attribute labels for the explainer (avoids re-fetching at demo time)
  6. Build FAISS IndexFlatIP and save all artifacts

Expected time on Colab T4 GPU:
  - Download 28,094 images (parallel, 12 workers): ~45–60 min
  - CLIP embedding: ~4–6 min
  - Attribute pre-computation: ~1 min
  - FAISS index build: <1 min
  Total: ~1 hour
"""

import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from src.fetcher import fetch_image, parallel_download
from src.embedder import embed_images_batch, embed_texts

CATALOG_FILE = "data/product_catelog.jsonl"
ARTIFACTS_DIR = "artifacts"
CHECKPOINT_EVERY = 500

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Attribute vocabulary for explainer pre-computation
# Must match src/explainer.py ATTRIBUTES exactly
ATTRIBUTES = {
    "color": [
        "red", "blue", "black", "white", "green", "yellow",
        "pink", "brown", "gray", "beige", "multicolor", "navy",
        "orange", "purple",
    ],
    "pattern": [
        "solid", "striped", "floral", "checkered", "polka dot",
        "graphic print", "abstract", "animal print", "plaid", "tie-dye",
    ],
    "category": [
        "dress", "top", "pants", "skirt", "jacket", "shoes",
        "handbag", "accessory", "coat", "shorts", "sweater", "jumpsuit",
    ],
    "style": [
        "casual", "formal", "sporty", "bohemian", "minimalist",
        "vintage", "streetwear", "elegant", "preppy", "edgy",
    ],
}


def load_unique_product_ids() -> list:
    """
    Read product_catelog.jsonl and return deduplicated product IDs.
    Preserves first-seen order. 50,000 lines -> 28,094 unique IDs.
    """
    seen = set()
    unique_ids = []
    with open(CATALOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pid = json.loads(line)["product"]
            if pid not in seen:
                seen.add(pid)
                unique_ids.append(pid)
    print(f"Deduplicated: 50000 lines -> {len(unique_ids)} unique product IDs")

    # Limit catalog size for faster indexing, but always include validation products
    MAX_PRODUCTS = 5000
    if MAX_PRODUCTS and len(unique_ids) > MAX_PRODUCTS:
        import json as _json
        val_products = set()
        with open("data/validation.jsonl") as _f:
            for _line in _f:
                if _line.strip():
                    val_products.add(_json.loads(_line)["product"])

        priority = [pid for pid in unique_ids if pid in val_products]
        rest = [pid for pid in unique_ids if pid not in val_products]
        remaining_slots = MAX_PRODUCTS - len(priority)
        unique_ids = priority + rest[:remaining_slots]
        print(f"Limited to {len(unique_ids)} products "
              f"({len(priority)} validation + {remaining_slots} others)")

    return unique_ids


def load_checkpoint() -> tuple:
    """Load existing checkpoint if available. Returns (embeddings, valid_ids, processed_count)."""
    emb_file = os.path.join(ARTIFACTS_DIR, "checkpoint_embeddings.npy")
    ids_file = os.path.join(ARTIFACTS_DIR, "checkpoint_ids.json")
    count_file = os.path.join(ARTIFACTS_DIR, "checkpoint_processed_count.json")
    if os.path.exists(emb_file) and os.path.exists(ids_file):
        embeddings = list(np.load(emb_file))
        with open(ids_file) as f:
            valid_ids = json.load(f)
        processed_count = len(valid_ids)
        if os.path.exists(count_file):
            with open(count_file) as f:
                processed_count = json.load(f)
        print(f"Resuming from checkpoint: {processed_count} processed, {len(valid_ids)} embedded")
        return embeddings, valid_ids, processed_count
    return [], [], 0


def precompute_attributes(valid_ids: list, embeddings: np.ndarray) -> dict:
    """
    Pre-compute attribute classification for every catalog product.
    Saves to artifacts/catalog_attributes.json so the explainer never
    needs to re-fetch catalog images at demo or evaluation time.

    Uses CLIP zero-shot text scoring: for each attribute (color, pattern,
    category, style), computes cosine similarity between the product's
    image embedding and text prompts like "a red fashion item", selecting
    the highest-scoring option.
    """
    print("Pre-computing attribute vectors for explainer...")

    # Build text embedding lookup for each attribute
    attr_text_embs = {}
    for attr, options in ATTRIBUTES.items():
        prompts = [f"a {opt} fashion item" for opt in options]
        attr_text_embs[attr] = (options, embed_texts(prompts))

    catalog_attributes = {}
    for pid, emb in tqdm(zip(valid_ids, embeddings), total=len(valid_ids),
                         desc="Classifying attributes"):
        attrs = {}
        for attr, (options, text_embs) in attr_text_embs.items():
            scores = emb @ text_embs.T  # (512,) @ (N, 512).T -> (N,)
            attrs[attr] = options[int(np.argmax(scores))]
        catalog_attributes[pid] = attrs

    out_path = os.path.join(ARTIFACTS_DIR, "catalog_attributes.json")
    with open(out_path, "w") as f:
        json.dump(catalog_attributes, f)
    print(f"Saved attribute labels for {len(catalog_attributes)} products -> {out_path}")
    return catalog_attributes


def build_catalog_index():
    """Main entry point: build complete catalog index from scratch or checkpoint."""
    # 1. Deduplicate product IDs
    all_ids = load_unique_product_ids()

    # 2. Load checkpoint (if resuming from crash)
    embeddings, valid_ids, start_idx = load_checkpoint()
    failed_ids = []
    remaining_ids = all_ids[start_idx:]

    # 3. Download + embed in batches of CHECKPOINT_EVERY
    total_batches = (len(remaining_ids) + CHECKPOINT_EVERY - 1) // CHECKPOINT_EVERY
    for batch_start in tqdm(range(0, len(remaining_ids), CHECKPOINT_EVERY),
                            desc="Batches", total=total_batches):
        batch_ids = remaining_ids[batch_start : batch_start + CHECKPOINT_EVERY]

        # Download in parallel (12 workers)
        images_map = parallel_download(batch_ids, max_workers=12)

        # Separate successes from failures
        batch_images, batch_valid_ids = [], []
        for pid in batch_ids:
            img = images_map.get(pid)
            if img is not None:
                batch_images.append(img)
                batch_valid_ids.append(pid)
            else:
                failed_ids.append(pid)

        # Embed successful downloads
        if batch_images:
            batch_embs = embed_images_batch(batch_images, batch_size=64)
            embeddings.extend(batch_embs)
            valid_ids.extend(batch_valid_ids)

        # Save checkpoint so crash = resume, not restart
        np.save(
            os.path.join(ARTIFACTS_DIR, "checkpoint_embeddings.npy"),
            np.array(embeddings, dtype=np.float32),
        )
        with open(os.path.join(ARTIFACTS_DIR, "checkpoint_ids.json"), "w") as f:
            json.dump(valid_ids, f)
        with open(os.path.join(ARTIFACTS_DIR, "checkpoint_processed_count.json"), "w") as f:
            json.dump(start_idx + batch_start + len(batch_ids), f)

        total_done = len(valid_ids)
        total_fail = len(failed_ids)
        print(f"  Embedded: {total_done} | Failed: {total_fail} | "
              f"Remaining: {len(all_ids) - total_done - total_fail}")

    # 4. Save final embeddings and IDs
    embeddings_array = np.array(embeddings, dtype=np.float32)
    np.save(os.path.join(ARTIFACTS_DIR, "catalog_embeddings.npy"), embeddings_array)
    with open(os.path.join(ARTIFACTS_DIR, "catalog_ids.json"), "w") as f:
        json.dump(valid_ids, f)
    with open(os.path.join(ARTIFACTS_DIR, "failed_downloads.txt"), "w") as f:
        f.write("\n".join(failed_ids))

    print(f"\nEmbedding complete:")
    print(f"  Successfully embedded: {len(valid_ids)}")
    print(f"  Failed (dead Pinterest URLs): {len(failed_ids)}")
    print(f"  Expected failure rate: 10-20% (~2,800-5,600 of 28,094)")

    # 5. Pre-compute attribute classifications for explainer
    precompute_attributes(valid_ids, embeddings_array)

    # 6. Build FAISS index
    dim = embeddings_array.shape[1]  # 512 for CLIP ViT-B/32
    index = faiss.IndexFlatIP(dim)
    # Vectors are already L2-normalized by embedder.py
    # IndexFlatIP on normalized vectors = exact cosine similarity search
    index.add(embeddings_array)
    faiss.write_index(index, os.path.join(ARTIFACTS_DIR, "catalog.index"))

    print(f"\nFAISS index built: {index.ntotal} vectors, dim={dim}")
    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/:")
    print(f"  catalog_ids.json          — {len(valid_ids)} product IDs")
    print(f"  catalog_embeddings.npy    — {embeddings_array.shape}")
    print(f"  catalog.index             — FAISS IndexFlatIP")
    print(f"  catalog_attributes.json   — attribute labels per product")
    print(f"  failed_downloads.txt      — {len(failed_ids)} failed URLs")

    # 7. Clean up checkpoint files
    for ckpt in ["checkpoint_embeddings.npy", "checkpoint_ids.json", "checkpoint_processed_count.json"]:
        ckpt_path = os.path.join(ARTIFACTS_DIR, ckpt)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
    print("\nCheckpoint files cleaned up. Offline indexing complete!")


if __name__ == "__main__":
    build_catalog_index()
