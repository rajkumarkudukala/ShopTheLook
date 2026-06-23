# Shop-the-Look: Visual Product Discovery System — Technical Report

---

## 1. Overall Approach

The Shop-the-Look system is framed as a **cross-modal visual retrieval** problem: given a lifestyle scene image containing one or more fashion items, identify and retrieve the exact matching products from a large catalog. The system bridges two fundamentally different visual domains — curated catalog product shots (clean backgrounds, single items) and unstructured lifestyle photography (multiple people, complex backgrounds, partial occlusions) — using a shared embedding space learned by CLIP.

### Two-Phase Architecture

The system is split into two strictly separated phases:

| Phase | When | Where | Duration | Purpose |
|-------|------|-------|----------|---------|
| **Offline Indexing** | One-time, before deployment | Colab T4 GPU | ~1 hour | Embed all 28,094 catalog products and build the FAISS search index |
| **Online Query** | Per user request, at demo time | CPU or GPU | <2 seconds per scene | Detect, embed, search, and explain matches in real time |

This separation is essential because embedding 28,094 product images through CLIP is computationally expensive (45–60 minutes for download alone), but searching the resulting index is near-instantaneous (<10 ms per FAISS query). By front-loading the heavy computation into a one-time offline job, the system achieves real-time latency at demo time without requiring persistent GPU infrastructure.

### Offline Phase (`offline_indexer.py`)

1. **Deduplication** — The raw catalog (`product_catelog.jsonl`) contains ~50,000 lines, but many product IDs are repeated (some up to 130 times). Deduplication reduces this to **28,094 unique products**, preserving first-seen order.
2. **Parallel Image Download** — Product images are fetched from Pinterest CDN using 12 concurrent workers, with disk caching, retry with exponential backoff, and 404-aware early exit for dead URLs.
3. **CLIP Embedding** — Each successfully downloaded image is embedded into a 512-dimensional L2-normalized vector using CLIP ViT-B/32, processed in batches of 64.
4. **Checkpointing** — Embeddings and valid IDs are saved every 500 items, enabling crash recovery without restarting from scratch.
5. **Attribute Pre-computation** — CLIP zero-shot text scoring classifies every catalog product across four attribute dimensions (color, pattern, category, style), saving results to `catalog_attributes.json`. This eliminates the need to re-fetch catalog images at runtime.
6. **FAISS Index Construction** — A `FAISS IndexFlatIP` is built over the L2-normalized embeddings, enabling exact cosine similarity search.

### Online Phase (`src/pipeline.py`)

1. **Fetch** the scene image from its Pinterest signature hash.
2. **Detect** fashion-relevant crops using YOLOS-tiny object detection.
3. **Embed** each detected crop with CLIP ViT-B/32.
4. **Search** the FAISS index for the top-K nearest catalog products.
5. **Explain** each match using attribute-level comparison between the scene crop and the pre-computed catalog product attributes.

---

## 2. Model Architecture and Methodology

### CLIP ViT-B/32 — Dual-Encoder Foundation

The system uses **OpenAI's CLIP (Contrastive Language–Image Pretraining)** with a ViT-B/32 vision backbone as its core embedding model. CLIP is a dual-encoder architecture consisting of:

- **Vision Encoder** — A Vision Transformer (ViT-B/32) that splits images into 32×32 patches and processes them through 12 transformer layers, producing a single 512-dimensional image embedding.
- **Text Encoder** — A 12-layer transformer that encodes text sequences into the same 512-dimensional space.

CLIP was contrastively pretrained on **400 million image-text pairs** scraped from the internet, learning to align images and their natural-language descriptions in a shared embedding space. The training objective maximizes cosine similarity between matched image-text pairs while minimizing it for mismatched pairs.

**Key properties exploited by this system:**

| Property | Relevance to Fashion Retrieval |
|----------|-------------------------------|
| 512-d L2-normalized embeddings | Cosine similarity computable via inner product; compact and efficient for FAISS |
| Zero-shot visual understanding | Recognizes colors, patterns, styles, and categories without task-specific training |
| Cross-domain generalization | Bridges the visual gap between clean catalog product shots and in-the-wild lifestyle photos |
| Text–image alignment | Enables attribute classification via zero-shot text scoring (e.g., "a red fashion item") |

### Why CLIP Works for Fashion

CLIP's web-scale pretraining corpus includes extensive fashion-related content — product listings, style blogs, social media posts — giving it implicit understanding of fashion-specific visual attributes (color, pattern, silhouette, style) without requiring fine-tuning on a domain-specific dataset. This zero-shot transfer capability is critical: it allows the system to match a red floral dress in a lifestyle photo to the same dress in a clean catalog shot, despite dramatic differences in pose, lighting, background, and scale.

### YOLOS-tiny — Fashion Item Detection

**YOLOS-tiny** (`hustvl/yolos-tiny`) is used for bounding-box detection of fashion-relevant objects in scene images. It is a lightweight object detector based on the DETR (Detection Transformer) paradigm, fine-tuned on COCO object categories.

The detector filters for fashion-relevant COCO labels: `person`, `tie`, `handbag`, `backpack`, `suitcase`, `hat`, `shoe`, `dress`, `shirt`, `pants`, `jacket`, `coat`, `skirt`, `belt`, `watch`, `glasses`, and `sunglasses`.

> **Critical Limitation:** YOLOS-tiny detects bounding boxes around **whole persons** and large accessories. It does **not** segment individual garments (e.g., isolating a shirt from pants within a person's bounding box). When a full-person detection occurs, the entire outfit is embedded as a single vector, reducing per-item matching precision. The system flags these crops with `is_full_person=True` and includes a corresponding note in the match explanation.

**Fallback behavior:** If no fashion-relevant objects are detected (e.g., due to unusual framing or small item size), the system falls back to using the full scene image as a single crop.

---

## 3. Product Representation Strategy

### Catalog Deduplication

The raw catalog file contains approximately 50,000 lines, but individual product IDs appear with high multiplicity — some products are repeated up to 130 times across different scene pairings. The offline indexer deduplicates on product ID, preserving first-seen order, yielding **28,094 unique products** for embedding.

### Image Acquisition and Resilience

Product images are hosted on Pinterest's CDN. The download pipeline (`src/fetcher.py`) is designed for resilience:

| Feature | Implementation |
|---------|---------------|
| Parallelism | 12 concurrent `ThreadPoolExecutor` workers |
| Caching | Disk-based; each image downloaded at most once (`cache/images/{sig}.jpg`) |
| Retries | 3 attempts with exponential backoff (1.5s × attempt number) |
| 404 handling | Immediate skip — dead URLs are not retried |
| Crash recovery | Checkpoint every 500 items; resume from last checkpoint on restart |

An expected **10–20% failure rate** (~2,800–5,600 of 28,094 images) occurs due to expired or removed Pinterest URLs. Failed product IDs are logged to `artifacts/failed_downloads.txt` and excluded from the index. This reduces effective catalog coverage but is unavoidable given the reliance on third-party hosted URLs.

### Embedding and Indexing

Successfully downloaded images are embedded via CLIP ViT-B/32 in batches of 64 (GPU-accelerated when available). All embeddings are **L2-normalized** at embedding time, which is a mandatory invariant: `FAISS IndexFlatIP` computes raw inner products, and only on L2-normalized vectors does inner product equal cosine similarity.

The final index artifact is a `FAISS IndexFlatIP` of dimensionality 512, containing one vector per successfully embedded product. This index performs **exact** (brute-force) nearest-neighbor search — appropriate for the current catalog size of ~28K products, where each query completes in under 10 ms.

### Attribute Pre-computation

To avoid re-fetching catalog images during online inference, the offline indexer pre-computes attribute labels for every indexed product using CLIP zero-shot text scoring. For each attribute dimension, the product's image embedding is compared against text embeddings of candidate labels using the prompt template `"a {option} fashion item"`:

| Attribute | Options (count) |
|-----------|----------------|
| Color | red, blue, black, white, green, yellow, pink, brown, gray, beige, multicolor, navy, orange, purple (14) |
| Pattern | solid, striped, floral, checkered, polka dot, graphic print, abstract, animal print, plaid, tie-dye (10) |
| Category | dress, top, pants, skirt, jacket, shoes, handbag, accessory, coat, shorts, sweater, jumpsuit (12) |
| Style | casual, formal, sporty, bohemian, minimalist, vintage, streetwear, elegant, preppy, edgy (10) |

Results are saved to `artifacts/catalog_attributes.json`, enabling the explainer module to generate attribute-level match explanations at runtime without any network calls for catalog products.

---

## 4. Retrieval and Ranking Methodology

### Query-Time Pipeline

The online retrieval flow for a single scene proceeds as follows:

```
Scene Signature → Fetch Image → YOLOS-tiny Detection → Per-Crop CLIP Embedding
    → FAISS Top-K Search → Threshold Classification → Attribute Explanation
```

1. **Detection** — YOLOS-tiny identifies fashion-relevant bounding boxes in the scene image. Each crop is independently processed through the remaining pipeline stages.
2. **Embedding** — Each detected crop is embedded into a 512-d L2-normalized vector via CLIP ViT-B/32.
3. **FAISS Search** — The crop embedding is queried against the pre-built `IndexFlatIP`. Top-K results (default K=5) are returned with cosine similarity scores. Each FAISS query completes in **<10 ms**.
4. **Exact Match Classification** — Each result's cosine similarity is compared against a tuned threshold to determine whether it qualifies as an "exact match" (same product) versus a "similar product."
5. **Explanation** — The explainer classifies the scene crop's attributes on-the-fly via CLIP text scoring, then compares them against the pre-computed catalog product attributes to generate a human-readable explanation highlighting shared and differing attributes.

### Threshold Tuning

The exact-match similarity threshold is **not hardcoded**. It is determined empirically by `tune_threshold.py`, which sweeps cosine similarity values from 0.75 to 0.99 (step 0.01) on the 288 in-catalog validation pairs. At each candidate threshold, the script computes precision, recall, and F1 for the binary classification "is this top-1 result the correct product?" The threshold maximizing F1 is saved to `artifacts/config.json` and loaded by `src/matcher.py` at startup.

### Multi-Crop Aggregation and Ranking

When a scene contains multiple detected crops (e.g., a person plus a handbag), predictions from all crops are collected into a unified candidate pool:

1. **Collect** all `(product_id, similarity_score, is_exact_match)` tuples across all crops.
2. **Deduplicate** by product ID, retaining the highest similarity score for each product.
3. **Rank** globally by descending similarity score.

This ensures that a product detected in multiple crops (e.g., visible in both a person crop and an accessory crop) appears only once in the final ranking at its best score.

### Multi-Product Scene Handling

Eight validation scenes contain 2–3 ground-truth products. During evaluation, a prediction at rank K is considered correct if it matches **any** of the true products for that scene, preventing undercounting of correct matches in multi-product scenarios.

---

## 5. Evaluation Methodology

### Dataset Composition

The validation set (`data/validation.jsonl`) contains **330 scene–product pairs**:

| Statistic | Value |
|-----------|-------|
| Total validation pairs | 330 |
| Unique scenes | 321 |
| Unique products | 206 |
| Products in catalog | 165 (80.1%) |
| Products not in catalog | 41 (19.9%) |
| In-catalog pairs | 288 |
| Not-in-catalog pairs | 42 |
| Multi-product scenes | 8 (7 with 2 products, 1 with 3 products) |

### Metrics

| Metric | Definition | Denominator |
|--------|------------|-------------|
| **Recall@K** (K ∈ {1, 5, 10}) | Fraction of evaluated scenes where at least one true product appears in the top-K predictions | All evaluated scenes (including those with out-of-catalog products) |
| **MRR** (Mean Reciprocal Rank) | Mean of 1/rank for the first correct prediction across all evaluated scenes; 0 if no correct prediction exists | All evaluated scenes |
| **Exact Match Rate** | Fraction of in-catalog pairs where the top-1 prediction is the correct product **and** the system flags it as an exact match | 288 in-catalog pairs only (42 not-in-catalog pairs excluded) |

### Out-of-Catalog Product Handling

41 of 206 unique validation products (19.9%) are **not present** in the indexed catalog. These products cannot be retrieved by any system, regardless of quality. They are handled as follows:

- **Recall@K and MRR**: Out-of-catalog products are **included** in the denominator and count as misses. This reflects the system's real-world performance where not all ground-truth products may exist in the catalog.
- **Exact Match Rate**: Out-of-catalog pairs are **excluded** from computation, since an exact match is impossible by definition. The metric is computed over the 288 in-catalog pairs only.

### Multi-Product Scene Handling in Evaluation

For the 8 multi-product scenes, a prediction is correct if it matches **any** of the scene's true products. This avoids penalizing the system for returning a valid product that happens to be a different ground-truth item in the same scene.

---

## 6. Limitations and Future Improvements

### Current Limitations

| Limitation | Impact |
|------------|--------|
| **YOLOS-tiny detects whole persons, not individual garments** | Full-person crops embed entire outfits as single vectors, reducing per-item matching precision. Individual garments (shirt, pants, shoes) cannot be independently retrieved from a person crop. |
| **Pinterest URL attrition (~10–20%)** | Dead or expired URLs permanently reduce catalog coverage. Failed products can never be matched, even if they are ground-truth answers. |
| **CLIP visual conflation** | CLIP may assign high similarity to visually similar but distinct items (e.g., two different white dresses with similar silhouettes), leading to false positive exact-match classifications. |
| **No fashion-specific fine-tuning** | The system uses general-purpose CLIP pretrained on web data. A fashion-domain model (e.g., FashionCLIP) would yield significantly better embeddings for fine-grained fashion attributes. |
| **Threshold tuned on evaluation set** | The exact-match threshold is tuned on the same 288 in-catalog validation pairs used for final evaluation. This introduces a slight optimistic bias; ideally, a held-out calibration set would be used. |
| **Detection failures on small/cluttered items** | YOLOS-tiny may miss small accessories (jewelry, belts, watches) or fail on heavily cluttered scenes with overlapping objects. The 64px minimum crop filter further excludes tiny detections. |

### Future Improvements

1. **FashionCLIP Embeddings** — Replace general-purpose CLIP with FashionCLIP, which is fine-tuned on fashion-specific image–text pairs and demonstrates significantly better performance on fine-grained fashion attribute recognition and cross-domain retrieval.

2. **Instance Segmentation** — Replace YOLOS-tiny with a fashion-aware instance segmentation model (e.g., Fashionpedia or Attribute-Mask R-CNN) to extract pixel-precise individual garment masks rather than coarse bounding boxes. This would enable independent retrieval for each garment in a multi-item outfit.

3. **Cross-Encoder Re-ranking** — Add a second-stage cross-encoder that jointly attends to the query crop and each top-K candidate product image. Cross-encoders are more expressive than bi-encoders for fine-grained similarity judgments and can significantly improve precision in the re-ranking stage.

4. **Approximate Nearest Neighbors** — Replace `IndexFlatIP` (brute-force) with `IndexIVFFlat` or HNSW-based indices for catalogs exceeding 1M products. Current exact search is appropriate at 28K products but will not scale to production-size catalogs without approximate methods.

5. **User Feedback Loop** — Integrate implicit (click-through) and explicit (thumbs up/down) user feedback to continuously improve retrieval quality. Feedback signals can be used to fine-tune embeddings, adjust similarity thresholds, or train a learned re-ranker.

6. **Held-Out Threshold Calibration** — Split the validation set into a calibration partition and an evaluation partition to eliminate optimistic bias in exact-match threshold tuning.

---

*End of Technical Report*
