"""
src/pipeline.py — End-to-end inference pipeline.

Full flow: scene signature → fetch image → detect fashion crops →
embed each crop → FAISS top-K search → generate explanations.

Returns structured results that can be consumed by both evaluate.py
and the Gradio demo (app.py).
"""

from src.fetcher import fetch_image
from src.detector import detect_fashion_crops
from src.embedder import embed_single_image
from src.matcher import match_crop
from src.explainer import explain_match


def process_scene(scene_signature: str, top_k: int = 5) -> dict:
    """
    Full end-to-end pipeline: scene signature → ranked catalog matches.

    Args:
        scene_signature: Pinterest image signature hash for the scene
        top_k: Number of catalog matches to return per detected crop

    Returns:
        Dict containing:
        - "scene_signature": str — input signature
        - "scene_image": PIL Image or None — the fetched scene image
        - "error": str or None — error message if scene unavailable
        - "items": list of dicts, each containing:
            - "crop": PIL Image — detected fashion item crop
            - "label": str — detection label (e.g., "person", "handbag")
            - "confidence": float — detection confidence
            - "is_full_person": bool — True when crop is whole-person box
            - "matches": list of dicts, each containing:
                - "product_id": str — catalog product hash
                - "similarity_score": float — cosine similarity
                - "is_exact_match": bool — above tuned threshold
                - "rank": int — 1-indexed rank
                - "explanation": str — human-readable match explanation
    """
    # 1. Fetch scene image
    scene_image = fetch_image(scene_signature)
    if scene_image is None:
        return {
            "scene_signature": scene_signature,
            "scene_image": None,
            "error": f"Scene image unavailable: {scene_signature}",
            "items": [],
        }

    # 2+. Run the image-based core pipeline
    return process_image(scene_image, top_k=top_k, scene_signature=scene_signature)


def process_image(scene_image, top_k: int = 5, scene_signature: str = "uploaded") -> dict:
    """
    Core pipeline on an already-loaded PIL image (e.g. a user upload).

    Same flow as process_scene but skips the Pinterest fetch step. Useful
    for direct image uploads where there is no signature hash.

    Args:
        scene_image: PIL Image of the scene
        top_k: Number of catalog matches per detected crop
        scene_signature: Optional label for the source (default "uploaded")

    Returns:
        Same dict structure as process_scene.
    """
    # Detect fashion crops in the scene
    crops = detect_fashion_crops(scene_image)

    # For each crop: embed → match → explain
    items = []
    for crop_info in crops:
        crop_emb = embed_single_image(crop_info["crop"])
        matches = match_crop(crop_emb, top_k=top_k)

        # Add explanations to each match
        for match in matches:
            match["explanation"] = explain_match(
                scene_embedding=crop_emb,
                product_id=match["product_id"],
                similarity_score=match["similarity_score"],
                is_full_person=crop_info["is_full_person"],
            )

        items.append({
            "crop": crop_info["crop"],
            "label": crop_info["label"],
            "confidence": crop_info["confidence"],
            "is_full_person": crop_info["is_full_person"],
            "matches": matches,
        })

    return {
        "scene_signature": scene_signature,
        "scene_image": scene_image,
        "error": None,
        "items": items,
    }
