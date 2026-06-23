"""
src/embedder.py — CLIP ViT-B/32 image and text embedder.

Provides:
- Batch image embedding (GPU-accelerated if available)
- Single image embedding
- Text embedding (for explainer attribute classification)

All embeddings are L2-normalized to enable cosine similarity
via FAISS IndexFlatIP (inner product on normalized vectors).

Output dimension: 512 (ViT-B/32)
"""

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "patrickjohncyh/fashion-clip"
EMBED_DIM = 512

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor once at import time
_model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
_processor = CLIPProcessor.from_pretrained(MODEL_NAME)


def embed_images_batch(pil_images: list, batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of PIL Images using CLIP vision encoder.

    Args:
        pil_images: List of PIL Image objects (any mode — will be converted)
        batch_size: Number of images to process per forward pass

    Returns:
        float32 array of shape (N, 512), L2-normalized.
        L2 normalization is REQUIRED — FAISS IndexFlatIP on normalized
        vectors computes cosine similarity (not raw inner product).
    """
    all_embeddings = []
    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i : i + batch_size]
        inputs = _processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = _model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output if hasattr(features, "pooler_output") else features[0]
        features = features / features.norm(dim=-1, keepdim=True)
        all_embeddings.append(features.cpu().numpy())
    return np.vstack(all_embeddings).astype(np.float32)


def embed_single_image(pil_image) -> np.ndarray:
    """
    Embed a single PIL Image.

    Returns:
        float32 array of shape (512,), L2-normalized.
    """
    return embed_images_batch([pil_image])[0]


def embed_texts(texts: list) -> np.ndarray:
    """
    Embed a list of text strings using CLIP text encoder.
    Used by the explainer for attribute classification.

    Args:
        texts: List of text strings to embed

    Returns:
        float32 array of shape (N, 512), L2-normalized.
    """
    inputs = _processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        features = _model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output if hasattr(features, "pooler_output") else features[0]
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)
