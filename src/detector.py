"""
src/detector.py — Fashion item detector and cropper for scene images.

Uses YOLOS fine-tuned on Fashionpedia (valentinafeve/yolos-fashionpedia),
which detects actual garments (shirt, dress, pants, skirt, shoe, bag, etc.)
rather than the generic COCO "person" box that the base YOLOS-tiny produces.

This matters for shop-the-look retrieval: a whole-person crop embeds the
entire outfit + body + background and does not match a clean single-product
catalog photo. Cropping the individual garment closes that domain gap.

Handling:
- Detect individual garments → crop each one separately
- If detection fails entirely → fall back to full scene image as one crop
"""

import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor

MODEL_NAME = "valentinafeve/yolos-fashionpedia"

# Fashionpedia's 46 categories (index order matches the model's label ids).
FASHIONPEDIA_LABELS = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan",
    "jacket", "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit",
    "cape", "glasses", "hat", "headband, head covering, hair accessory", "tie",
    "glove", "watch", "belt", "leg warmer", "tights, stockings", "sock",
    "shoe", "bag, wallet", "scarf", "umbrella", "hood", "collar", "lapel",
    "epaulette", "sleeve", "pocket", "neckline", "buckle", "zipper", "applique",
    "bead", "bow", "flower", "fringe", "ribbon", "rivet", "ruffle", "sequin",
    "tassel",
]

# Sub-garment parts that are not useful standalone retrieval targets — skip them.
SKIP_LABELS = {
    "sleeve", "pocket", "neckline", "collar", "lapel", "epaulette", "buckle",
    "zipper", "applique", "bead", "bow", "flower", "fringe", "ribbon", "rivet",
    "ruffle", "sequin", "tassel", "hood",
}

# Minimum crop size in pixels — ignore tiny detections
MIN_CROP_PX = 40

# Detection confidence threshold
SCORE_THRESHOLD = 0.30

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model + processor once at import time
_processor = YolosImageProcessor.from_pretrained(MODEL_NAME)
_model = YolosForObjectDetection.from_pretrained(MODEL_NAME).to(device).eval()


def detect_fashion_crops(pil_image: Image.Image) -> list:
    """
    Detect fashion garments in a scene image and return cropped regions.

    Args:
        pil_image: PIL Image of the scene

    Returns:
        List of dicts, each containing:
        - "crop": PIL Image of the detected region
        - "label": str — detected garment label
        - "confidence": float — detection confidence score
        - "is_full_person": bool — always False (garment-level detection)
        - "bbox": tuple (x1, y1, x2, y2) — bounding box coordinates

    Fallback: if nothing detected, returns the full image as one crop
    with label="full_scene".
    """
    inputs = _processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = _model(**inputs)

    # (height, width) for rescaling normalized boxes to pixel coords
    target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
    results = _processor.post_process_object_detection(
        outputs, threshold=SCORE_THRESHOLD, target_sizes=target_sizes
    )[0]

    crops = []
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label_idx = int(label_id)
        if label_idx >= len(FASHIONPEDIA_LABELS):
            continue
        label = FASHIONPEDIA_LABELS[label_idx]

        # Skip sub-garment parts that aren't standalone retrieval targets
        if label in SKIP_LABELS:
            continue

        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        w, h = x2 - x1, y2 - y1

        # Skip tiny detections
        if w < MIN_CROP_PX or h < MIN_CROP_PX:
            continue

        crop = pil_image.crop((x1, y1, x2, y2))
        crops.append({
            "crop": crop,
            "label": label,
            "confidence": float(score),
            "is_full_person": False,
            "bbox": (x1, y1, x2, y2),
        })

    # Fallback: use full image if nothing was detected
    if not crops:
        w, h = pil_image.size
        crops.append({
            "crop": pil_image,
            "label": "full_scene",
            "confidence": 1.0,
            "is_full_person": False,
            "bbox": (0, 0, w, h),
        })

    return crops
