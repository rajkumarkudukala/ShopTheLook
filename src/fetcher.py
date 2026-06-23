"""
src/fetcher.py — Robust image downloader for Pinterest-hosted fashion images.

Handles:
- Disk caching (each image downloaded at most once)
- Retry with exponential backoff
- 404-aware early exit (dead Pinterest URLs)
- Parallel batch downloading with configurable concurrency

Both product images and scene images use the same URL convention.
"""

import os
import time
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_DIR = "cache/images"
os.makedirs(CACHE_DIR, exist_ok=True)


def convert_to_url(signature: str) -> str:
    """
    Convert a Pinterest signature hash to a full image URL.
    Works for both product and scene signatures.

    Example:
        convert_to_url("0027e30879ce3d87f82f699f148bff7e")
        → "http://i.pinimg.com/400x/00/27/e3/0027e30879ce3d87f82f699f148bff7e.jpg"
    """
    prefix = 'https://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)


def fetch_image(signature: str, retries: int = 3, delay: float = 1.5):
    """
    Download the image for a signature hash.
    Returns PIL Image (RGB) or None if permanently unavailable.
    Caches to disk so each image is downloaded at most once.

    Args:
        signature: Pinterest image signature hash
        retries: Number of retry attempts for transient failures
        delay: Base delay between retries (multiplied by attempt number)

    Returns:
        PIL Image in RGB mode, or None if image is unavailable
    """
    cache_path = os.path.join(CACHE_DIR, f"{signature}.jpg")

    # Return cached version if available
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            os.remove(cache_path)  # corrupt file — delete and re-download

    url = convert_to_url(signature)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10, headers=headers)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img.save(cache_path, format="JPEG", quality=85)
                return img
            elif resp.status_code == 404:
                return None  # permanently gone — no point retrying
            else:
                time.sleep(delay * (attempt + 1))
        except requests.exceptions.Timeout:
            time.sleep(delay * (attempt + 1))
        except Exception:
            time.sleep(delay * (attempt + 1))

    return None


def parallel_download(signatures: list, max_workers: int = 12) -> dict:
    """
    Download images for a list of signature hashes in parallel.

    Args:
        signatures: List of Pinterest signature hashes
        max_workers: Number of concurrent download threads.
            12 workers: fast enough (~45 min for 28k images)
            without triggering Pinterest rate limits.

    Returns:
        Dict mapping {signature: PIL Image or None}
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sig = {
            executor.submit(fetch_image, sig): sig for sig in signatures
        }
        for future in as_completed(future_to_sig):
            sig = future_to_sig[future]
            try:
                results[sig] = future.result()
            except Exception:
                results[sig] = None
    return results
