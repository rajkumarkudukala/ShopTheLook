"""
deploy_to_hf.py — One-shot deploy of Shop-the-Look to Hugging Face Spaces.

Prerequisites:
    1. Create a token at https://huggingface.co/settings/tokens  (role: "write")
    2. Authenticate, either:
         huggingface-cli login          (paste the token)
       or set an env var before running this script:
         $env:HF_TOKEN="hf_xxx"         (PowerShell)

Usage:
    python deploy_to_hf.py --space your-username/shop-the-look

What it uploads:
    app.py, requirements.txt, README.md (with Spaces frontmatter),
    src/, data/, and the pre-built artifacts/ (FAISS index, etc.).
    It does NOT upload cache/, venv/, the dataset zip, or the docx.
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo

# Files / folders to ship to the Space
INCLUDE = ["app.py", "requirements.txt", "README.md", "src", "data", "artifacts"]

# Patterns to never upload (defense-in-depth on top of INCLUDE)
IGNORE = [
    "cache/*", "venv/*", "*.zip", "*.docx", "__pycache__/*",
    "*.pyc", "test_*.py", ".git/*", "checkpoint_*",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--space", required=True,
        help="Target Space id, e.g. 'username/shop-the-look'",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create the Space as private (default: public)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")  # falls back to cached login if None
    api = HfApi(token=token)

    # 1. Create the Space (no-op if it already exists)
    print(f"Creating Space: {args.space} ...")
    create_repo(
        repo_id=args.space,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    # 2. Upload each included path
    for path in INCLUDE:
        if not os.path.exists(path):
            print(f"  skip (missing): {path}")
            continue
        if os.path.isdir(path):
            print(f"  uploading folder: {path}/ ...")
            api.upload_folder(
                folder_path=path,
                path_in_repo=path,
                repo_id=args.space,
                repo_type="space",
                ignore_patterns=IGNORE,
            )
        else:
            print(f"  uploading file: {path} ...")
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=path,
                repo_id=args.space,
                repo_type="space",
            )

    print(f"\nDone. Your Space is building at:")
    print(f"  https://huggingface.co/spaces/{args.space}")
    print("First build takes a few minutes (installs deps + downloads models).")


if __name__ == "__main__":
    main()
