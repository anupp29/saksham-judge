"""
upload_to_hf.py
---------------
Run this ONCE on your local machine (or Colab) to create the Space
and push all files including the model weights.

Prerequisites:
    pip install huggingface_hub

Usage:
    python upload_to_hf.py --token hf_xxxxxxxxxxxx --user YOUR_HF_USERNAME
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Hugging Face write token")
    parser.add_argument("--user", required=True, help="Your HF username")
    parser.add_argument("--repo", default="boxing-judge", help="Space repo name")
    parser.add_argument(
        "--model-weights",
        default=str(HERE / "best_boxing_model.pth"),
        help="Path to best_boxing_model.pth",
    )
    args = parser.parse_args()

    repo_id = f"{args.user}/{args.repo}"
    api = HfApi(token=args.token)

    # ── 1. create (or reuse) the Space ─────────────────────────────────────
    print(f"Creating Space: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
        token=args.token,
    )

    # ── 2. upload source files ─────────────────────────────────────────────
    source_files = [
        "app.py",
        "model.py",
        "inference.py",
        "feature_extractor.py",
        "requirements.txt",
        "README.md",
    ]

    for fname in source_files:
        path = HERE / fname
        if not path.exists():
            print(f"  SKIP (not found): {fname}")
            continue
        print(f"  Uploading {fname} …")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="space",
        )

    # ── 3. upload model weights ────────────────────────────────────────────
    weights_path = Path(args.model_weights)
    if not weights_path.exists():
        print(f"ERROR: model weights not found at {weights_path}")
        print("Copy best_boxing_model.pth next to this script and re-run.")
        return

    print(f"  Uploading model weights ({weights_path.stat().st_size // 1024} KB) …")
    api.upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo="best_boxing_model.pth",
        repo_id=repo_id,
        repo_type="space",
    )

    print()
    print("=" * 60)
    print(f"✅  Space live at: https://huggingface.co/spaces/{repo_id}")
    print("    It will rebuild automatically (takes ~2–3 min).")
    print("=" * 60)


if __name__ == "__main__":
    main()