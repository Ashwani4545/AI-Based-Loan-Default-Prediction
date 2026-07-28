# src/hf_hub_manager.py
"""
Hugging Face Hub CLI & Script Manager for AegisBank.

Usage:
  python src/hf_hub_manager.py push --repo my-org/my-model --token hf_...
  python src/hf_hub_manager.py pull --repo my-org/my-model
  python src/hf_hub_manager.py push-dataset --repo my-org/my-dataset
"""

import sys
import os
import argparse
import logging
from pathlib import Path

sys_path_root = Path(__file__).resolve().parent.parent
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from utils.hf_hub import (
    upload_model_to_hf,
    download_model_from_hf,
    upload_dataset_to_hf,
    check_hf_hub_available,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="AegisBank Hugging Face Hub Manager")
    parser.add_argument("action", choices=["push", "pull", "push-dataset", "check"], help="Action to perform")
    parser.add_argument("--repo", type=str, default=None, help="Hugging Face repository ID (e.g. username/repo-name)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Access Token")

    args = parser.parse_args()

    if args.action == "check":
        available = check_hf_hub_available()
        print(f"Hugging Face Hub Library Installed: {available}")
        if not available:
            print("Run: pip install huggingface_hub transformers")
        sys.exit(0)

    elif args.action == "push":
        log.info("Pushing model artifacts to Hugging Face Model Hub...")
        res = upload_model_to_hf(repo_id=args.repo, token=args.token)
        print(res)

    elif args.action == "pull":
        log.info("Pulling model artifacts from Hugging Face Model Hub...")
        res = download_model_from_hf(repo_id=args.repo, token=args.token)
        print(res)

    elif args.action == "push-dataset":
        log.info("Pushing dataset to Hugging Face Datasets Hub...")
        res = upload_dataset_to_hf(repo_id=args.repo, token=args.token)
        print(res)


if __name__ == "__main__":
    main()
