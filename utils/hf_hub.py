# utils/hf_hub.py
"""
Hugging Face Hub Utility Module for AegisBank Loan Default System.
Provides functions for model artifact pushing/pulling, dataset syncing,
and status checking with HF Hub.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

sys_path_root = Path(__file__).resolve().parent.parent
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from utils.config import (
    BASE_DIR, CHAMPION_MODEL_PATH, CHALLENGER_MODEL_PATH,
    FEATURES_PATH, METRICS_PATH, PROCESSED_DATA_PATH,
    HF_MODEL_REPO, HF_DATASET_REPO, HF_TOKEN,
)

log = logging.getLogger(__name__)


def check_hf_hub_available() -> bool:
    """Return True if huggingface_hub Python library is installed."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        return False


def upload_model_to_hf(
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    commit_message: str = "Update AegisBank model artifacts",
    files_to_upload: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Upload champion model, features, and metrics to Hugging Face Model Hub.
    """
    repo_id = repo_id or HF_MODEL_REPO
    token = token or HF_TOKEN or os.environ.get("HF_TOKEN")

    if not check_hf_hub_available():
        return {
            "status": "error",
            "message": "huggingface_hub package is not installed. Install with `pip install huggingface_hub`.",
        }

    if not token:
        return {
            "status": "error",
            "message": "HF_TOKEN is missing. Provide a Hugging Face Access Token to upload to Hub.",
        }

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
    except Exception as e:
        log.warning(f"Repo creation notice for {repo_id}: {e}")

    default_files = [CHAMPION_MODEL_PATH, FEATURES_PATH, METRICS_PATH]
    if os.path.exists(CHALLENGER_MODEL_PATH):
        default_files.append(CHALLENGER_MODEL_PATH)

    files = files_to_upload or default_files
    uploaded_files = []

    for file_path in files:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            log.info(f"Uploading {filename} to HF Hub repo {repo_id}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            uploaded_files.append(filename)

    return {
        "status": "success",
        "repo_id": repo_id,
        "uploaded_files": uploaded_files,
        "repo_url": f"https://huggingface.co/{repo_id}",
    }


def download_model_from_hf(
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    target_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download champion model artifacts from Hugging Face Model Hub.
    """
    repo_id = repo_id or HF_MODEL_REPO
    token = token or HF_TOKEN or os.environ.get("HF_TOKEN")
    target_dir = target_dir or os.path.join(BASE_DIR, "models")

    if not check_hf_hub_available():
        return {
            "status": "error",
            "message": "huggingface_hub package is not installed.",
        }

    from huggingface_hub import hf_hub_download

    artifacts = ["champion_model.pkl", "model_features.pkl", "model_metrics.json"]
    downloaded = []

    os.makedirs(target_dir, exist_ok=True)

    for artifact in artifacts:
        try:
            log.info(f"Downloading {artifact} from HF Hub {repo_id}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=artifact,
                token=token,
                local_dir=target_dir,
            )
            downloaded.append(os.path.basename(downloaded_path))
        except Exception as e:
            log.warning(f"Could not download {artifact} from {repo_id}: {e}")

    if downloaded:
        return {
            "status": "success",
            "repo_id": repo_id,
            "downloaded_files": downloaded,
            "target_dir": target_dir,
        }
    else:
        return {
            "status": "error",
            "message": f"Failed to download model artifacts from Hugging Face Hub ({repo_id}).",
        }


def upload_dataset_to_hf(
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    data_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload cleaned dataset to Hugging Face Datasets Hub.
    """
    repo_id = repo_id or HF_DATASET_REPO
    token = token or HF_TOKEN or os.environ.get("HF_TOKEN")
    data_path = data_path or PROCESSED_DATA_PATH

    if not check_hf_hub_available():
        return {"status": "error", "message": "huggingface_hub is not installed."}

    if not token:
        return {"status": "error", "message": "HF_TOKEN is missing."}

    if not os.path.exists(data_path):
        return {"status": "error", "message": f"Dataset file does not exist: {data_path}"}

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    try:
        create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True)
        filename = os.path.basename(data_path)
        api.upload_file(
            path_or_fileobj=data_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload processed loan dataset",
        )
        return {
            "status": "success",
            "repo_id": repo_id,
            "dataset_url": f"https://huggingface.co/datasets/{repo_id}",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
