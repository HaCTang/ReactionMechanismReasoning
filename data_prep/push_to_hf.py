#!/usr/bin/env python3
"""Push the assembled training dataset to the mech-infer-train HF org (private)."""
import sys
from pathlib import Path
from huggingface_hub import HfApi

DEFAULT_REPO_ID = "Haocheng1/mech-infer-train-v1"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_UPLOADS = [
    (
        DATA_DIR / "mech_infer_train_multistep",
        "mech_infer_train_multistep",
        "Add mech-infer-train multi-step core: 3,310 decontaminated mechanism trajectories",
    ),
    (
        DATA_DIR / "mech_infer_train_singlestep",
        "mech_infer_train_singlestep",
        "Add mech-infer-train single-step B/C similar unlabeled reactions",
    ),
    (
        DATA_DIR / "sarpang",
        "sarpang",
        "Add Sarpong augmented reaction data",
    ),
]


def upload_one(api: HfApi, folder: Path, repo_id: str, path_in_repo: str, commit_message: str) -> None:
    """Upload one local dataset folder to HuggingFace."""
    print("folder:", folder, "exists:", folder.exists())
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    print("files:", [p.name for p in folder.iterdir()])

    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    print(f"uploaded to {repo_id}/{path_in_repo}")


def ensure_dataset_repo(api: HfApi, repo_id: str) -> str:
    """Ensure a dataset repo exists without falling back to another namespace."""
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print("repo exists:", repo_id)
        return repo_id
    except Exception as exc:
        print(f"repo_info failed for {repo_id} ({type(exc).__name__}: {exc})")

    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        print("repo ensured:", repo_id)
        return repo_id
    except Exception as exc:
        print(f"create_repo failed for {repo_id} ({type(exc).__name__}: {exc})")
        raise


def main() -> int:
    api = HfApi()
    who = api.whoami()
    user_name = who.get("name")
    repo_id = DEFAULT_REPO_ID
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        repo_id = arg if "/" in arg else f"{user_name}/{arg}"
    print("authenticated as:", user_name, "| requested repo:", repo_id)

    try:
        print("existing files in requested repo:", api.list_repo_files(repo_id, repo_type="dataset"))
    except Exception as e:
        print(f"could not list requested repo ({type(e).__name__}: {e})")

    repo_id = ensure_dataset_repo(api, repo_id)
    print("upload target repo:", repo_id)

    if len(sys.argv) > 2:
        folder = Path(sys.argv[2])
        path_in_repo = sys.argv[3] if len(sys.argv) > 3 else folder.name
        upload_one(
            api,
            folder,
            repo_id,
            path_in_repo,
            f"Add {path_in_repo}",
        )
    else:
        for folder, path_in_repo, commit_message in DEFAULT_UPLOADS:
            upload_one(api, folder, repo_id, path_in_repo, commit_message)

    print("FILES NOW:", api.list_repo_files(repo_id, repo_type="dataset"))
    print("URL: https://huggingface.co/datasets/" + repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
