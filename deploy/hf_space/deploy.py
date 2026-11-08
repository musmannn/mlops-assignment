"""Assemble and upload the model API to a Hugging Face Space (Task 7).

Prerequisites (you do these once, by hand):
  1. Create a free account at https://huggingface.co
  2. Settings -> Access Tokens -> New token (type "Write"). Put it in .env as HF_TOKEN=...
  3. models/champion.keras + champion.json exist (python -m training.register)

Run:
  python deploy/hf_space/deploy.py --user <hf-username>

The script also stores MODEL_API_KEY from .env as a Space secret
(visible in the Space under Settings -> Variables and secrets, value hidden).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi

from training.common import load_dotenv

SPACE = "24L-0672_A2_Ops"


def build_context(dst: Path) -> None:
    here = Path(__file__).parent
    shutil.copy(here / "Dockerfile", dst / "Dockerfile")
    shutil.copy(here / "README.md", dst / "README.md")
    reqs = [line for line in (ROOT / "requirements-serve.txt").read_text().splitlines()
            if not line.startswith(("psycopg2", "sqlalchemy"))]  # the API has no database
    (dst / "requirements.txt").write_text("\n".join(reqs) + "\n")
    shutil.copytree(ROOT / "visionguard", dst / "visionguard",
                    ignore=shutil.ignore_patterns("__pycache__", "web"))
    (dst / "models").mkdir()
    for f in ["champion.keras", "champion.json"]:
        shutil.copy(ROOT / "models" / f, dst / "models" / f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your Hugging Face username")
    args = ap.parse_args()
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN missing from .env"
    repo_id = f"{args.user}/{SPACE}"
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    # The API key becomes a Space *secret* (env var inside the container), never a file in the repo.
    api_key = os.environ.get("MODEL_API_KEY")
    assert api_key, "MODEL_API_KEY missing from .env"
    api.add_space_secret(repo_id, "MODEL_API_KEY", api_key)
    with tempfile.TemporaryDirectory() as tmp:
        build_context(Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="space",
                          commit_message="Deploy VisionGuard model API")
    sub = repo_id.lower().replace("_", "-").replace("/", "-")
    print(f"Space:   https://huggingface.co/spaces/{repo_id}")
    print(f"API URL: https://{sub}.hf.space   (building takes ~5 min; watch the 'Logs' tab)")


if __name__ == "__main__":
    main()
