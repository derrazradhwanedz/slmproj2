"""Downloads every benchmark listed in src/config/becnmarks.yaml as JSONL.

Run directly as: python src/utils/benchmarks.py
"""

import os
import sys
from typing import Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from loader import load_yaml

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmarks",
)


def _repo_id_from_url(url: str) -> str:
    """Extract a HuggingFace dataset repo id from its hub URL.

    Args:
        url: A URL like "https://huggingface.co/datasets/<owner>/<name>".

    Returns:
        The "<owner>/<name>" repo id.
    """
    return url.rstrip("/").split("datasets/")[-1]


def save_dataset(
    repo_id: str,
    output_dir: str = OUTPUT_DIR,
    config: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> None:
    """Download a HuggingFace dataset and save every split as JSONL.

    Args:
        repo_id: HuggingFace dataset repo id (e.g. "nguyen-brat/aqua").
        output_dir: Root directory to save benchmarks under. Each dataset
            gets its own subfolder, named dataset_name.
        config: Optional dataset config name, for repos that expose more
            than one (e.g. "openai/gsm8k" needs "main" or "socratic").
        dataset_name: Folder name to save under. Defaults to the part of
            repo_id after the last "/", but should be given explicitly for
            multi-config repos (e.g. "tasksource/bigbench") so the folder
            is named after the benchmark, not the shared repo.
    """
    dataset = load_dataset(repo_id, config) if config else load_dataset(repo_id)
    dataset_name = dataset_name or repo_id.split("/")[-1]
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    for split_name, split_data in dataset.items():  # type: ignore[union-attr]
        out_path = os.path.join(dataset_dir, f"{split_name}.jsonl")
        split_data.to_json(out_path)
        print(f"Saved {split_name} ({len(split_data)} rows) to {out_path}")


def save_all_benchmarks(output_dir: str = OUTPUT_DIR) -> None:
    """Download and save every benchmark listed in becnmarks.yaml.

    Each entry is either a plain URL string, or a dict with "url" and an
    optional "config" key for repos that require a config name.

    Args:
        output_dir: Root directory to save benchmarks under.
    """
    benchmarks = load_yaml("becnmarks.yaml")
    for name, entry in benchmarks.items():
        if isinstance(entry, dict):
            url, config = entry["url"], entry.get("config")
        else:
            url, config = entry, None

        repo_id = _repo_id_from_url(url)
        print(f"Downloading {name} ({repo_id})...")
        try:
            save_dataset(repo_id, output_dir, config=config, dataset_name=name)
        except Exception as exc:
            print(f"Failed to download {name} ({repo_id}): {exc}")


if __name__ == "__main__":
    save_all_benchmarks()
