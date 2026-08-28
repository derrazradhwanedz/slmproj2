"""Dataset loading utilities."""

import json
import logging
import os
from typing import Any, Dict, List, Union

# Plain stdlib logger, not the app's logger.get_logger wrapper: that wrapper
# lives in the logger module, which imports from this utils package (to load
# its own YAML config) - importing it back here would be a circular import.
# This still routes through whatever handlers configure_logging() installs,
# since Python's logging registry is a global singleton.
logger = logging.getLogger(__name__)


def load_json_files_from_directory(path: str, recursive: bool = True) -> List[Dict[str, Any]]:
    """Load all JSON and JSONL files from a directory.

    Args:
        path: Directory path to search for JSON/JSONL files.
        recursive: Whether to include subdirectories. Defaults to True.

    Returns:
        A list of all parsed JSON records across matching files.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    all_records: List[Dict[str, Any]] = []
    walker = os.walk(path) if recursive else [(path, [], os.listdir(path))]

    for root, _, files in walker:
        for file in files:
            if not (file.endswith(".json") or file.endswith(".jsonl")):
                continue

            full_path = os.path.join(root, file)
            try:
                with open(full_path, "r", encoding="utf-8") as infile:
                    content = infile.read().strip()

                if file.endswith(".jsonl"):
                    records = [json.loads(line) for line in content.split("\n") if line.strip()]
                else:
                    data: Union[List[Any], Dict[str, Any]] = json.loads(content)
                    records = data if isinstance(data, list) else [data]

                all_records.extend(records)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(f"Skipping unreadable file {full_path}: {exc}")

    return all_records
