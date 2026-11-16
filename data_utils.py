"""Small helpers shared by the data scripts."""
import json

import pandas as pd

import config


def load_raw() -> pd.DataFrame:
    """Load the raw EducationQ JSON file into a DataFrame.

    The file may be a JSON array or JSON Lines, and may wrap the records
    in a top-level key, so we handle each case.
    """
    text = config.RAW_FILE.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:  # JSON Lines
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(data, dict):  # e.g. {"data": [...]} or {"train": [...]}
        lists = [v for v in data.values() if isinstance(v, list)]
        data = lists[0] if lists else [data]

    return pd.DataFrame(data)
