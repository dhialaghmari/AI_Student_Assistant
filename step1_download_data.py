"""
STEP 1 — Download the EducationQ dataset from Hugging Face.

Source: https://huggingface.co/datasets/SunriserFuture/EducationQ
License: CC-BY-4.0 (cite the authors in your report, see README).

Run:  python step1_download_data.py
Output: data/raw/educationq_full.json
"""
import shutil

from huggingface_hub import hf_hub_download

import config


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {config.HF_FILE} from {config.HF_DATASET} ...")
    cached_path = hf_hub_download(
        repo_id=config.HF_DATASET,
        filename=config.HF_FILE,
        repo_type="dataset",
    )
    shutil.copy(cached_path, config.RAW_FILE)
    size_kb = config.RAW_FILE.stat().st_size / 1024
    print(f"Saved to {config.RAW_FILE}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
