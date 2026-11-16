"""
STEP 2 — Inspect the raw dataset before cleaning it.

Prints the shape, columns, missing values, duplicates and the
distribution of subjects so you know what needs cleaning.
Copy the numbers into the "Dataset" section of your report.

Run:  python step2_explore_data.py
"""
import pandas as pd

import config
from data_utils import load_raw

pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 80)


def main() -> None:
    df = load_raw()

    print("=" * 70)
    print(f"Rows: {len(df)}   Columns: {len(df.columns)}")
    print("Columns:", list(df.columns))

    print("\n--- Missing values per column ---")
    print(df.isna().sum().to_string())

    print("\n--- Duplicates ---")
    print("Duplicate ids:      ", df["id"].duplicated().sum())
    print("Duplicate questions:", df["question"].str.strip().str.lower().duplicated().sum())

    for col in ["source", "category", "difficulty", "subdomain"]:
        if col in df.columns:
            print(f"\n--- {col} ---")
            print(df[col].fillna("<missing>").value_counts().to_string())

    print("\n--- Question length (characters) ---")
    print(df["question"].str.len().describe().round(0).to_string())

    print("\n--- Number of options per question ---")
    print(df["options"].apply(len).value_counts().sort_index().to_string())

    print("\n--- Example row ---")
    print(df.iloc[0].to_string())
    print("=" * 70)
    print(f"Source: {config.DATASET_URL}")


if __name__ == "__main__":
    main()
