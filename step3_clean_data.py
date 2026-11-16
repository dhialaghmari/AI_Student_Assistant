"""
STEP 3 — Clean the dataset and turn it into a knowledge base.

What it does (matches section 7 of the specification):
  1. Clean unnecessary text   (HTML tags, extra spaces, odd characters)
  2. Handle missing values    (drop unusable rows, fill optional fields)
  3. Remove duplicate data    (same id or same question text)
  4. Organize the information (one readable "document" per question,
                               containing question + correct answer +
                               explanation + subject, ready for RAG)

Run:  python step3_clean_data.py
Output: data/processed/educationq_clean.csv  (for humans / the report)
        data/processed/educationq_clean.jsonl (used by step 4)
"""
import json
import re
import unicodedata

import pandas as pd

import config
from data_utils import load_raw

LETTERS = "ABCDEFGHIJ"


# ------------------------------------------------------------------ text
def clean_text(value) -> str:
    """Normalize a piece of text: unicode, HTML tags, whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = text.replace("​", "")              # zero-width spaces
    text = re.sub(r"[ \t]+", " ", text)            # repeated spaces
    text = re.sub(r"\n\s*\n+", "\n", text)         # blank lines
    return text.strip()


def parse_options(value) -> list[str]:
    """Options are normally a list; accept a JSON string just in case."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    options = [clean_text(o) for o in value]
    # Drop empty / placeholder options that some MMLU-Pro rows contain.
    return [o for o in options if o and o.upper() != "N/A"]


def normalize_category(row) -> str:
    """Use `category`; for GPQA rows fall back to the subdomain."""
    cat = clean_text(row.get("category")).lower()
    if not cat:
        sub = clean_text(row.get("subdomain"))
        cat = re.sub(r"\s*\(.*\)", "", sub).lower()   # "Physics (general)" -> "physics"
    return cat or "general"


# ---------------------------------------------------------------- answer
def resolve_answer(row, raw_options) -> tuple[str, str]:
    """Return (letter, answer text) for the correct option, or ("", "")."""
    raw = [clean_text(o) for o in raw_options] if isinstance(raw_options, (list, tuple)) else []
    idx = row.get("answer_index")
    if idx is None or pd.isna(idx):
        letter = clean_text(row.get("answer")).upper()[:1]
        idx = LETTERS.find(letter) if letter else -1
    idx = int(idx)
    if 0 <= idx < len(raw) and raw[idx]:
        return LETTERS[idx], raw[idx]
    return "", ""


# -------------------------------------------------------------- document
def build_document(r: dict) -> str:
    """The text that gets embedded and shown to the LLM as context."""
    parts = [
        f"Subject: {r['category'].title()}",
        f"Question: {r['question']}",
        f"Correct answer: {r['answer_text']}",
    ]
    if r["explanation"]:
        parts.append(f"Explanation: {r['explanation']}")
    other = [o for o in r["options"] if o != r["answer_text"]]
    if other:
        parts.append("Other options considered (incorrect): " + "; ".join(other))
    return "\n".join(parts)


def main() -> None:
    df = load_raw()
    report = {"raw rows": len(df)}

    # 1. Clean text fields ------------------------------------------------
    for col in ["id", "source", "question", "difficulty", "original_source", "explanation"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(clean_text)

    df["category"] = df.apply(normalize_category, axis=1)
    resolved = df.apply(lambda r: resolve_answer(r, r["options"]), axis=1)
    df["answer_letter"] = resolved.str[0]
    df["answer_text"] = resolved.str[1]
    df["options"] = df["options"].apply(parse_options)

    # 2. Handle missing values -------------------------------------------
    before = len(df)
    df = df[(df["question"] != "") & (df["answer_text"] != "")]
    report["dropped (missing question or answer)"] = before - len(df)
    df["difficulty"] = df["difficulty"].replace("", "unknown")
    df["original_source"] = df["original_source"].replace("", "unknown")
    # explanation stays "" when absent (only GPQA rows have one)

    # 3. Remove duplicates -----------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset="id")
    df["_q_norm"] = df["question"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset="_q_norm").drop(columns="_q_norm")
    report["dropped (duplicates)"] = before - len(df)

    # 4. Organize into knowledge-base documents --------------------------
    df = df.reset_index(drop=True)
    df["document"] = df.apply(lambda r: build_document(r.to_dict()), axis=1)

    keep = ["id", "source", "category", "difficulty", "original_source",
            "question", "options", "answer_letter", "answer_text",
            "explanation", "document"]
    df = df[keep]
    report["clean rows"] = len(df)

    # Save ----------------------------------------------------------------
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(config.CLEAN_JSONL, orient="records", lines=True, force_ascii=False)
    csv_df = df.copy()
    csv_df["options"] = csv_df["options"].apply(lambda o: " | ".join(o))
    csv_df.to_csv(config.CLEAN_CSV, index=False)

    print("Cleaning report")
    for k, v in report.items():
        print(f"  {k:<40} {v}")
    print("\nRows per subject:")
    print(df["category"].value_counts().to_string())
    print(f"\nSaved {config.CLEAN_CSV}\nSaved {config.CLEAN_JSONL}")
    print("\nExample document:\n" + "-" * 70 + "\n" + df.loc[0, "document"])


if __name__ == "__main__":
    main()
