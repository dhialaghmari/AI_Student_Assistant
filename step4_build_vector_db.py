"""
STEP 4 — Generate embeddings and store them in ChromaDB.

For every cleaned question we embed a compact text
(subject + question + correct answer + explanation) with a Sentence
Transformer, then store the vector, the full document and its metadata
in a persistent ChromaDB collection on disk (folder `chroma_db/`).

Run:  python step4_build_vector_db.py
"""
import time

import pandas as pd

import config
from embeddings import embed, get_collection

BATCH = 256


def embedding_text(row) -> str:
    """What we embed. Leaving out the wrong options keeps vectors focused."""
    text = f"{row['category']}: {row['question']}\nAnswer: {row['answer_text']}"
    if row["explanation"]:
        text += f"\n{row['explanation']}"
    return text


def main() -> None:
    df = pd.read_json(config.CLEAN_JSONL, lines=True).fillna("")
    print(f"Loaded {len(df)} clean questions")
    print(f"Embedding model: {config.EMBEDDING_MODEL}")

    collection = get_collection(create=True)
    start = time.time()

    for i in range(0, len(df), BATCH):
        batch = df.iloc[i : i + BATCH]
        collection.add(
            ids=batch["id"].tolist(),
            embeddings=embed([embedding_text(r) for _, r in batch.iterrows()]),
            documents=batch["document"].tolist(),
            metadatas=[
                {
                    "id": r["id"],
                    "category": r["category"],
                    "source": r["source"],
                    "original_source": r["original_source"],
                    "difficulty": r["difficulty"],
                    "question": r["question"][:1000],
                    "answer": f"{r['answer_letter']}) {r['answer_text']}",
                }
                for _, r in batch.iterrows()
            ],
        )
        print(f"  stored {min(i + BATCH, len(df))}/{len(df)}")

    print(f"\nDone in {time.time() - start:.1f}s — {collection.count()} vectors "
          f"in collection '{config.COLLECTION_NAME}' at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
