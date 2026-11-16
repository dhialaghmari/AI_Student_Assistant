"""
STEP 7 — Test the system (section 9 of the specification).

Four test groups:
  A. Questions directly from the dataset      -> should retrieve the same item
  B. Same questions with different wording    -> should still retrieve it
  C. Questions NOT in the dataset             -> should return "not found"
  D. Results broken down by subject/category

For group B, if an LLM is configured it writes real paraphrases; otherwise a
simple automatic rewording is used (keywords only, shuffled order).

Run:  python step7_evaluate.py            (retrieval tests, fast)
      python step7_evaluate.py --answers  (also generates a few full answers)
Output: results/evaluation.csv, results/sample_answers.md
"""
import argparse
import random
import re

import pandas as pd

import config
import llm
from rag import answer_question, retrieve

RESULTS_DIR = config.ROOT_DIR / "results"
SAMPLES_PER_SUBJECT = 4
SEED = 42

OUT_OF_SCOPE = [
    "Who won the football World Cup in 2022?",
    "What is the best recipe for chocolate cake?",
    "What will the weather be like in Tunis tomorrow?",
    "Recommend me a good Netflix series to watch tonight.",
    "How do I reset my Instagram password?",
    "What is the price of Bitcoin today?",
    "Who is the lead singer of the band Coldplay?",
    "What time does the supermarket close on Sunday?",
]

STOPWORDS = set("""a an the of to in on for and or is are was were be by with as at from that
this which what who whom how why when where does do did it its into than then there their
following best most following statement statements""".split())


def reword(question: str, rng: random.Random) -> str:
    """Paraphrase with the LLM if available, else a keyword shuffle."""
    if llm.is_configured():
        try:
            return llm.chat(
                "Rewrite the student's question with different wording but the same meaning. "
                "Do not answer it. Return only the rewritten question.",
                question,
            )
        except Exception:
            pass
    words = [w for w in re.findall(r"\w+", question) if w.lower() not in STOPWORDS]
    head, tail = words[: len(words) // 2], words[len(words) // 2 :]
    rng.shuffle(head)
    return " ".join(tail + head)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also generate full answers")
    args = parser.parse_args()

    rng = random.Random(SEED)
    df = pd.read_json(config.CLEAN_JSONL, lines=True).fillna("")
    sample = pd.concat([g.sample(min(len(g), SAMPLES_PER_SUBJECT), random_state=SEED)
                        for _, g in df.groupby("category")])
    print(f"Testing {len(sample)} dataset questions across {sample['category'].nunique()} subjects\n")

    rows = []
    for _, r in sample.iterrows():
        for group, query in [("A. exact", r["question"]), ("B. reworded", reword(r["question"], rng))]:
            hits = retrieve(query, k=config.TOP_K, max_distance=2.0)  # no threshold: measure ranking
            ids = [h.id for h in hits]
            rows.append({
                "group": group, "category": r["category"], "expected_id": r["id"],
                "query": query[:300],
                "top1_correct": bool(ids) and ids[0] == r["id"],
                f"in_top{config.TOP_K}": r["id"] in ids,
                "top1_similarity": hits[0].similarity if hits else 0,
                "passes_threshold": bool(hits) and (1 - hits[0].similarity) <= config.MAX_DISTANCE,
            })

    for q in OUT_OF_SCOPE:
        hits = retrieve(q, k=1, max_distance=2.0)
        sim = hits[0].similarity if hits else 0
        rows.append({
            "group": "C. out of scope", "category": "-", "expected_id": "-", "query": q,
            "top1_similarity": sim,
            # Correct behaviour here is to be REJECTED by the threshold.
            "passes_threshold": (1 - sim) <= config.MAX_DISTANCE,
        })

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "evaluation.csv", index=False)

    topk = f"in_top{config.TOP_K}"
    in_scope = res[res["group"] != "C. out of scope"]
    print("=== A & B: retrieval accuracy ===")
    print(in_scope.groupby("group")[["top1_correct", topk, "passes_threshold", "top1_similarity"]]
          .mean().round(3).to_string())

    oos = res[res["group"] == "C. out of scope"]
    print(f"\n=== C: out-of-scope questions correctly rejected: "
          f"{(~oos['passes_threshold']).mean():.0%} ===")
    print(oos[["query", "top1_similarity", "passes_threshold"]].to_string(index=False))

    print("\n=== D: top-1 accuracy by subject (exact + reworded) ===")
    print(in_scope.groupby("category")["top1_correct"].mean().round(2).sort_values().to_string())

    print("\nTip: if in-scope questions fail the threshold, raise MAX_DISTANCE; if "
          "out-of-scope ones pass, lower it. Similarity = 1 - distance.")
    print(f"Saved {RESULTS_DIR / 'evaluation.csv'}")

    if args.answers:
        lines = ["# Sample answers\n"]
        examples = [sample.iloc[0]["question"], reword(sample.iloc[1]["question"], rng), OUT_OF_SCOPE[0]]
        for q in examples:
            result = answer_question(q)
            lines.append(f"## Q: {q}\n\n{result.answer}\n")
            lines += [f"- [{s.rank}] `{s.id}` ({s.category}, sim {s.similarity})" for s in result.sources]
            lines.append("")
        (RESULTS_DIR / "sample_answers.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved {RESULTS_DIR / 'sample_answers.md'}")


if __name__ == "__main__":
    main()
