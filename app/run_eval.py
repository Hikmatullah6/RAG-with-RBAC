"""Score answer quality with Ragas.

Run from the app folder:
    python run_eval.py            (all test cases)
    python run_eval.py --limit 3  (quick check)

Results are saved to eval/results/<timestamp>.csv so you can compare runs.
"""
import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")  # Ragas prints a lot of deprecation noise

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from guardrails import can_see_pii, get_injection_classifier, redact_pii
from ingest import get_embeddings
from rag import answer, get_llm
from retrieve import get_vectorstore

load_dotenv()

EVAL_DIR = Path(__file__).resolve().parent / "eval"
JUDGE_MODEL = "openai/gpt-oss-120b"  # the grader; use a stronger model than the chatbot if you can
SECONDS_BETWEEN_QUESTIONS = 2        # stay under Groq's free-tier rate limits

METRICS = [
    Faithfulness(),                      # is the answer supported by the retrieved chunks?
    ResponseRelevancy(),                 # does it actually answer the question?
    LLMContextPrecisionWithReference(),  # were the useful chunks ranked near the top?
    LLMContextRecall(),                  # did retrieval find everything needed for the reference answer?
]


def run_chatbot(cases):
    """Ask every test question as its role and record what happened."""
    vectorstore, llm, classifier = get_vectorstore(), get_llm(), get_injection_classifier()
    samples = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] ({case['role']}) {case['question']}")
        reply, results = answer(case["question"], vectorstore, llm, classifier, case["role"])
        # Grade against the same (masked) text the LLM actually saw
        mask = not can_see_pii(case["role"])
        contexts = [redact_pii(doc.page_content) if mask else doc.page_content for doc, _ in results]
        samples.append({
            "user_input": case["question"],
            "response": reply,
            "retrieved_contexts": contexts,
            "reference": case["reference"],
        })
        time.sleep(SECONDS_BETWEEN_QUESTIONS)
    return samples


def score(samples):
    judge = LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=METRICS,
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=2, timeout=180, max_retries=6),  # gentle on rate limits
        raise_exceptions=False,  # a failed grading shows as NaN instead of crashing the run
    )
    return result.to_pandas()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only run the first N test cases")
    args = parser.parse_args()

    cases = json.loads((EVAL_DIR / "test_cases.json").read_text(encoding="utf-8"))[: args.limit]
    df = score(run_chatbot(cases))

    # Attach id and role so results can be grouped and traced back
    df.insert(0, "role", [c["role"] for c in cases])
    df.insert(0, "id", [c["id"] for c in cases])

    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{datetime.now():%Y-%m-%d_%H-%M}.csv"
    df.to_csv(out_file, index=False)

    metric_cols = [m.name for m in METRICS if m.name in df.columns]
    print("\n=== Average scores (0 to 1, higher is better) ===")
    print(df[metric_cols].mean().round(3).to_string())
    print("\n=== By role ===")
    print(df.groupby("role")[metric_cols].mean().round(2).to_string())
    print("\n=== Weakest answers (lowest faithfulness) ===")
    print(df.sort_values("faithfulness")[["id", "faithfulness", "response"]].head(3).to_string(index=False))
    print(f"\nFull results saved to {out_file}")