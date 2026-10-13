"""Usage logging, cost tracking, budget alerts and per-user limits.

Every question writes one line (JSON) to logs/usage.jsonl:
  time, user, role, question, outcome, tokens, cost, latency
"""
import json
import logging
import threading
import time
from datetime import date, datetime
from pathlib import Path

from config import APP_DIR

LOG_FILE = APP_DIR.parent / "logs" / "usage.jsonl"   # RAG-with-RBAC/logs/usage.jsonl

DAILY_BUDGET_USD = 1.00          # alert when today's total cost goes over this
USER_DAILY_QUESTION_LIMIT = 50   # per user per day; caps cost and slows down abuse
UNLIMITED_USERS = {"eval"}       # the evaluation script asks many questions on purpose

# USD per 1 million tokens: (input, output).
# Check https://groq.com/pricing and update these; prices change.
PRICES = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "meta-llama/llama-prompt-guard-2-22m": (0.03, 0.03),
}

logger = logging.getLogger("rag.monitoring")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
_write_lock = threading.Lock()  # Streamlit can handle several users at once


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    if model not in PRICES:
        logger.warning("No price set for model %s; counting its cost as $0", model)
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


class RequestTracker:
    """Collects everything about one question, then saves it as one log line."""

    def __init__(self, username: str, role: str, question: str):
        self.start = time.perf_counter()
        self.record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "user": username,
            "role": role,
            "question": question,
            "outcome": "unknown",   # answered / blocked_injection / no_access / out_of_scope / rate_limited / error
            "llm_calls": [],
        }

    def add_usage(self, model: str, response) -> None:
        """Record tokens from a LangChain response (usage_metadata holds the counts)."""
        usage = getattr(response, "usage_metadata", None) or {}
        tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        self.record["llm_calls"].append({
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost_usd(model, tokens_in, tokens_out),
        })

    def set_outcome(self, outcome: str) -> None:
        self.record["outcome"] = outcome

    def save(self) -> None:
        calls = self.record["llm_calls"]
        self.record["tokens_in"] = sum(c["tokens_in"] for c in calls)
        self.record["tokens_out"] = sum(c["tokens_out"] for c in calls)
        self.record["cost_usd"] = round(sum(c["cost_usd"] for c in calls), 8)
        self.record["latency_s"] = round(time.perf_counter() - self.start, 2)

        spent_before = today_cost()
        LOG_FILE.parent.mkdir(exist_ok=True)
        with _write_lock, LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.record) + "\n")

        # Alert once, at the moment today's spending crosses the budget
        spent_after = spent_before + self.record["cost_usd"]
        if spent_before <= DAILY_BUDGET_USD < spent_after:
            logger.warning("BUDGET ALERT: today's cost $%.4f passed the daily budget of $%.2f",
                           spent_after, DAILY_BUDGET_USD)


# ---------------- Reading the log ----------------
def read_logs(day: date | None = None) -> list[dict]:
    """All log records, or only those from one day."""
    if not LOG_FILE.exists():
        return []
    records = []
    with LOG_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if day:
        records = [r for r in records if r["time"].startswith(day.isoformat())]
    return records


def today_cost() -> float:
    return sum(r.get("cost_usd", 0) for r in read_logs(date.today()))


def questions_today(username: str) -> int:
    return sum(1 for r in read_logs(date.today()) if r["user"] == username)


def over_user_limit(username: str) -> bool:
    if username in UNLIMITED_USERS:
        return False
    return questions_today(username) >= USER_DAILY_QUESTION_LIMIT