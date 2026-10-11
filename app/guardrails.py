"""Guardrails around the RAG pipeline.

question -> [input guard] -> retrieve -> [relevance guard] -> LLM -> [PII guard] -> answer
"""
import re
from dataclasses import dataclass

from langchain_groq import ChatGroq

# ---------------- Settings (tune these with real questions) ----------------
INJECTION_MODEL = "meta-llama/llama-prompt-guard-2-22m"
INJECTION_THRESHOLD = 0.5      # attack probability at or above this -> block
RELEVANCE_THRESHOLD = 1.3      # best distance above this -> out of scope (good matches were ~0.6-1.0)
MAX_QUESTION_LENGTH = 1000     # very long inputs are a common way to smuggle in instructions
PII_ALLOWED_ROLES: set[str] = set()  # roles allowed to see unmasked PII; e.g. {"hr"} if HR needs contact details

OUT_OF_SCOPE_MESSAGE = "That's outside what I can help with. I answer questions about FinSolve's internal documents."
BLOCKED_MESSAGE = "This request was blocked by a safety check. Please rephrase your question."


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""   # message shown to the user when blocked


# ---------------- 1. Input guard: prompt injection ----------------
def get_injection_classifier():
    return ChatGroq(model=INJECTION_MODEL, temperature=0)


def check_input(question: str, classifier) -> GuardResult:
    if len(question) > MAX_QUESTION_LENGTH:
        return GuardResult(False, f"Please keep questions under {MAX_QUESTION_LENGTH} characters.")
    try:
        # Prompt Guard replies with just a number: the probability the text is an attack
        score = float(classifier.invoke(question).content.strip())
    except Exception:
        # Fail closed: if the safety check can't run, don't let the question through
        return GuardResult(False, "The safety check is unavailable right now. Please try again.")
    if score >= INJECTION_THRESHOLD:
        return GuardResult(False, BLOCKED_MESSAGE)
    return GuardResult(True)


# ---------------- 2. Relevance guard: out-of-scope questions ----------------
def check_relevance(results) -> GuardResult:
    """results = (doc, distance) pairs. Lower distance = closer match."""
    best = min(score for _, score in results)
    if best > RELEVANCE_THRESHOLD:
        return GuardResult(False, OUT_OF_SCOPE_MESSAGE)
    return GuardResult(True)


# ---------------- 3. PII guard: mask personal data ----------------
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Runs of digits with phone-style separators (spaces, dashes, dots, brackets, +).
# Commas are NOT allowed, so amounts like 1,129,637.9 are left alone.
PHONE_CANDIDATE = re.compile(r"\+?\(?\d[\d\s().-]{7,}\d")


def _mask_phone(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group())
    return "[REDACTED PHONE]" if len(digits) >= 10 else match.group()  # 10+ digits = phone number


def redact_pii(text: str) -> str:
    text = EMAIL_PATTERN.sub("[REDACTED EMAIL]", text)
    text = PHONE_CANDIDATE.sub(_mask_phone, text)
    return text


def can_see_pii(role: str) -> bool:
    return role.strip().lower() in PII_ALLOWED_ROLES


if __name__ == "__main__":
    # Quick self-test for the PII guard (no API calls)
    samples = [
        "Contact jane.doe@finsolve.com or +91 98765 43210.",
        "Phone: (416) 555-0199",
        "Salary is 1,129,637.9 and Q2 vendor spend was $125 million.",
        "Joined on 2021-03-15.",
    ]
    for s in samples:
        print(f"{s}\n  -> {redact_pii(s)}")