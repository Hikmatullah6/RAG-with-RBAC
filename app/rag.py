"""RAG with RBAC: retrieve allowed chunks -> build a prompt -> ask the LLM.
Run:  python rag.py
"""
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from guardrails import (
    can_see_pii, check_input, check_relevance, get_injection_classifier, redact_pii,
)
from rbac import find_department
from retrieve import get_vectorstore, retrieve

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
SHOW_PROMPT = False  # set True to print exactly what the LLM receives

NO_ACCESS_MESSAGE = "You don't have access to information on that, or it doesn't exist."

SYSTEM_PROMPT = """You are an internal assistant for FinSolve Technologies.
Answer the question using ONLY the numbered context provided.
- If the context does not contain the answer, say: "I don't know based on the available documents."
  In that case, do not cite any sources.
- Otherwise, cite the context numbers you used, like [1] or [2][4].
- Each chunk starts with its section path in square brackets (for example, which quarter it is from).
  Use it to make sure you are answering about the right period or topic."""


def get_llm():
    return ChatGroq(model=MODEL_NAME, temperature=0)  # 0 = consistent, factual answers


def build_context(results, mask_pii: bool) -> str:
    """Turn retrieve() results into one numbered block of text."""
    parts = []
    for i, (doc, score) in enumerate(results, start=1):  # retrieve() returns (doc, score) pairs
        text = redact_pii(doc.page_content) if mask_pii else doc.page_content
        parts.append(f"[{i}] (source: {doc.metadata['source']})\n{text}")
    return "\n---\n".join(parts)


def answer(question: str, vectorstore, llm, classifier, role: str):
    """Returns (reply_text, results). Same shape on every path, so callers need no special cases."""
    # 1. Input guard: block prompt-injection attempts before touching any data
    check = check_input(question, classifier)
    if not check.allowed:
        return check.reason, []

    # 2. RBAC: only search departments this role may see
    results = retrieve(vectorstore, question, find_department(role))
    if not results:
        return NO_ACCESS_MESSAGE, []

    # 3. Relevance guard: nothing close enough -> out of scope, skip the LLM call
    check = check_relevance(results)
    if not check.allowed:
        return check.reason, []

    # 4. PII guard, part 1: mask personal data BEFORE the LLM sees it
    mask = not can_see_pii(role)
    context = build_context(results, mask_pii=mask)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    if SHOW_PROMPT:
        print("\n========== PROMPT SENT TO LLM ==========")
        print(f"Context:\n{context}\n\nQuestion: {question}")
        print("========================================\n")

    response = llm.invoke(messages)
    # 5. PII guard, part 2: mask the answer too, as a backstop
    reply = redact_pii(response.content) if mask else response.content
    return reply, results


if __name__ == "__main__":
    # Check the role first, so a bad role fails fast before the models load
    role = input("What is your role: ")
    departments = find_department(role)
    if not departments:
        raise SystemExit("Role was not found.")  # prints the message and exits with an error code

    print("Here are the departments you can access:", departments)

    # Load once, then reuse for every question
    vectorstore = get_vectorstore()
    llm = get_llm()
    classifier = get_injection_classifier()
    print("Ask a question (or 'quit').")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        reply, results = answer(question, vectorstore, llm, classifier, role)
        print(f"Answer:\n{reply}\n")
        if results:
            print("Sources:")
            for i, (doc, score) in enumerate(results, start=1):
                section = doc.metadata.get("section", "")
                print(f"  [{i}] {doc.metadata['source']}  {section}  (score {score:.3f})")