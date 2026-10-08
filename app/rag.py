"""Basic RAG: retrieve chunks -> build a prompt with them -> ask the LLM.
Run:  python rag.py
"""
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from retrieve import get_vectorstore, retrieve
from rbac import find_department

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
SHOW_PROMPT = False  # print exactly what the LLM receives; set False once you trust it

SYSTEM_PROMPT = """You are an internal assistant for FinSolve Technologies.
Answer the question using ONLY the numbered context provided.
- If the context does not contain the answer, say: "I don't know based on the available documents."
- Cite the context numbers you used, like [1] or [2][4].
- Each chunk starts with its section path in square brackets (for example, which quarter it is from).
  Use it to make sure you are answering about the right period or topic."""


def get_llm():
    return ChatGroq(model=MODEL_NAME, temperature=0)  # 0 = consistent, factual answers


def build_context(results) -> str:
    """Turn retrieve() results into one numbered block of text."""
    parts = []
    for i, (doc, score) in enumerate(results, start=1):  # retrieve() returns (doc, score) pairs
        parts.append(f"[{i}] (source: {doc.metadata['source']})\n{doc.page_content}")
    return "\n---\n".join(parts)  # join the whole list once


def answer(question: str, vectorstore, llm, departments: list):
    results = retrieve(vectorstore, question, 5, departments)
    if len(results) == 0:
        response = "No Access"
        return response, results
    
    context = build_context(results)

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    if SHOW_PROMPT:
        print("\n========== PROMPT SENT TO LLM ==========")
        print(f"Context:\n{context}\n\nQuestion: {question}")
        print("========================================\n")

    response = llm.invoke(messages)
    return response.content, results


if __name__ == "__main__":
    # Load once, then reuse for every question (much faster than reloading each time)
    vectorstore = get_vectorstore()
    llm = get_llm()
    
    role = input("What is your role: ")
    departments = find_department(role)
    if len(departments) == 0:
        print("Role was not found. ")
        sys.exit(0)
    
    print("Here are the departments you can access:", departments)
    print("Ask a question (or 'quit').")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        reply, results = answer(question, vectorstore, llm, departments)
        print(f"Answer:\n{reply}\n")
        print("Sources:")
        for i, (doc, score) in enumerate(results, start=1):
            section = doc.metadata.get("section", "")
            print(f"  [{i}] {doc.metadata['source']}  {section}  (score {score:.3f})")