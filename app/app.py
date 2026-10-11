"""Streamlit chat UI with login.
Run:  streamlit run app.py
"""
import streamlit as st

from config import CHROMA_DIR
from rag import answer, get_llm
from rbac import USERS, authenticate, find_department
from retrieve import get_vectorstore

st.set_page_config(page_title="FinSolve Assistant", page_icon="💬")


# Streamlit reruns this whole file on every interaction.
# cache_resource makes these load ONCE and be reused on every rerun.
@st.cache_resource(show_spinner="Loading models...")
def load_resources():
    return get_vectorstore(), get_llm()


def show_sources(results):
    with st.expander(f"Sources ({len(results)})"):
        for i, (doc, score) in enumerate(results, start=1):
            section = doc.metadata.get("section", "")
            st.caption(f"[{i}] {doc.metadata['department']} / {doc.metadata['source']} — {section} (score {score:.3f})")


# ---------------- Login screen ----------------
if "user" not in st.session_state:
    st.title("💬 FinSolve Assistant")

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        user = authenticate(username, password)
        if user:
            st.session_state.user = user
            st.session_state.messages = []  # fresh chat for each login
            st.rerun()
        else:
            st.error("Invalid username or password.")

    with st.expander("Demo accounts (password: demo123)"):
        for name, info in USERS.items():
            st.write(f"**{name}** — {info['role']}")

    st.stop()  # nothing below runs until someone is logged in


# ---------------- Logged in ----------------
user = st.session_state.user
departments = find_department(user["role"])  # recomputed from the role on every rerun

with st.sidebar:
    st.write(f"**User:** {user['username']}")
    st.write(f"**Role:** {user['role']}")
    st.write("**Can access:** " + ", ".join(departments))
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()

st.title("💬 FinSolve Assistant")

if not CHROMA_DIR.exists():
    st.error("No database found. Run `python ingest.py` first.")
    st.stop()

vectorstore, llm = load_resources()

# Redraw the conversation so far (the page is rebuilt on every rerun)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

# Handle a new question
if question := st.chat_input("Ask about company documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching..."):
                reply, results = answer(question, vectorstore, llm, departments)
        except Exception as e:  # e.g. Groq rate limit or network error
            reply, results = f"Something went wrong: {e}", []
        st.markdown(reply)
        if results:
            show_sources(results)

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": results})