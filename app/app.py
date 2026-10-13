"""Streamlit chat UI with login, plus a usage dashboard for C-level users.
Run:  streamlit run app.py
"""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from config import CHROMA_DIR
from guardrails import get_injection_classifier
from monitoring import DAILY_BUDGET_USD, USER_DAILY_QUESTION_LIMIT, questions_today, read_logs
from rag import answer, get_llm
from rbac import USERS, authenticate, find_department
from retrieve import get_vectorstore

st.set_page_config(page_title="FinSolve Assistant", page_icon="💬")


# Streamlit reruns this whole file on every interaction.
# cache_resource makes these load ONCE and be reused on every rerun.
@st.cache_resource(show_spinner="Loading models...")
def load_resources():
    return get_vectorstore(), get_llm(), get_injection_classifier()


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
is_c_level = user["role"] == "c_level"

with st.sidebar:
    st.write(f"**User:** {user['username']}")
    st.write(f"**Role:** {user['role']}")
    st.write("**Can access:** " + ", ".join(departments))
    st.caption(f"Questions today: {questions_today(user['username'])} / {USER_DAILY_QUESTION_LIMIT}")
    page = st.radio("Page", ["Chat", "Usage dashboard"]) if is_c_level else "Chat"
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()


# ---------------- Usage dashboard (C-level only) ----------------
def usage_dashboard():
    st.title("📊 Usage dashboard")
    logs = pd.DataFrame(read_logs())
    if logs.empty:
        st.info("No usage yet. Ask a few questions first.")
        return
    logs["day"] = logs["time"].str[:10]
    today = logs[logs["day"] == date.today().isoformat()]

    cost_today = today["cost_usd"].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Cost today", f"${cost_today:.4f}", help=f"Daily budget: ${DAILY_BUDGET_USD:.2f}")
    col2.metric("Questions today", len(today))
    col3.metric("Avg latency today", f"{today['latency_s'].mean():.1f}s" if len(today) else "-")
    if cost_today > DAILY_BUDGET_USD:
        st.error(f"Over budget: ${cost_today:.4f} spent today (budget ${DAILY_BUDGET_USD:.2f}).")
    else:
        st.progress(min(cost_today / DAILY_BUDGET_USD, 1.0), text="Share of daily budget used")

    st.subheader("Daily cost (last 14 days)")
    last_14 = [(date.today() - timedelta(days=d)).isoformat() for d in range(13, -1, -1)]
    daily = logs.groupby("day")["cost_usd"].sum().reindex(last_14, fill_value=0)
    st.bar_chart(daily)

    st.subheader("Today by role")
    if len(today):
        by_role = today.groupby("role").agg(questions=("question", "count"), cost_usd=("cost_usd", "sum"))
        st.dataframe(by_role, width="stretch")

    st.subheader("Outcomes today (how often each guard fired)")
    if len(today):
        st.bar_chart(today["outcome"].value_counts())

    st.subheader("Most expensive questions")
    top = logs.sort_values("cost_usd", ascending=False).head(5)
    st.dataframe(top[["time", "user", "role", "question", "tokens_in", "tokens_out", "cost_usd"]],
                 hide_index=True, width="stretch")


if page == "Usage dashboard":
    usage_dashboard()
    st.stop()


# ---------------- Chat ----------------
st.title("💬 FinSolve Assistant")

if not CHROMA_DIR.exists():
    st.error("No database found. Run `python ingest.py` first.")
    st.stop()

vectorstore, llm, classifier = load_resources()

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
                reply, results = answer(question, vectorstore, llm, classifier,
                                        user["role"], username=user["username"])
        except Exception as e:  # e.g. Groq rate limit or network error
            reply, results = f"Something went wrong: {e}", []
        st.markdown(reply)
        if results:
            show_sources(results)

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": results})