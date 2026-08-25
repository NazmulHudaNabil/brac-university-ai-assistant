"""
BRAC University AI Assistant — Clean, Simple & Professional Streamlit UI.
"""

import os
import sys
import uuid
import json
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="BRAC University AI Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="auto",
)


# ---------------------------------------------------------------------------
# Backend Invoker (Thread-isolated to support both venv and system python)
# ---------------------------------------------------------------------------
def call_backend_chat(query: str, conversation_id: str, chat_history: list = None):
    """Calls the running FastAPI backend over HTTP."""
    api_url = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000/chat")
    payload = {
        "query": query,
        "conversation_id": conversation_id,
        "chat_history": chat_history or []
    }
    
    try:
        import requests
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise RuntimeError(f"Failed to connect to backend at {api_url}: {str(e)}\n\nMake sure the FastAPI server is running: `python3 -m app.main`")

    class SimpleCitation:
        def __init__(self, **kw):
            self.title = kw.get("title") or "BRAC University Document"
            self.url = kw.get("url")
            self.snippet = kw.get("snippet", "")

    class SimpleResponse:
        def __init__(self, answer, sources, request_id):
            self.answer = answer
            self.sources = [SimpleCitation(**s) for s in sources]
            self.request_id = request_id

    return SimpleResponse(
        answer=data.get("answer", ""),
        sources=data.get("sources", []),
        request_id=data.get("request_id", str(uuid.uuid4()))
    )


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"session_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 BRACU Assistant")
    st.caption("AI-powered campus intelligence assistant.")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.conversation_id = f"session_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("💡 Example Questions")

    example_questions = [
        "What GPA is required for O-Level and A-Level applicants?",
        "CSE প্রোগ্রামে ভর্তির যোগ্যতা কী?",
        "What programs does BRAC Business School offer?",
        "ব্র্যাক ইউনিভার্সিটিতে কি স্কলারশিপ বা ফাইন্যান্সিয়াল এইড পাওয়া যায়?",
        "Who serves as the Chancellor of BRAC University?",
        "কোন পরিস্থিতিতে একজন শিক্ষার্থী কোর্স ড্রপ করতে পারে?"
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    st.markdown("---")
    st.subheader("📊 Observability")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("Logfire", "https://logfire-us.pydantic.dev/nazmulhudanabil/marathon", use_container_width=True)
    with col2:
        st.link_button("LangSmith", "https://smith.langchain.com/o/default/projects/p/brac_university_ai_assistant", use_container_width=True)


# ---------------------------------------------------------------------------
# Main Chat Header
# ---------------------------------------------------------------------------
st.title("🎓 BRAC University AI Assistant")
st.caption("Ask questions about admissions, courses, fees, governance, and campus facilities.")

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    title = src.get("title") or f"Document #{i}"
                    url = src.get("url")
                    snippet = src.get("snippet", "")
                    
                    link_text = f" — [View Page]({url})" if url else ""
                    st.markdown(f"**[{i}] {title}**{link_text}")
                    if snippet:
                        st.caption(f'"{snippet}"')


# ---------------------------------------------------------------------------
# Chat Input & Response Generation
# ---------------------------------------------------------------------------
prompt = st.chat_input("Ask a question about BRAC University...")

# Check if an example question was clicked
if hasattr(st.session_state, "pending_question") and st.session_state.pending_question:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None

if prompt:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate and Display Assistant Response directly
    with st.chat_message("assistant"):
        with st.spinner("Searching BRAC University knowledge base..."):
            try:
                prior_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                response = call_backend_chat(
                    prompt,
                    st.session_state.conversation_id,
                    chat_history=prior_history
                )
                answer = response.answer
                sources = [
                    {"title": s.title, "url": s.url, "snippet": s.snippet}
                    for s in response.sources
                ]
            except Exception as e:
                answer = f"Sorry, an error occurred while processing your request: {str(e)}"
                sources = []

        # Display answer directly in the chat message (NOT inside any status container)
        st.markdown(answer)

        # Display sources
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
                for i, src in enumerate(sources, 1):
                    title = src.get("title") or f"Document #{i}"
                    url = src.get("url")
                    snippet = src.get("snippet", "")

                    link_text = f" — [View Page]({url})" if url else ""
                    st.markdown(f"**[{i}] {title}**{link_text}")
                    if snippet:
                        st.caption(f'"{snippet}"')

        # Save to session messages
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
