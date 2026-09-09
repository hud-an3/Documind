
import json
import time
import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "source_filter" not in st.session_state:
    st.session_state.source_filter = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post(path: str, **kwargs):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=60, **kwargs)
        return r
    except Exception as e:
        return None


def api_delete(path: str):
    try:
        r = requests.delete(f"{API_URL}{path}", timeout=10)
        return r.ok
    except Exception:
        return False


def format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = []
    for s in sources:
        method_badge = {
            "llamaindex": "📄",
            "aws_rekognition": "👁️ AWS",
            "gcp_vision": "👁️ GCP",
            "web_crawl": "🌐",
            "mock_vision": "🧪",
        }.get(s.get("ingestion_method", ""), "📎")
        lines.append(f"{method_badge} **{s['source']}** · page {s['page']}")
    return "\n".join(lines)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 DocuMind")
    st.caption(f"Session: `{st.session_state.session_id}`")

    # Health check
    health = api_get("/health")
    if health:
        st.success(f"API online · {health.get('vector_store', '?')} · {health.get('vision_provider', '?')}")
    else:
        st.error("⚠️ API offline — run: `uvicorn api.main:app --reload`")

    st.divider()

    # ── Upload file ───────────────────────────────────────────────────────
    st.markdown("### 📁 Upload Document")
    uploaded = st.file_uploader(
        "PDF, DOCX, TXT, MD, JPG, PNG",
        type=["pdf", "docx", "txt", "md", "jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed",
    )
    if uploaded:
        if st.button("⬆️ Ingest File", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded.name}…"):
                r = api_post("/ingest", files={"file": (uploaded.name, uploaded.getvalue())})
            if r and r.ok:
                d = r.json()
                st.success(
                    f"✅ **{d['filename']}**\n\n"
                    f"{d['chunks_created']} chunks · {d['avg_chunk_words']} words/chunk · "
                    f"{d['duration_seconds']}s"
                )
                st.rerun()
            else:
                detail = r.json().get("detail") if r else "API unreachable"
                st.error(f"❌ {detail}")

    st.divider()

    # ── Ingest URL ────────────────────────────────────────────────────────
    st.markdown("### 🌐 Ingest Web URL")
    url_input = st.text_input("https://…", label_visibility="collapsed", placeholder="https://example.com/report")
    if st.button("⬆️ Ingest URL", use_container_width=True) and url_input:
        with st.spinner(f"Crawling {url_input}…"):
            r = api_post("/ingest/url", json={"url": url_input})
        if r and r.ok:
            d = r.json()
            st.success(f"✅ **{d['filename']}** · {d['chunks_created']} chunks")
            st.rerun()
        else:
            detail = r.json().get("detail") if r else "Failed"
            st.error(f"❌ {detail}")

    st.divider()

    # ── Ingested documents ────────────────────────────────────────────────
    st.markdown("### 📚 Ingested Documents")
    sources = api_get("/sources") or []

    if not sources:
        st.caption("No documents yet. Upload one above.")
    else:
        # Source filter selector
        all_option = "🔍 All documents"
        options = [all_option] + [s["source"] for s in sources]
        selected = st.selectbox("Search scope", options, label_visibility="collapsed")
        st.session_state.source_filter = None if selected == all_option else selected

        st.markdown("---")
        for s in sources:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 **{s['source']}**  \n`{s['chunks']} chunks`")
            if col2.button("🗑️", key=f"del_{s['source']}", help=f"Delete {s['source']}"):
                if api_delete(f"/document/{s['source']}"):
                    st.success(f"Deleted {s['source']}")
                    st.rerun()

    st.divider()

    # ── Session ───────────────────────────────────────────────────────────
    st.markdown("### 💬 Conversation")
    col1, col2 = st.columns(2)
    if col1.button("🆕 New chat", use_container_width=True):
        api_delete(f"/session/{st.session_state.session_id}")
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    stats = api_get("/stats")
    if stats and col2.button("📊 Stats", use_container_width=True):
        st.session_state.show_stats = not st.session_state.get("show_stats", False)

    if st.session_state.get("show_stats") and stats:
        st.json(stats)


# ── Main chat area ─────────────────────────────────────────────────────────────

scope_label = f"📄 `{st.session_state.source_filter}`" if st.session_state.source_filter else "📚 All documents"
st.markdown(f"### 🧠 DocuMind &nbsp;&nbsp; <small>{scope_label}</small>", unsafe_allow_html=True)

if not sources:
    st.info("👈 Upload a document in the sidebar to get started.")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                for s in msg["sources"]:
                    badge = {"llamaindex": "📄", "aws_rekognition": "👁️ AWS", "gcp_vision": "👁️ GCP",
                             "web_crawl": "🌐", "mock_vision": "🧪"}.get(s.get("ingestion_method", ""), "📎")
                    st.markdown(f"{badge} **{s['source']}** · page {s['page']}")
                    st.caption(s["excerpt"])

# Chat input
if question := st.chat_input(
    f"Ask about {st.session_state.source_filter or 'your documents'}…",
    disabled=not bool(sources),
):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking…"):
            r = api_post(
                "/query",
                json={
                    "question": question,
                    "session_id": st.session_state.session_id,
                    "source_filter": st.session_state.source_filter,
                },
            )
        if r and r.ok:
            data = r.json()
            placeholder.markdown(data["answer"])
            sources_used = data.get("sources", [])
            if sources_used:
                with st.expander(f"📎 {len(sources_used)} source(s) · {data['duration_seconds']}s"):
                    for s in sources_used:
                        badge = {"llamaindex": "📄", "aws_rekognition": "👁️ AWS", "gcp_vision": "👁️ GCP",
                                 "web_crawl": "🌐", "mock_vision": "🧪"}.get(s.get("ingestion_method", ""), "📎")
                        st.markdown(f"{badge} **{s['source']}** · page {s['page']}")
                        st.caption(s["excerpt"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": sources_used,
            })
        else:
            err = r.json().get("detail", "Unknown error") if r else "API unreachable"
            placeholder.error(f"❌ {err}")
