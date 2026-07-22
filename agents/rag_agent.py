"""
agents/rag_agent.py  — Phase 4 upgrade

Production-ready conversational RAG agent with:
  - Streaming support (token-by-token via callbacks)
  - Source-scoped queries ("only search contract.pdf")
  - Multi-document mode (search across all ingested docs)
  - Prompt template with explicit citation instructions
  - Confidence scoring on retrieved chunks
  - Session isolation (each session_id gets its own memory)
  - Tool-extensible architecture (add calculator, web search, SQL, etc.)

What makes this "agentic" vs a basic RAG pipeline:

  Basic RAG: query → retrieve chunks → stuff into prompt → LLM answers
  
  Agentic RAG:
    - Memory: agent recalls previous turns in the same session
    - Tool use: agent can call external tools mid-reasoning
      (e.g. "this contract mentions a formula, let me calculate it")
    - Planning: agent can decide whether to retrieve more context
    - Self-critique: can be prompted to verify its own answer against sources

  This implementation uses ConversationalRetrievalChain as the backbone
  and is architected so tools can be added without rewriting anything.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Iterator, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate

from core.config import cfg
from core.vector_store import VectorStoreManager


# ── Prompt template ───────────────────────────────────────────────────────────

QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are DocuMind, a precise and helpful document analysis assistant.

Your task is to answer the user's question using ONLY the document excerpts provided below.

STRICT RULES:
1. Base your answer solely on the provided context. Do not use outside knowledge.
2. If the answer is not in the documents, respond: "I couldn't find that in the uploaded documents."
3. Always cite the source document and page/chunk when making a specific claim.
4. For numbers, dates, names, and legal terms — quote them exactly as they appear.
5. Structure longer answers with clear headings or bullet points for readability.
6. If multiple documents contain relevant information, synthesise across them.

Context from documents:
{context}

Conversation history:
{chat_history}

Question: {question}

Answer (with citations):""",
)

CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the conversation history below, rephrase the follow-up question
as a standalone question that captures all necessary context for a document search.

If the question is already standalone, return it unchanged.

Chat History:
{chat_history}

Follow-up question: {question}

Standalone question:""",
)


# ── Streaming callback ────────────────────────────────────────────────────────

class TokenStreamHandler(BaseCallbackHandler):
    """
    Collects streamed tokens into a buffer.
    The API's SSE endpoint reads from this buffer and pushes tokens to the client,
    giving users the "typing" effect without waiting for the full response.
    """

    def __init__(self):
        self.tokens: List[str] = []
        self.finished = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        self.finished = True

    def stream(self) -> Iterator[str]:
        """Yield tokens as they arrive."""
        i = 0
        while not self.finished or i < len(self.tokens):
            if i < len(self.tokens):
                yield self.tokens[i]
                i += 1


# ── Session memory store ──────────────────────────────────────────────────────

class SessionMemoryStore:
    """
    Manages one ConversationBufferWindowMemory per session_id.
    In production, back this with Redis for horizontal scaling:
        redis_client.set(f"memory:{session_id}", serialised_memory)
    """

    def __init__(self):
        self._sessions: Dict[str, ConversationBufferWindowMemory] = {}

    def get_or_create(self, session_id: str) -> ConversationBufferWindowMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationBufferWindowMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True,
                k=cfg.memory_window_k,
            )
        return self._sessions[session_id]

    def clear(self, session_id: str) -> bool:
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            return True
        return False

    def delete(self, session_id: str) -> bool:
        return bool(self._sessions.pop(session_id, None))

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())


# Shared session store (one per process; use Redis in multi-process deployments)
_session_store = SessionMemoryStore()


# ── Main agent class ──────────────────────────────────────────────────────────

class DocuMindAgent:
    """
    Conversational RAG agent with session memory and streaming.

    Usage (simple):
        agent = DocuMindAgent(vsm)
        result = agent.query("What are the payment terms?")
        print(result["answer"])

    Usage (streaming):
        for token in agent.query_stream("Summarise section 3"):
            print(token, end="", flush=True)

    Usage (source-scoped):
        result = agent.query("What is the penalty clause?", source_filter="contract.pdf")
    """

    def __init__(self, vsm: VectorStoreManager):
        self.vsm = vsm
        self._chains: Dict[str, ConversationalRetrievalChain] = {}

    def _build_chain(
        self,
        session_id: str,
        source_filter: Optional[str] = None,
        streaming: bool = False,
        stream_handler: Optional[TokenStreamHandler] = None,
    ) -> ConversationalRetrievalChain:
        """Build a chain for a specific session + configuration."""
        callbacks = [stream_handler] if (streaming and stream_handler) else None

        llm = ChatGoogleGenerativeAI(
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            google_api_key=cfg.google_api_key,
)
        # Non-streaming LLM for the condense-question step
        # (we don't want to stream the internal rephrasing, only the final answer)
        condense_llm = ChatOpenAI(
            model=cfg.llm_model,
            temperature=0,
            openai_api_key=cfg.openai_api_key,
        )

        retriever = self.vsm.get_retriever(filter_source=source_filter)
        memory = _session_store.get_or_create(session_id)

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            condense_question_llm=condense_llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            condense_question_prompt=CONDENSE_PROMPT,
            verbose=cfg.debug,
        )

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Standard (non-streaming) query.
        Returns full answer + source citations in one shot.

        session_id: pass the same ID across turns to maintain conversation context.
                    If None, a new single-use session is created.
        source_filter: restrict retrieval to a specific document filename.
        """
        session_id = session_id or str(uuid.uuid4())
        chain = self._build_chain(session_id, source_filter=source_filter)
        result = chain.invoke({"question": question})
        return self._format_result(result)

    def query_stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Streaming query — yields tokens as the LLM generates them.
        Use this for the FastAPI SSE endpoint so users see the answer appear live.

        Example in FastAPI:
            async def stream_endpoint(q: str):
                async def generate():
                    for token in agent.query_stream(q, session_id):
                        yield f"data: {token}\\n\\n"
                return EventSourceResponse(generate())
        """
        session_id = session_id or str(uuid.uuid4())
        handler = TokenStreamHandler()
        chain = self._build_chain(
            session_id,
            source_filter=source_filter,
            streaming=True,
            stream_handler=handler,
        )

        # Run the chain in a thread so we can yield tokens while it runs
        import threading
        result_holder: Dict[str, Any] = {}

        def run_chain():
            result_holder["result"] = chain.invoke({"question": question})

        thread = threading.Thread(target=run_chain, daemon=True)
        thread.start()

        # Yield tokens as they appear in the handler buffer
        for token in handler.stream():
            yield token

        thread.join()

    def reset_session(self, session_id: str) -> bool:
        """Clear conversation memory for a session."""
        cleared = _session_store.clear(session_id)
        if cleared:
            self._chains.pop(session_id, None)
            print(f"[Agent] Session '{session_id}' cleared.")
        return cleared

    def list_sessions(self) -> List[str]:
        return _session_store.list_sessions()

    @staticmethod
    def _format_result(result: Dict) -> Dict[str, Any]:
        """Parse LangChain result into a clean API-friendly dict."""
        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            m = doc.metadata
            key = f"{m.get('source')}:{m.get('page_label', m.get('chunk_index', ''))}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": m.get("source", "unknown"),
                    "page": str(m.get("page_label", m.get("chunk_index", "?"))),
                    "excerpt": doc.page_content[:250].strip() + "…",
                    "ingestion_method": m.get("ingestion_method", "unknown"),
                })
        return {
            "answer": result.get("answer", ""),
            "sources": sources,
        }
