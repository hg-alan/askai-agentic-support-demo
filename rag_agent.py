import os
import glob
import json
import uuid
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ---------- API KEY RESOLUTION ----------

def get_api_key() -> str:
    """
    Resolve OPENAI_API_KEY from:
    1) .env / environment (local dev)
    2) Streamlit Cloud secrets (if available)
    """
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")

    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            key = None

    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. "
            "Set it locally in a .env file or in Streamlit Cloud secrets."
        )
    return key


api_key = get_api_key()
client = OpenAI(api_key=api_key)

# ---------- CHROMA SETUP ----------

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
COLLECTION_NAME = "support_kb"
_KB_DOCS_CACHE: List[Tuple[str, str]] = []  # (filename, text)


def _get_or_create_collection():
    existing = {c.name: c for c in chroma_client.list_collections()}
    if COLLECTION_NAME in existing:
        return existing[COLLECTION_NAME]
    return chroma_client.create_collection(name=COLLECTION_NAME)


collection = _get_or_create_collection()

# ---------- KB LOADING / DISPLAY ----------


def load_docs_from_folder(folder_path: str) -> List[Tuple[str, str]]:
    docs = []
    for path in glob.glob(os.path.join(folder_path, "**/*"), recursive=True):
        if os.path.isfile(path) and path.lower().endswith((".md", ".txt")):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    docs.append((os.path.basename(path), text))
    return docs


def get_corpus_markdown() -> str:
    """
    Entire KB as markdown for UI (ground truth view).
    """
    global _KB_DOCS_CACHE
    if not _KB_DOCS_CACHE:
        _KB_DOCS_CACHE = load_docs_from_folder("docs")

    if not _KB_DOCS_CACHE:
        return "_No documentation loaded._"

    parts = []
    for filename, text in _KB_DOCS_CACHE:
        parts.append(f"### {filename}\n\n{text}")
    return "\n\n---\n\n".join(parts)


# ---------- EMBEDDING / INDEXING ----------


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Chunk strategy optimized for small FAQ-style docs.

    1. If the text contains markdown headings (`#`), split into sections:
       each heading + its following lines becomes one chunk.
    2. If no headings found, fall back to a simple sliding window.
    """

    lines = text.splitlines()
    sections: List[str] = []
    current: List[str] = []

    # Section-based chunking for markdown-style docs
    for line in lines:
        if line.strip().startswith("#"):
            # start of a new section
            if current:
                joined = " ".join(x.strip() for x in current if x.strip())
                if joined:
                    sections.append(joined)
                current = []
        if line.strip():  # skip pure blank lines
            current.append(line)
    # flush last
    if current:
        joined = " ".join(x.strip() for x in current if x.strip())
        if joined:
            sections.append(joined)

    # If we successfully built sections, use them
    if sections:
        return sections

    # Fallback: sliding window (for non-markdown/plain text)
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        start = end - overlap
    return chunks

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        start = end - overlap
    return chunks


def get_embeddings(texts: List[str]):
    if isinstance(texts, str):
        texts = [texts]
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in resp.data]


def build_index() -> Tuple[chromadb.api.models.Collection.Collection, int]:
    """
    Rebuild vector index from ./docs. Returns (collection, num_chunks).
    """
    global _KB_DOCS_CACHE
    _KB_DOCS_CACHE = load_docs_from_folder("docs")
    if not _KB_DOCS_CACHE:
        raise RuntimeError("No documents found in ./docs. Add some .md or .txt files.")

    # Drop existing collection for determinism
    for c in chroma_client.list_collections():
        if c.name == COLLECTION_NAME:
            chroma_client.delete_collection(name=COLLECTION_NAME)

    col = chroma_client.create_collection(name=COLLECTION_NAME)

    ids, contents, embeddings = [], [], []
    idx = 0
    for filename, text in _KB_DOCS_CACHE:
        for chunk in chunk_text(text):
            ids.append(f"{filename}-{idx}")
            contents.append(chunk)
            embeddings.append(get_embeddings(chunk)[0])
            idx += 1

    col.add(ids=ids, documents=contents, embeddings=embeddings)

    global collection
    collection = col

    return col, len(ids)


# ---------- RETRIEVAL ----------


def retrieve_top_chunks(query: str, k: int = 4) -> List[str]:
    """
    Get top-k chunks (no over-clever filtering).
    The model will decide if they're sufficient.
    """
    q_emb = get_embeddings(query)[0]
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents"],
    )
    docs = result.get("documents", [[]])[0] or []
    return docs


# ---------- TOOL: ESCALATION ----------


def escalate_ticket(user_question: str, retrieved_context: str) -> Dict[str, Any]:
    ticket_id = str(uuid.uuid4())[:8]
    print(f"\n[ESCALATION] Created ticket {ticket_id} for: {user_question}\n")
    return {
        "ticket_id": ticket_id,
        "status": "created",
        "assigned_team": "Tier-2 Support",
        "note": "Escalated because documentation was insufficient or ambiguous.",
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "escalate_ticket",
            "description": (
                "Use this when documentation does not clearly answer the question, "
                "is missing, or the request is risky/compliance-sensitive."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_question": {"type": "string"},
                    "retrieved_context": {"type": "string"},
                },
                "required": ["user_question", "retrieved_context"],
            },
        },
    }
]


# ---------- MAIN AGENT ----------


def answer_question(question: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    - Retrieve top chunks.
    - Let LLM either answer from docs or call escalate_ticket.
    - Return (final_answer, concatenated_context, meta).
    """
    chunks = retrieve_top_chunks(question, k=2)
    context = "\n\n".join(chunks)

    system_prompt = (
        "You are a precise, trustworthy support agent for an imaginary SaaS company.\n"
        "You MUST base answers only on the provided documentation.\n"
        "If the docs clearly answer, respond concisely using that info.\n"
        "If docs are missing, unclear, or the question is risky/compliance-sensitive, "
        "DO NOT guess: instead CALL the `escalate_ticket` tool.\n"
    )

    context_for_prompt = (
        context if context.strip() else "[NO MATCHING DOCUMENTATION FOUND]"
    )

    user_content = (
        f"User question:\n{question}\n\n"
        f"Retrieved documentation:\n{context_for_prompt}"
    )

    meta: Dict[str, Any] = {
        "mode": "answer_from_docs",
        "tool_called": None,
        "ticket": None,
        "retrieved_chunks": chunks,
    }

    # First call: decide answer vs tool
    first = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
    )

    msg = first.choices[0].message

    # Tool path
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            if tool_call.function.name == "escalate_ticket":
                args = json.loads(tool_call.function.arguments)
                tool_result = escalate_ticket(
                    user_question=args["user_question"],
                    retrieved_context=args["retrieved_context"],
                )

                meta["mode"] = "escalated"
                meta["tool_called"] = "escalate_ticket"
                meta["ticket"] = tool_result

                second = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        msg,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": "escalate_ticket",
                            "content": json.dumps(tool_result),
                        },
                    ],
                    temperature=0.1,
                )

                final = second.choices[0].message.content.strip()
                return final, context, meta

    # Direct answer path
    final_answer = (msg.content or "").strip()
    return final_answer, context, meta
