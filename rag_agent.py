import os
import glob
import json
import uuid
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ---------- Setup ----------

def get_api_key() -> str:
    """
    Resolve OPENAI_API_KEY from:
    1) Environment / .env (local dev)
    2) Streamlit secrets (Streamlit Cloud)
    """
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")

    if not key:
        try:
            import streamlit as st  # only present in Streamlit env
            key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            key = None

    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. "
            "Set it in a local .env file, or in Streamlit Cloud secrets as OPENAI_API_KEY."
        )
    return key


api_key = get_api_key()
client = OpenAI(api_key=api_key)

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
COLLECTION_NAME = "support_kb"

# keep a simple in-memory copy of KB for display
_KB_DOCS_CACHE: List[Tuple[str, str]] = []


def _get_or_create_collection():
    existing = {c.name: c for c in chroma_client.list_collections()}
    if COLLECTION_NAME in existing:
        return existing[COLLECTION_NAME]
    return chroma_client.create_collection(name=COLLECTION_NAME)


collection = _get_or_create_collection()

# ---------- Data loading ----------


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
    Return the entire KB in a readable markdown form for the UI.
    """
    global _KB_DOCS_CACHE
    if not _KB_DOCS_CACHE:
        _KB_DOCS_CACHE = load_docs_from_folder("docs")

    if not _KB_DOCS_CACHE:
        return "No documentation loaded."

    parts = []
    for filename, text in _KB_DOCS_CACHE:
        parts.append(f"### {filename}\n\n{text}")
    return "\n\n---\n\n".join(parts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
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


# ---------- Indexing ----------


def build_index() -> Tuple[chromadb.api.models.Collection.Collection, int]:
    """
    Rebuilds the vector index from ./docs. Returns (collection, num_chunks).
    """
    global _KB_DOCS_CACHE

    # reset KB cache
    _KB_DOCS_CACHE = load_docs_from_folder("docs")

    if not _KB_DOCS_CACHE:
        raise RuntimeError("No documents found in ./docs. Add some .md or .txt files.")

    # drop existing collection to keep behavior deterministic
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


# ---------- Retrieval ----------


def retrieve_relevant_chunks(
    query: str,
    k: int = 4,
    similarity_threshold: float = 0.7,
) -> List[str]:
    """
    Return chunks that are semantically relevant.

    Uses cosine distance from Chroma:
      similarity = 1 - distance

    - If similarity >= similarity_threshold, we treat it as relevant.
    - If distances are missing for any reason, we conservatively keep the chunks.
    """
    q_emb = get_embeddings(query)[0]

    result = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "distances"],
    )

    docs = result.get("documents", [[]])[0] or []
    dists = result.get("distances", [[]])[0] or []

    # If Chroma didn't return distances, just return docs (best-effort)
    if not dists or len(dists) != len(docs):
        return docs

    relevant = []
    for doc, dist in zip(docs, dists):
        # Defensive: some backends may return None
        if dist is None:
            relevant.append(doc)
            continue

        sim = 1.0 - float(dist)
        if sim >= similarity_threshold:
            relevant.append(doc)

    return relevant



# ---------- Agentic behavior: tool-calling ----------


def escalate_ticket(user_question: str, retrieved_context: str) -> Dict[str, Any]:
    """
    Mock escalation side-effect.
    """
    ticket_id = str(uuid.uuid4())[:8]
    print(f"\n[ESCALATION] Created ticket {ticket_id} for question: {user_question}\n")
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
                "Use this when the available documentation is insufficient, "
                "out of scope, or when a human should review."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_question": {
                        "type": "string",
                        "description": "The original user question.",
                    },
                    "retrieved_context": {
                        "type": "string",
                        "description": "The documentation snippets the agent saw.",
                    },
                },
                "required": ["user_question", "retrieved_context"],
            },
        },
    }
]


def answer_question(question: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    - Retrieve relevant chunks (may be empty).
    - Let the LLM decide:
        - answer from docs, OR
        - call escalate_ticket tool.
    - Return (final_answer, concatenated_context, meta).
    """
    relevant_chunks = retrieve_relevant_chunks(question, k=4)
    context = "\n\n".join(relevant_chunks)

    system_prompt = (
        "You are a precise, trustworthy support agent for an imaginary SaaS company.\n"
        "You have two options:\n"
        "1) If the answer is clearly supported by the provided documentation, answer concisely.\n"
        "2) If documentation is missing, unclear, or the question is risky/compliance-sensitive, "
        "CALL the `escalate_ticket` tool instead of guessing.\n"
        "Never invent policies. Prefer escalation when unsure."
    )

    context_for_prompt = context if context else "[NO RELEVANT DOCS FOUND]"

    user_content = (
        f"User question:\n{question}\n\n"
        f"Retrieved documentation:\n{context_for_prompt}"
    )

    meta: Dict[str, Any] = {
        "mode": "answer_from_docs",
        "tool_called": None,
        "ticket": None,
        "relevant_chunks": relevant_chunks,
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
