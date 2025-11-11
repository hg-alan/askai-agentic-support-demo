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

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=api_key)

chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
COLLECTION_NAME = "support_kb"


def _get_or_create_collection():
    existing = {c.name: c for c in chroma_client.list_collections()}
    if COLLECTION_NAME in existing:
        return existing[COLLECTION_NAME]
    return chroma_client.create_collection(name=COLLECTION_NAME)


# global collection handle used by retrieval
collection = _get_or_create_collection()

# ---------- Data loading & embeddings ----------


def load_docs_from_folder(folder_path: str):
    docs = []
    for path in glob.glob(os.path.join(folder_path, "**/*"), recursive=True):
        if os.path.isfile(path) and path.lower().endswith((".md", ".txt")):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    docs.append((os.path.basename(path), text))
    return docs


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


def build_index() -> Tuple[chromadb.api.models.Collection.Collection, int]:
    """Rebuilds the vector index from ./docs. Returns (collection, num_chunks)."""
    # Drop existing collection for determinism
    for c in chroma_client.list_collections():
        if c.name == COLLECTION_NAME:
            chroma_client.delete_collection(name=COLLECTION_NAME)

    col = chroma_client.create_collection(name=COLLECTION_NAME)

    docs = load_docs_from_folder("docs")
    if not docs:
        raise RuntimeError("No documents found in ./docs. Add some .md or .txt files.")

    ids, contents, embeddings = [], [], []
    idx = 0

    for filename, text in docs:
        for chunk in chunk_text(text):
            ids.append(f"{filename}-{idx}")
            contents.append(chunk)
            embeddings.append(get_embeddings(chunk)[0])
            idx += 1

    col.add(ids=ids, documents=contents, embeddings=embeddings)

    # update global reference used by retrieval
    global collection
    collection = col

    return col, len(ids)


def retrieve_context(query: str, k: int = 4) -> str:
    q_emb = get_embeddings(query)[0]
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )
    docs = result.get("documents", [[]])[0]
    return "\n\n".join(docs)


# ---------- Agentic behavior: tool (function) calling ----------

def escalate_ticket(user_question: str, retrieved_context: str) -> Dict[str, Any]:
    """
    Mock side-effect to demonstrate agentic behavior.
    In a real system this would hit Zendesk/Jira/Slack/etc.
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
    Agentic answer:
    - Retrieve context
    - Let LLM decide to:
        - answer from docs, OR
        - call escalate_ticket (tool) for human escalation.
    - Return (final_answer, retrieved_context, meta).
    """
    context = retrieve_context(question, k=4)

    system_prompt = (
        "You are a precise, trustworthy support agent for an imaginary SaaS company.\n"
        "You have two options:\n"
        "1) If the answer is clearly supported by the provided documentation, answer concisely.\n"
        "2) If documentation is missing, unclear, or the question is risky/compliance-sensitive, "
        "CALL the `escalate_ticket` tool instead of guessing.\n"
        "Never invent policies. Prefer escalation when unsure."
    )

    user_content = (
        f"User question:\n{question}\n\n"
        f"Retrieved documentation:\n{context or '[NO RELEVANT DOCS FOUND]'}"
    )

    meta: Dict[str, Any] = {
        "mode": "answer_from_docs",
        "tool_called": None,
        "ticket": None,
    }

    # First model call: decide whether to answer or call a tool
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

    # If the model chose to call a tool → execute it and do a second call
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

    # Otherwise, model answered directly from docs
    final_answer = (msg.content or "").strip()
    return final_answer, context, meta
