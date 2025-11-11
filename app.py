import streamlit as st
from rag_agent import build_index, answer_question, get_corpus_markdown

# ---------- Page config ----------

st.set_page_config(
    page_title="AskAI Agentic Support Demo",
    layout="wide",
)

# ---------- One-time index build ----------

if "index_built" not in st.session_state:
    _, chunks = build_index()
    st.session_state["index_built"] = True
    st.session_state["chunks"] = chunks

# ---------- Header ----------

st.title("AskAI Agentic Support Demo")
st.caption(
    "RAG + tool-calling agent: answers from internal docs, or autonomously escalates when unsure."
)

# ---------- Suggested test questions ----------

st.markdown("**Try one of these example questions:**")

col1, col2, col3, col4 = st.columns(4)

examples = {
    "refund_docs": "What is your refund policy?",
    "shipping_docs": "How long does express shipping take?",
    "refund_escalate": "Can I get a refund after 6 months?",
    "carry_on_escalate": "What can I bring in my carry on?",
}

if col1.button("Refund policy"):
    st.session_state["user_question"] = examples["refund_docs"]
if col2.button("Express shipping"):
    st.session_state["user_question"] = examples["shipping_docs"]
if col3.button("6-month refund (escalate)"):
    st.session_state["user_question"] = examples["refund_escalate"]
if col4.button("Carry-on (escalate)"):
    st.session_state["user_question"] = examples["carry_on_escalate"]

# ---------- Show full KB (ground truth) ----------

with st.expander("View full knowledge base (ground truth)"):
    st.markdown(get_corpus_markdown())

# ---------- Question input ----------

question = st.text_input(
    "Ask a support question:",
    key="user_question",
    placeholder="e.g. What is your refund policy?",
)

# ---------- Run agent ----------

if question:
    with st.spinner("Thinking (agent may escalate)..."):
        answer, context, meta = answer_question(question)

    st.subheader("Answer")
    st.markdown(answer)

    st.subheader("Agent reasoning")

    if meta.get("mode") == "escalated":
        ticket = meta.get("ticket", {}) or {}
        st.markdown(
            f"- **Decision**: escalate to human (insufficient / unclear docs)\n"
            f"- **Tool used**: `escalate_ticket`\n"
            f"- **Ticket ID**: `{ticket.get('ticket_id', 'n/a')}`\n"
            f"- **Assigned team**: {ticket.get('assigned_team', 'Tier-2 Support')}"
        )
    else:
        st.markdown(
            "- **Decision**: answered directly from retrieved documentation\n"
            "- **No escalation triggered**"
        )

    with st.expander("Show retrieved context (top chunks the agent saw)"):
        chunks = meta.get("retrieved_chunks") or []
        if chunks:
            for i, chunk in enumerate(chunks, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.code(chunk, language="markdown")
        else:
            st.markdown("_No documentation was retrieved for this query._")

# ---------- How this demo works ----------

with st.expander("How this demo works (tech + behavior)"):
    st.markdown(
        """
**Tech stack**

- Python + Streamlit
- OpenAI:
  - `text-embedding-3-small` for embeddings
  - `gpt-4.1-mini` for reasoning + tool calling
- ChromaDB as in-memory vector store
- Markdown docs in `/docs` as the internal KB

**Agent behavior**

1. User asks a question.
2. We embed the question and retrieve the top relevant chunks from the KB.
3. We pass the question + those chunks into the model.
4. The model either:
   - Answers strictly from that documentation, or
   - Calls the `escalate_ticket` tool when docs don't clearly answer / are risky.
5. On escalation, a mock ticket is created instead of hallucinating policy.
6. The UI exposes:
   - the full ground-truth KB,
   - the exact chunks the agent consulted,
   - and whether it chose to answer or escalate.

This is the pattern you'd use in production CX:
reliable when grounded, honest when not.
        """
    )
