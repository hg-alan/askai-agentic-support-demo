import streamlit as st
from rag_agent import build_index, answer_question

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

# Buttons: set session state only. Streamlit will rerun automatically.

if col1.button("Refund policy"):
    st.session_state["user_question"] = examples["refund_docs"]

if col2.button("Express shipping"):
    st.session_state["user_question"] = examples["shipping_docs"]

if col3.button("6-month refund (escalate)"):
    st.session_state["user_question"] = examples["refund_escalate"]

if col4.button("Carry-on (escalate)"):
    st.session_state["user_question"] = examples["carry_on_escalate"]

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
            f"- **Decision**: escalate to human (insufficient / sensitive context)\n"
            f"- **Tool used**: `escalate_ticket`\n"
            f"- **Ticket ID**: `{ticket.get('ticket_id', 'n/a')}`\n"
            f"- **Assigned team**: {ticket.get('assigned_team', 'Tier-2 Support')}"
        )
    else:
        st.markdown(
            "- **Decision**: answer directly from retrieved documentation\n"
            "- **No escalation triggered**"
        )

    with st.expander("Show retrieved context (for transparency)"):
        st.code(context or "No relevant documentation found.", language="markdown")

# ---------- How this demo works ----------

with st.expander("How this demo works (tech + behavior)"):
    st.markdown(
        """
**Tech stack**

- Python + Streamlit
- OpenAI:
  - `text-embedding-3-small` for vector embeddings
  - `gpt-4.1-mini` for reasoning + tool calling
- ChromaDB as the in-memory vector store
- Markdown docs in `/docs` as the internal knowledge base

**Agent behavior**

1. The user asks a question.
2. The agent embeds the question and runs semantic search over the docs.
3. It sends the question + top-matched snippets to the model.
4. The model decides:
   - If the docs clearly answer: respond using that context.
   - If docs are missing/ambiguous/risky: call the `escalate_ticket` tool.
5. On escalation, a mock ticket is created instead of hallucinating a policy.
6. The UI exposes retrieved context + decision path for explainability.

This mirrors a production CX pattern:
grounded answers when safe, structured escalation when not.
        """
    )
