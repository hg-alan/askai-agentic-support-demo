import streamlit as st
from rag_agent import build_index, answer_question

# ---------- One-time index build ----------

if "index_built" not in st.session_state:
    col, chunks = build_index()
    st.session_state["index_built"] = True
    st.session_state["chunks"] = chunks

st.set_page_config(page_title="AskAI Agentic Support Demo", layout="wide")

st.title("AskAI Agentic Support Demo")
st.caption(
    "RAG + tool-calling agent: answers from internal docs, or autonomously escalates when unsure."
)

# ---------- Suggested test questions ----------

st.markdown("**Try one of these example questions:**")

col1, col2, col3, col4 = st.columns(4)
examples = {
    "Refund policy (answer from docs)": "What is your refund policy?",
    "Express shipping (answer from docs)": "How long does express shipping take?",
    "Edge case refund (should escalate)": "Can I get a refund after 6 months?",
    "Off-topic / risky (should escalate)": "What can I bring in my carry on?",
}

# Clicking a button sets the text input via session_state, then reruns
if col1.button("Refund policy"):
    st.session_state["user_question"] = examples["Refund policy (answer from docs)"]
    st.experimental_rerun()
if col2.button("Express shipping"):
    st.session_state["user_question"] = examples["Express shipping (answer from docs)"]
    st.experimental_rerun()
if col3.button("6-month refund (escalate)"):
    st.session_state["user_question"] = examples["Edge case refund (should escalate)"]
    st.experimental_rerun()
if col4.button("Carry-on (escalate)"):
    st.session_state["user_question"] = examples["Off-topic / risky (should escalate)"]
    st.experimental_rerun()

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

    if meta["mode"] == "escalated":
        ticket = meta.get("ticket", {})
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
- OpenAI for:
  - `text-embedding-3-small` (vector embeddings)
  - `gpt-4.1-mini` (reasoning + tool calling)
- ChromaDB as the in-memory vector store
- Simple Markdown docs in `/docs` as the knowledge base

**Agent behavior**

1. Take the user question.
2. Embed the question and run semantic search over the internal docs.
3. Pass the top matches + question into the model.
4. The model decides:
   - If docs clearly answer: respond using that context.
   - If docs are missing/ambiguous/risky: call the `escalate_ticket` tool.
5. On escalation, a mock ticket is created and shown, instead of hallucinating a policy.
6. We expose retrieved context and the decision path to keep the agent explainable.

This mirrors a production pattern Ask-AI / CX teams care about:
grounded answers when safe, structured escalation when not.
        """
    )
