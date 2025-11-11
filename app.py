import streamlit as st
from rag_agent import build_index, answer_question, get_corpus_markdown

# ---------- Page config ----------

st.set_page_config(
    page_title="AskAI Agentic Support Demo",
    layout="wide",
)

# ---------- Global styles ----------

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    max-width: 1000px;
}
h1 {
    font-size: 2.4rem;
    margin-bottom: 0.25rem;
}
h2 {
    font-size: 1.3rem;
    margin-top: 1.75rem;
    margin-bottom: 0.4rem;
}
h3 {
    font-size: 1.05rem;
    margin-bottom: 0.35rem;
}
.small-muted {
    font-size: 0.9rem;
    color: #9ca3af;
}
.stButton > button {
    border-radius: 999px;
    padding: 0.35rem 1.1rem;
    border: 1px solid #374151;
    background-color: transparent;
    font-size: 0.9rem;
}
.stButton > button:hover {
    border-color: #4b5563;
    background-color: #111827;
}
.answer-box {
    padding: 0.9rem 1.0rem;
    border-radius: 10px;
    background-color: #111827;
    border: 1px solid #374151;
    font-size: 1rem;
    line-height: 1.6;
}
.reason-box {
    padding: 0.7rem 1.0rem;
    border-radius: 10px;
    background-color: #020817;
    border: 1px solid #27272a;
    font-size: 0.95rem;
}
.reason-box ul {
    margin: 0.25rem 0 0.25rem 1.1rem;
}
div[data-testid="stExpander"] {
    border-radius: 10px !important;
    border: 1px solid #27272a !important;
    background-color: #020817 !important;
}
code, pre {
    font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- One-time index build ----------

if "index_built" not in st.session_state:
    _, chunks = build_index()
    st.session_state["index_built"] = True
    st.session_state["chunks"] = chunks

# Ensure the controlled input key exists
if "user_question" not in st.session_state:
    st.session_state["user_question"] = ""

# ---------- Header ----------

st.title("AskAI Agentic Support Demo")
st.markdown(
    '<div class="small-muted">'
    "RAG + tool-calling support agent: answers from internal docs, or escalates transparently when unsure."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------- Ask a question ----------

st.markdown("#### Ask a question")

st.text_input(
    " ",
    key="user_question",
    placeholder='Type a support question, e.g. "What is your refund policy?"',
    label_visibility="collapsed",
)

st.markdown('<div class="small-muted">Or try one of these:</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

examples = {
    "refund_docs": "What is your refund policy?",
    "shipping_docs": "How long does express shipping take?",
    "refund_strict": "Can I get a refund after 6 months?",
    "carry_on_escalate": "What can I bring in my carry on?",
}

example_query = None

with c1:
    if st.button("Refund policy"):
        example_query = examples["refund_docs"]
with c2:
    if st.button("Express shipping"):
        example_query = examples["shipping_docs"]
with c3:
    if st.button("Refund after 6 months"):
        example_query = examples["refund_strict"]
with c4:
    if st.button("Carry-on (escalate)"):
        example_query = examples["carry_on_escalate"]

# Decide which question to answer this run:
# - If an example button was clicked, use that.
# - Otherwise, use whatever is in the text input.
current_q = (example_query or st.session_state.get("user_question", "")).strip()

# ---------- Run agent + show answer ----------

if current_q:
    with st.spinner("Thinking (agent may escalate)..."):
        answer, context, meta = answer_question(current_q)

    # Answer
    st.markdown("## 💬 Answer")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    # Agent reasoning
    st.markdown("## 🧠 Agent reasoning")
    mode = meta.get("mode")
    ticket = meta.get("ticket") or {}
    retrieved_chunks = meta.get("retrieved_chunks") or []

    if mode == "escalated":
        reasoning_lines = [
            "- **Decision**: escalate to human (insufficient / unclear docs)",
            "- **Tool used**: `escalate_ticket`",
            f"- **Ticket ID**: `{ticket.get('ticket_id', 'n/a')}`",
            f"- **Assigned team**: {ticket.get('assigned_team', 'Tier-2 Support')}",
        ]
    else:
        reasoning_lines = [
            "- **Decision**: answered directly from retrieved documentation",
            "- **No escalation triggered**",
        ]

    st.markdown(
        '<div class="reason-box">' + "<br>".join(reasoning_lines) + "</div>",
        unsafe_allow_html=True,
    )

    # Retrieved context
    with st.expander("📄 Retrieved context (top chunks the agent consulted)", expanded=False):
        if retrieved_chunks:
            for i, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"**Chunk {i}**")
                st.code(chunk, language="markdown")
        else:
            st.markdown("_No documentation was retrieved for this query._")

# ---------- Knowledge base + explainer at bottom ----------

st.markdown("---")

with st.expander("📚 View full knowledge base (ground truth)", expanded=False):
    st.markdown(get_corpus_markdown())

with st.expander("ℹ️ How this demo works (tech + behavior)", expanded=False):
    st.markdown(
        """
**What this shows**

- Answers are grounded in your internal docs.
- If the docs don’t cover something (e.g. carry-on rules), the agent:
  - refuses to hallucinate,
  - calls a tool (`escalate_ticket`) to hand off to a human,
  - and surfaces that decision transparently.

**Architecture**

1. Markdown docs in `/docs` form the internal knowledge base.
2. On startup, they’re embedded with `text-embedding-3-small` into ChromaDB.
3. For each question:
   - We retrieve the top matching chunk(s).
   - We call `gpt-4.1-mini` with:
       - system prompt (support agent + rules),
       - the user question,
       - retrieved chunks,
       - a tool definition for `escalate_ticket`.
4. The model either:
   - answers from the provided context, or
   - invokes `escalate_ticket`, which creates a mock escalation payload.
5. This UI:
   - highlights the final answer,
   - explains the decision,
   - shows the exact chunks consulted,
   - and exposes the full KB for transparency.
        """
    )
