import streamlit as st
from rag_agent import build_index, answer_question

# Build index once on app start
if "index_built" not in st.session_state:
    col, chunks = build_index()
    st.session_state["index_built"] = True
    st.session_state["chunks"] = chunks

st.title("AskAI Agentic Support Demo")
st.caption(
    "RAG + tool-calling agent: answers from internal docs, or autonomously escalates when unsure."
)

question = st.text_input("Ask a support question:")

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
            f"- **Tool used**: `{meta.get('tool_called')}`\n"
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
