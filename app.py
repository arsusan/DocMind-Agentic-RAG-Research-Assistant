import streamlit as st
from agent import ask

st.set_page_config(page_title="DocMind — Agentic RAG Assistant", page_icon="📄")
st.title("📄 DocMind — Agentic RAG Research Assistant")
st.caption("Ask a question — the agent decides on its own whether it needs to search your documents.")

if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Ask about your uploaded documents...")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(question)
            st.markdown(answer)
    st.session_state.history.append(("assistant", answer))