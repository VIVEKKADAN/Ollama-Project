"""
app.py
------
Step 5: the UI. Run with:  streamlit run app.py
Upload a marketing/media file (CSV, XLSX, PDF, or TXT), then ask questions
in the chat box. The agent decides whether to search the document, run
pandas analysis, or both.
"""

import os
import tempfile

import streamlit as st

from ingest import build_vectorstore, load_dataframe
from tools import build_tools
from agent import build_agent, get_langfuse_handler

st.set_page_config(page_title="Marketing Agentic RAG", layout="wide")
st.title("📊 Marketing Agentic RAG Assistant")
st.caption("100% local & free — Ollama + LangChain + LangGraph + Chroma + Langfuse")

with st.sidebar:
    st.header("1. Upload your data")
    uploaded = st.file_uploader(
        "Marketing / media file", type=["pdf", "csv", "xlsx", "xls", "txt"]
    )
    model_name = st.text_input("Ollama chat model", value="llama3.2")

    st.header("2. Langfuse tracing (optional)")
    st.caption(
        "Free at cloud.langfuse.com. Set LANGFUSE_PUBLIC_KEY and "
        "LANGFUSE_SECRET_KEY as environment variables before launching "
        "Streamlit to enable tracing."
    )
    lf_status = "✅ enabled" if os.environ.get("LANGFUSE_PUBLIC_KEY") else "⚠️ not configured"
    st.caption(f"Status: {lf_status}")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "loaded_file" not in st.session_state:
    st.session_state.loaded_file = None

# ---- Ingest on upload ----
if uploaded is not None and st.session_state.loaded_file != uploaded.name:
    with st.spinner("Ingesting file (chunking + local embeddings)..."):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        vectorstore = None
        df = None

        if suffix.lower() in (".pdf", ".txt", ".csv"):
            vectorstore = build_vectorstore(tmp_path)
        if suffix.lower() in (".csv", ".xlsx", ".xls"):
            df = load_dataframe(tmp_path)

        tools = build_tools(vectorstore, df)
        if not tools:
            st.error("Could not build any tools from this file type.")
        else:
            st.session_state.agent = build_agent(tools, model_name=model_name)
            st.session_state.loaded_file = uploaded.name
            st.session_state.messages = []

            if df is not None:
                st.sidebar.success(f"Dataframe loaded: {df.shape[0]} rows x {df.shape[1]} cols")
                st.sidebar.dataframe(df.head())
            if vectorstore is not None:
                st.sidebar.success("Document indexed for semantic search")

    st.success(f"'{uploaded.name}' is ready. Ask your questions below!")

# ---- Chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat input ----
query = st.chat_input("Ask about your marketing/media data...")
if query:
    if st.session_state.agent is None:
        st.warning("Please upload a file first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        callbacks = []
        if os.environ.get("LANGFUSE_PUBLIC_KEY"):
            try:
                callbacks = [get_langfuse_handler()]
            except Exception as e:
                st.sidebar.warning(f"Langfuse tracing disabled: {e}")

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    config = {"callbacks": callbacks} if callbacks else {}
                    result = st.session_state.agent.invoke(
                        {"messages": [{"role": "user", "content": query}]},
                        config=config,
                    )
                    answer = result["messages"][-1].content
                except Exception as e:
                    answer = f"Error: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
