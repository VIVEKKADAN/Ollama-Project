# Marketing Agentic RAG Assistant

Local, free, agentic Retrieval-Augmented Generation system for marketing and media data analysis. Upload a campaign CSV, Excel sheet, PDF report, or text file and ask natural-language questions. An LLM-driven agent decides — per question — whether to run pandas analysis, semantic document search, or both, then answers grounded in the actual data.

Runs entirely on local infrastructure: local LLM (Ollama), local vector database (Chroma), local embeddings (Ollama). Langfuse (free tier) adds observability and tracing.

## Why "agentic" RAG, not just RAG

A standard RAG pipeline always follows the same path: embed query → retrieve chunks → generate. Here, a LangGraph ReAct agent decides per question which tool(s) to call — semantic search, pandas analysis, both, or neither — and can chain multiple tool calls, using the result of one to inform the next.

Example: "Is actual CAC under our target?" → the agent first searches the uploaded report for the target CAC, then runs a pandas computation for the actual CAC, then compares the two in its final answer.

## Architecture

```
Upload file
   │
   ▼
ingest.py ──► text-like files (pdf/txt/csv): chunk → embed locally (Ollama) → store in Chroma
          ──► tabular files (csv/xlsx): load into a pandas DataFrame
   │
   ▼
tools.py ──► search_marketing_documents  (semantic search over Chroma)
          ──► analyze_marketing_data      (pandas REPL scoped to the DataFrame)
   │
   ▼
agent.py ──► LangGraph ReAct agent wraps a local ChatOllama model + both tools
          ──► reasons per-question about which tool(s) to call, loops until it can answer
          ──► Langfuse callback traces every step (prompt, tool call, latency, tokens)
   │
   ▼
app.py ──► Streamlit chat UI: upload, ask, see the answer
```

## Tech stack

Python · LangChain · LangGraph · Ollama (LLM + embeddings) · ChromaDB (vector store) · Langfuse (observability) · Streamlit (UI) · Pandas

## Setup

### 1. Install Ollama
Download from https://ollama.com, then verify:
```bash
ollama --version
```

### 2. Pull the required models
```bash
ollama pull llama3.2          # chat/reasoning model
ollama pull nomic-embed-text  # embedding model for RAG search
```
On lower-RAM machines, `ollama pull llama3.2:1b` is a lighter alternative — pass it into the model field in the sidebar.

### 3. Python environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Langfuse tracing (optional)
Create a free project at https://cloud.langfuse.com, then set:
```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
```
The app runs without these; tracing is simply skipped if unset.

## Usage

```bash
streamlit run app.py
```
Open the printed URL (usually http://localhost:8501). Upload a campaign CSV (a sample is included at `sample_marketing_data.csv`) or a PDF marketing report, then ask questions such as:
- "Which channel had the highest ROAS?"
- "Summarize the key risks mentioned in the report."
- "What was total spend vs conversions by month?"
- "Compare Instagram and Google Ads performance."

## Deployment notes

Since the LLM runs locally via Ollama, public cloud hosts (e.g. Streamlit Community Cloud) can't reach it out of the box. Options:
- Run locally (above) — fully functional for demos.
- Expose temporarily with a tunnel (e.g. `ngrok http 8501`) for a live shareable link.
- Swap `ChatOllama` for a hosted `langchain_*` chat model (e.g. a free-tier API) to enable true public deployment — the rest of the architecture is unchanged.

## Project structure
```
ollama_agentic_rag/
├── app.py                      # Streamlit UI
├── agent.py                    # LangGraph ReAct agent + Langfuse callback
├── tools.py                    # RAG retriever tool + pandas analysis tool
├── ingest.py                   # File loading, chunking, embedding, vector store
├── requirements.txt
├── sample_marketing_data.csv   # Sample dataset for quick testing
└── README.md
```

## Limitations & possible extensions
- Local models are weaker than frontier hosted models, so accuracy is lower than a cloud-LLM equivalent.
- `PythonAstREPLTool` executes generated code — in a multi-tenant production setting this should run sandboxed (e.g. containerized).
- Possible next steps: streaming responses, persistent conversation memory, source citations on retrieved passages, evaluation via Langfuse's dataset/eval features, and a more scalable vector store (e.g. Qdrant) at larger data volumes.

## Troubleshooting
- **"model not found"**: run `ollama pull <model_name>` for the model set in the sidebar.
- **Slow responses**: try a smaller model (`llama3.2:1b`) or reduce `k` in `tools.py`'s retriever.
- **Langfuse not tracing**: confirm the environment variables are set in the same terminal used to launch `streamlit run app.py`.
- **CSV columns not recognized**: check the sidebar's dataframe preview after upload to confirm the file loaded correctly.
