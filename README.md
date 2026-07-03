# Marketing Agentic RAG Assistant (Local, Free, Ollama + LangChain + LangGraph + Langfuse)

Upload a marketing/media file (CSV campaign data, PDF report, Excel, or TXT) and ask
natural-language questions. An **agent** decides whether to run pandas analysis, do
semantic document search, or both — then answers grounded in the actual data.

Everything runs **locally and free**: local LLM (Ollama), local vector DB (Chroma),
local embeddings (Ollama). Langfuse (free tier) adds observability/tracing.

---

## 1. Architecture (what's happening, step by step)

```
Upload file
   │
   ▼
ingest.py ──► if text-like (pdf/txt/csv): chunk text → embed locally (Ollama) → store in Chroma (vector DB)
          ──► if tabular (csv/xlsx): load into a pandas DataFrame
   │
   ▼
tools.py ──► Tool 1: search_marketing_documents  (semantic search over Chroma — classic RAG)
          ──► Tool 2: analyze_marketing_data      (Python/pandas REPL scoped to the DataFrame)
   │
   ▼
agent.py ──► LangGraph ReAct agent wraps a local ChatOllama model + both tools
          ──► On each question, the LLM reasons: "do I need to search text, run math, both, or neither?"
          ──► Calls tool(s), reads results, loops until it can answer
          ──► Langfuse callback traces every step (prompt, tool call, latency, tokens)
   │
   ▼
app.py ──► Streamlit chat UI: upload, ask, see the answer
```

This is what makes it **agentic RAG** rather than plain RAG: a plain RAG pipeline
always does retrieve → generate. Here, the LLM itself decides per-question which
tool(s) to call (or none), and can chain multiple tool calls — e.g. first search the
report for "what's the target CAC", then run pandas to compute actual CAC, then
compare them in its final answer.

---

## 2. Prerequisites (one-time setup on your Lenovo laptop)

### a) Install Ollama
Download from https://ollama.com (Windows/Mac/Linux installer). Verify:
```bash
ollama --version
```

### b) Pull the two models you need (both free, run 100% locally)
```bash
ollama pull llama3.2          # the chat/reasoning model (agent brain)
ollama pull nomic-embed-text  # the embedding model (for RAG search)
```
> If your laptop has limited RAM (<8GB free), use `ollama pull llama3.2:1b` instead
> and pass `--model llama3.2:1b` / type it in the sidebar model field.

### c) Python environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### d) (Optional but recommended for the resume) Langfuse — free tracing
1. Go to https://cloud.langfuse.com → sign up free → create a project.
2. Copy your **Public Key** and **Secret Key**.
3. Set them as environment variables before launching:
```bash
# macOS/Linux
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."

# Windows PowerShell
setx LANGFUSE_PUBLIC_KEY "pk-lf-..."
setx LANGFUSE_SECRET_KEY "sk-lf-..."
```
If you skip this, the app still works — it just won't log traces.

---

## 3. Run it today

```bash
streamlit run app.py
```
Open the URL Streamlit prints (usually http://localhost:8501). Upload a CSV of
campaign data (columns like `channel, spend, impressions, clicks, conversions, date`)
or a PDF marketing report, then ask things like:
- "Which channel had the highest ROAS?"
- "Summarize the key risks mentioned in the report."
- "What was total spend vs conversions by month?"
- "Compare Instagram and Google Ads performance."

Don't have a file handy? Ask me and I can generate a sample marketing CSV for you
to test with immediately.

### About "deploying"
Because the LLM (Ollama) runs on your own laptop, true public cloud deployment
(Streamlit Cloud, etc.) won't work out of the box — the cloud server can't reach
your local Ollama. Your options today:
1. **Run it locally** (above) — this is a fully functional, demoable app; perfectly
   normal for a portfolio project and what most interviewers will ask you to show
   via screen share or a recorded demo.
2. **Expose it temporarily** with a tunnel like `ngrok http 8501` so you can share
   a live link for a demo call.
3. **Swap the LLM call** to a free-tier hosted model (e.g. Groq's free API) if you
   ever need a truly public deployment — the rest of the architecture (LangChain
   tools, LangGraph agent, Langfuse tracing, Streamlit UI) stays identical, you'd
   only change `ChatOllama` to another `langchain_*` chat model class.

---

## 4. Project structure
```
ollama_agentic_rag/
├── app.py          # Streamlit UI
├── agent.py        # LangGraph ReAct agent + Langfuse callback
├── tools.py        # RAG retriever tool + pandas analysis tool
├── ingest.py       # File loading, chunking, embedding, vector store
├── requirements.txt
└── README.md
```

---

## 5. Resume bullet points (use/adapt these)

- Built an **agentic RAG system** using **LangChain + LangGraph**, running a fully
  local **Ollama** LLM and embedding model, that autonomously chooses between
  semantic document search (Chroma vector DB) and pandas-based data analysis to
  answer natural-language marketing/analytics questions.
- Implemented **observability with Langfuse**, tracing every agent step, tool call,
  and token usage for debugging and evaluation.
- Designed a tool-calling architecture (ReAct pattern) enabling multi-step
  reasoning: the agent chains document retrieval and numeric computation to answer
  compound questions without hallucinating figures.
- Shipped an interactive **Streamlit** front end supporting file upload (PDF/CSV/
  Excel) and conversational Q&A, deployable with zero paid infrastructure.

---

## 6. Interview prep — likely questions & how to answer

**Q: What makes this "agentic" and not just RAG?**
A: In plain RAG, every query follows the same fixed path: embed query → retrieve
chunks → stuff into prompt → generate. Here, an LLM-driven controller (LangGraph's
ReAct loop) decides *per query* which tool(s) to call — semantic search, pandas
analysis, both, or neither — and can make multiple tool calls in sequence,
observing results between calls before producing a final answer.

**Q: Why two tools instead of one?**
A: RAG (vector search) answers "what does the document say," but is bad at exact
math — LLMs can hallucinate numbers from retrieved text. A pandas tool lets the
agent execute real code against the actual data for anything numeric (sums,
group-bys, trends), while retrieval handles qualitative/context questions. This
also demonstrates understanding of RAG's limitations, which interviewers like.

**Q: How do you keep the agent grounded / avoid hallucination?**
A: The system prompt forces tool use for any numeric claim, tool outputs are fed
back into the loop as observations, and Langfuse traces let you audit exactly
which tool produced which number. You could add citation of the exact chunk
source next.

**Q: Why Ollama instead of OpenAI/Anthropic API?**
A: Zero cost, no API key, data never leaves the laptop (relevant for sensitive
marketing/client data), and it's a good way to demonstrate you understand local
model deployment — a real skill companies with data-privacy constraints care about.

**Q: What are the limitations / what would you improve for production?**
A: Local models are weaker than frontier hosted models, so accuracy is lower.
The PythonAstREPLTool executes generated code, which is a security consideration
in a multi-tenant production system (I'd sandbox it, e.g. in a container, in real
production). I'd also add: streaming responses, conversation memory across
sessions, source-citation in retrieved answers, evaluation via Langfuse's
dataset/eval features, and swapping Chroma for a scalable vector DB (e.g. Qdrant)
at larger scale.

**Q: Walk me through what happens when I ask "which channel had highest ROAS?"**
A: The agent's LLM reasons this needs a computation → calls `analyze_marketing_data`
with a pandas expression computing `revenue/spend` grouped by channel → gets the
result back as an observation → LLM formats a final natural-language answer citing
the actual channel and number → Langfuse logs the whole trace (prompt, tool call,
tool output, final answer, latency).

---

## 7. Troubleshooting
- **"model not found"**: run `ollama pull <model_name>` for the model you typed in
  the sidebar.
- **Slow responses**: try a smaller model (`llama3.2:1b`) or reduce `k` in
  `tools.py`'s retriever.
- **Langfuse not tracing**: confirm env vars are set in the *same terminal* you
  launched `streamlit run app.py` from.
- **CSV column names not recognized**: check `st.sidebar.dataframe(df.head())`
  after upload to confirm the DataFrame loaded correctly.
