"""
agent.py
--------
Step 4: wire the LLM + tools into an agent, and wire in Langfuse for
observability (every tool call, prompt, and token gets traced for free
at cloud.langfuse.com).

We use LangGraph's prebuilt `create_react_agent`. "ReAct" = the model
loops: Reason about the question -> decide to call a tool (or not) ->
observe the tool result -> reason again -> ... -> final answer.
That loop is what makes this "agentic" rather than a single
retrieve-then-generate RAG call.
"""

import os
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

SYSTEM_PROMPT = """You are a marketing & media analytics assistant.

You have access to tools:
- search_marketing_documents: for qualitative questions about the content of an uploaded document/report.
- analyze_marketing_data: for numeric/statistical questions about an uploaded dataset (df).

Rules:
1. If the question needs a number (total, average, top N, trend, comparison, %), you MUST use analyze_marketing_data — never guess numbers yourself.
2. If the question is about wording, context, strategy, or explanations in an uploaded document, use search_marketing_documents.
3. You may use both tools if a question needs both.
4. Always ground your final answer in the tool output. If a tool errors or returns nothing useful, say what went wrong instead of making something up.
5. Be concise and specific: cite the actual numbers/column names you computed.
"""


def get_langfuse_handler():
    """Create a Langfuse callback handler from environment variables.
    Sign up free at https://cloud.langfuse.com, create a project, and set:
      LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST (optional)
    If the keys aren't set, this will raise and app.py will simply skip tracing.
    """
    from langfuse.callback import CallbackHandler

    return CallbackHandler(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


def build_agent(tools, model_name: str = "llama3.2"):
    """Build the ReAct agent. model_name must already be pulled in Ollama
    (e.g. `ollama pull llama3.2`)."""
    llm = ChatOllama(model=model_name, temperature=0)
    agent = create_react_agent(llm, tools, state_modifier=SYSTEM_PROMPT)
    return agent
