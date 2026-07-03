"""
tools.py
--------
Step 3: define the tools the agent is allowed to call.

Tool 1 - search_marketing_documents: semantic search over the uploaded
  file's chunks (RAG). Good for qualitative / "what does this say about X".

Tool 2 - analyze_marketing_data: gives the agent a Python (pandas) REPL
  scoped to the uploaded dataframe. Good for numeric questions
  ("total spend by channel", "which campaign had best ROAS").

This mix is what makes it "agentic RAG" rather than plain RAG: the LLM
decides, per question, which tool(s) to invoke and can chain them.
"""

from langchain.tools.retriever import create_retriever_tool
from langchain_experimental.tools import PythonAstREPLTool


def build_tools(vectorstore=None, df=None):
    tools = []

    if vectorstore is not None:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        retriever_tool = create_retriever_tool(
            retriever,
            name="search_marketing_documents",
            description=(
                "Search the uploaded marketing/media file (PDF, report, or CSV notes) "
                "for relevant passages. Use this for qualitative questions about the "
                "content, context, wording, or narrative of the uploaded document — "
                "e.g. 'what does the report say about our TikTok strategy?'"
            ),
        )
        tools.append(retriever_tool)

    if df is not None:
        python_tool = PythonAstREPLTool(locals={"df": df})
        python_tool.name = "analyze_marketing_data"
        python_tool.description = (
            "Run pandas code to answer NUMERIC / statistical questions about the "
            "uploaded dataset. A pandas DataFrame called `df` is already loaded. "
            "Columns available: " + ", ".join(map(str, df.columns)) + ". "
            "Always write an expression whose result is the final answer "
            "(e.g. df.groupby('channel')['conversions'].sum().sort_values(ascending=False)). "
            "Use this for totals, averages, top-N, trends, correlations, comparisons, etc."
        )
        tools.append(python_tool)

    return tools
