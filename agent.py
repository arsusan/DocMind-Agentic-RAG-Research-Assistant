import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from sentence_transformers import SentenceTransformer
from supabase import create_client
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Step 1: define the retrieval tool the agent can choose to call ----
@tool
def search_documents(query: str) -> str:
    """Search the uploaded documents for information relevant to the query.
    Use this whenever the user asks something that might be answered by the
    uploaded documents (facts, details, specifics from the files)."""
    query_embedding = embedder.encode(query).tolist()
    result = supabase.rpc(
        "match_documents",
        {"query_embedding": query_embedding, "match_count": 4}
    ).execute()

    if not result.data:
        return "No relevant documents found."

    formatted = []
    for row in result.data:
        source = row["metadata"].get("source", "unknown")
        formatted.append(f"[Source: {source}] {row['content']}")
    return "\n\n".join(formatted)

tools = [search_documents]

# ---- Step 2: the LLM, aware it can call the tool above ----
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are DocMind, a research assistant. You have access to a tool called "
    "search_documents that searches the user's uploaded files. Use it when the "
    "question could be answered from those documents. If the question is general "
    "small talk or doesn't need document lookup, answer directly without the tool. "
    "When you do use retrieved content, cite the source file name in your answer."
))

# ---- Step 3: the graph itself ----
def call_model(state: MessagesState):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    tools_condition,   # prebuilt: routes to "tools" if the LLM asked for a tool call, else END
)
graph.add_edge("tools", "agent")  # after retrieving, go back to the LLM to write the final answer

app_graph = graph.compile()

def ask(question: str) -> str:
    result = app_graph.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content