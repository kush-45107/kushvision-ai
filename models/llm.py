import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from serpapi import GoogleSearch
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

SERP_KEY = os.getenv("SERP_API_KEY")

def search_web(query: str) -> str:
    """Fetch top Google snippets via SerpAPI."""
    params = {
        "q": query,
        "api_key": SERP_KEY,
        "engine": "google",
        "num": 5,
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    snippets = []
    if "organic_results" in results:
        for r in results["organic_results"][:5]:
            if "snippet" in r:
                snippets.append(r["snippet"])

    return "\n".join(snippets) if snippets else "No results found."

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

@tool
def get_creator_info(query: str) -> str:
    """
    Use this tool when the user asks who made you, who created you,
    who built you, who is your developer, who is your creator,
    or who owns KushVision.
    """
    return "KushVision AI was created and developed by Kushagra Srivastawa."

@tool
def get_about_info(query: str) -> str:
    """
    Use this tool when the user asks what is KushVision AI, what is this website,
    what can you do, who are you, tell me about this chatbot, or anything
    about the platform itself.
    """
    return """KushVision AI is an advanced AI chatbot developed by Kushagra Srivastawa.

It is a smart multimodal AI assistant where you can:
- Chat with AI and ask anything
- Upload Files and chat with your documents
- Generate AI images from text
- Use voice input and get voice responses

KushVision AI is designed to provide a complete AI experience with multiple powerful features in one platform."""


@tool
def realtime_search(query: str) -> str:
    """
    Use this tool for any question that needs current, live, or real-time information.
    This includes: date, today's news, weather, stock prices, sports scores, election results,
    current events, bitcoin price, gold price, or anything happening right now or recently.
    """
    try:
        result = search_web(query)
        return result if result else "Sorry, I couldn't fetch real-time data right now."
    except Exception as e:
        return f"Real-time search failed: {str(e)}"


tools = [get_creator_info, get_about_info, realtime_search]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    # add_messages handles appending — this IS the memory
    messages: Annotated[Sequence[BaseMessage], add_messages]

SYSTEM_PROMPT = SYSTEM_PROMPT = """You are KushVision AI, an advanced AI assistant created by Kushagra Srivastawa.

You have access to the following tools:
- get_creator_info: for questions about who made/created/developed you
- get_about_info: for questions about what KushVision AI is or what you can do
- realtime_search: for current/live/real-time information like news, weather, prices, scores, date, time

STRICT Rules:
- Always use get_creator_info when user asks about your creator or developer.
- Always use get_about_info when user asks about the platform or your capabilities.
- ALWAYS use realtime_search for ANY of these — do NOT answer from memory:
  * today's date, current date, what day is it, what time is it
  * news, weather, stock prices, sports scores, election results
  * bitcoin price, gold price, any live/current data
  * anything with words like "today", "now", "current", "latest"
- You do NOT know the current date or time. You MUST use realtime_search for it.
- For normal questions, answer directly from your knowledge without using tools.
- Be helpful, friendly, and concise.
"""

def agent_node(state: AgentState) -> AgentState:
    """Main agent node — calls LLM with tools."""
    messages = state["messages"]

    # Prepend system message if not already there
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Router: if the last message has tool calls, go to tools; else end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

tool_node = ToolNode(tools)

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")  # after tools, go back to agent

agent = graph_builder.compile()

# In-memory conversation store (per session)
# For multi-user: use a dict keyed by session/user id
conversation_history: list[BaseMessage] = []


def chat(user_input: str) -> str:
    """
    Send a message and get a response.
    Maintains full conversation history (memory) across turns.
    """
    global conversation_history

    # Add user message to history
    conversation_history.append(HumanMessage(content=user_input))

    # Run the graph with full history
    result = agent.invoke({"messages": conversation_history})

    # Update history with the full result (includes tool messages, AI messages)
    conversation_history = list(result["messages"])

    # Return the last AI response
    for msg in reversed(conversation_history):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "Sorry, I couldn't generate a response."


def reset_memory():
    """Clear conversation history to start a fresh session."""
    global conversation_history
    conversation_history = []
    print("Memory cleared.")