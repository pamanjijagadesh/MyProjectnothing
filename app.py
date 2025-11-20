import asyncio
import time
import os
from typing import Annotated, Sequence, TypedDict
from operator import add as add_messages
from dotenv import load_dotenv
import streamlit as st
# LangChain/LangGraph Imports
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.graph import StateGraph, END

load_dotenv(".env")

# ==================== Configuration and Setup ====================

MISTRAL_ENDPOINT = "https://mistral-small-2503-Pamanji-test.southcentralus.models.ai.azure.com"
MISTRAL_API_KEY = "5SKKbylMh5ueyeSfvUre68vknfYZMVAr"

# Initialize DuckDuckGo Search Wrapper
wrapper = DuckDuckGoSearchAPIWrapper(max_results=50)

# ========== Custom Data Classes and State ==========
class AgentState(TypedDict):
    """The state for the LangGraph agent."""
    # The 'add_messages' operator adds new messages to the end of the sequence
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ========== LangGraph Conditional Logic ==========
def should_continue(state: AgentState) -> str:
    """
    Decides whether the agent should continue to the tool executor or end.
    Returns the name of the next node ('tool_executor' or '__end__').
    """
    last_message = state['messages'][-1]
    # Check if the LLM requested a tool call
    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        return "tool_executor"
    return END

# ========== LLM Prompts ==========
SYSTEM_INSTRUCTION = (
    """
You are a strictly web-dependent assistant.

For every user query, you must first call the `search_web` tool to retrieve information.
You are forbidden from using your own internal knowledge, reasoning, assumptions, or prior training data to answer any question.
If the information from the web search is sufficient, provide a concise and helpful answer based ONLY on the search results.
If the search results are empty or irrelevant, politely inform the user that you could not find the information.
"""
)

# ========== LLM & Chain Setup ==========
try:
    # LLM instances are typically designed to be thread-safe/asynchronous compatible
    llm = ChatMistralAI(
        endpoint=MISTRAL_ENDPOINT,
        mistral_api_key=MISTRAL_API_KEY,
        temperature=0.3,
    )
except Exception as e:
    print(f"Warning: Failed to initialize ChatMistralAI. Error: {e}")
    llm = None 

# ========== Agent Tools ==========
@tool
def search_web(query: str) -> str:
    """Run DuckDuckGo search and return the search results as a concatenated string."""
    try:
        # Note: wrapper.run is synchronous, but LangGraph handles running this 
        # in a thread pool within the async `take_action` node using tool.ainvoke().
        results = wrapper.run(query)
        print(f"Web Search results for: {query}")
        print("======================================")
        print(results[:200] + "...")
        print("======================================\n")
        return results if results else "No search results available."
    except Exception as e:
        print(f"[Search Error] Failed to execute search: {e}")
        return "Search failed due to an execution error. No context available."

# ========== LangGraph Agent Nodes ==========
tools = [search_web]
tools_dict = {tool_.name: tool_ for tool_ in tools}

# Bind tools to the LLM
# The LLM is bound with tools once and used across the graph nodes.
parent = llm.bind_tools(tools) if llm else None

def call_llm(state: AgentState) -> AgentState:
    """Node to call the main LLM with the full conversation history and system prompt."""
    if not parent:
        raise ValueError("LLM is not initialized. Check your API keys.")
        
    # Prepend the system instruction
    messages = [SystemMessage(content=SYSTEM_INSTRUCTION)] + list(state["messages"])
    
    print("🧠 Invoking LLM...")
    # LangGraph handles the synchronous `parent.invoke` within an async thread if needed.
    message = parent.invoke(messages)
    
    return {"messages": [message]}

async def take_action(state: AgentState) -> AgentState:
    """Node to execute the tool calls concurrently (fully asynchronous)."""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    results = []
    tool_tasks = []

    # Prepare concurrent tool execution using the asynchronous tool invoker (.ainvoke)
    for t in tool_calls:
        if t['name'] not in tools_dict:
            error_msg = f"Invalid Tool Name: {t['name']}"
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=error_msg))
            continue
            
        tool_func = tools_dict[t['name']]
        tool_args = t['args']

        print(f"🔧 Scheduling Tool: {t['name']} with args: {tool_args}")
        # Append the asynchronous call to the task list
        tool_tasks.append(tool_func.ainvoke(tool_args))

    # Run all tool calls concurrently
    tool_results = await asyncio.gather(*tool_tasks)

    # Process results and create ToolMessages
    for t, result in zip(tool_calls, tool_results):
        print(f"✅ Tool executed: {t['name']} -> {str(result)[:50]}...")
        # Send the raw string result back to the LLM
        results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
            
    return {"messages": results}

# -----------------------
# Graph Definition
# -----------------------
agent_graph = StateGraph(AgentState)
agent_graph.add_node("llm", call_llm)
agent_graph.add_node("tool_executor", take_action)

# If the LLM requests a tool call, execute the tool. Otherwise, end the flow.
agent_graph.add_conditional_edges("llm", should_continue, {"tool_executor": "tool_executor", END: END})

# After the tool is executed, feed the results back to the LLM for final answer generation.
agent_graph.add_edge("tool_executor", "llm")

# Set the starting point
agent_graph.set_entry_point("llm")

# Compile the graph for execution
assistant_agent = agent_graph.compile()

# ========== MAIN EXECUTION FUNCTION ==========
async def run_agent(query: str):
    """Executes the LangGraph agent for a single query asynchronously."""
    if not llm:
        print("\n--- CRITICAL ERROR ---")
        print("Please configure MISTRAL_ENDPOINT and MISTRAL_API_KEY before running.")
        return

    print("=" * 60)
    print(f"Agent Query: {query}")
    print("=" * 60)
    
    start = time.time()
    
    initial_state = {"messages": [HumanMessage(content=query)]}

    try:
        # Astream the execution of the graph
        async for step in assistant_agent.astream(initial_state):
            # Extract the step and print progress
            node_name, node_state = list(step.items())[0]

            if node_name == END:
                final_message = node_state['messages'][-1].content
                print("\n\n--- FINAL RESPONSE ---")
                print(final_message)
                return final_message
            
            # Print the output from the final LLM call (if not a tool call)
            if node_name == 'llm' and should_continue(node_state) == END:
                 # Print only if it's the final output of the LLM before END
                 last_message = node_state['messages'][-1]
                 print("\nAssistant:", end=" ", flush=True)
                 print(last_message.content)
                 return last_message.content

    except Exception as e:
        print(f"\n❌ An error occurred during agent execution: {e}")
            
    print(f"\n\n⏱️ Total Time: {time.time() - start:.2f}s")
    print("=" * 60)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Simple Nutrition Health Chatbot",
    page_icon="🥗"
)

st.markdown("""
<div style="
    padding: 20px; 
    border-radius: 15px; 
    background-color: #F0FFF0; 
    border: 1px solid #E0F5E0;
    text-align: center;
">
    <h1 style="color:#2E8B57; margin-bottom: 0;">🥦 Nutrition Expert</h1>
    <p style="margin-top: 5px; color:#3A5F0B;">
    </p>
</div>
""", unsafe_allow_html=True)

# --- SESSION STATE FOR CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- USER INPUT ---
user_input = st.chat_input("Ask a question...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # --- BOT THINKING (two spinners) ---
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching and thinking..."):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            response = loop.run_until_complete(run_agent(user_input))

        st.write(response)

    # Save bot message
    st.session_state.messages.append({"role": "assistant", "content": response})