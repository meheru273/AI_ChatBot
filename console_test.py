# console_test.py
# A simple script to test if the basic chatbot functionality works
# Run this before setting up the server to verify your LLM connection works

from typing import Annotated
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Set the API key directly if not in environment
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "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"

# Set up a simple chatbot without tools
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Create LLM
print("Creating ChatGroq instance...")
llm = ChatGroq(model_name="Gemma2-9b-It")  

# Define chatbot function
def chatbot(state: State):
    print("Calling LLM...")
    response = llm.invoke(state["messages"])
    print(f"Got response: {response.content}")
    return {"messages": [response]}

# Build graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Test with a simple message
print("\n=== Testing Simple Chatbot ===")
user_input = input("Enter a message to test (or press Enter for default): ") or "Tell me a short joke"
print(f"\nSending message: '{user_input}'")

# Format as expected by LangChain
messages = [("user", user_input)]

print("\nStreaming response:")
print("-" * 40)
events = graph.stream(
    {"messages": messages},
    stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()

print("-" * 40)
print("Test completed!")