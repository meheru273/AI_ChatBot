# server.py - Based on your working console code
from typing import Annotated
from typing_extensions import TypedDict
import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Tools imports
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio



# Load environment variables
load_dotenv()


# Set up tools
arvix_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arvix_tool = ArxivQueryRun(api_wrapper=arvix_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

tools = [wiki_tool, arvix_tool]

# Set up LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
llm = ChatGroq(model_name="Gemma2-9b-It")  # Will use env var automatically
llm_with_tools = llm.bind_tools(tools=tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Set up FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def read_root():
    return {"message": "Server is running"}

@app.post("/chat")
async def chat_post(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])
        print(f"Received messages: {json.dumps(messages, indent=2)}")
        
        # Convert frontend messages to the format your graph expects
        graph_messages = []
        for msg in messages:
            if msg["role"] == "user":
                graph_messages.append(("user", msg["content"]))
            elif msg["role"] == "assistant":
                graph_messages.append(("assistant", msg["content"]))
            elif msg["role"] == "system":
                graph_messages.append(("system", msg["content"]))
        
        print(f"Converted to graph messages: {graph_messages}")
        
        # Stream responses
        async def event_stream():
            try:
                # Use the graph to generate responses
                for event in graph.stream(
                    {"messages": graph_messages},
                    stream_mode="values"
                ):
                    try:
                        last_message = event["messages"][-1]
                        # Handle different message formats
                        if hasattr(last_message, "type"):
                            role = last_message.type
                            content = last_message.content
                        elif isinstance(last_message, dict):
                            role = last_message.get("role", "assistant")
                            content = last_message.get("content", "")
                        else:
                            # Fallback for other formats
                            role = "assistant"
                            content = str(last_message)
                            
                        print(f"Sending: {role} - {content}")
                        yield f"data: {json.dumps({'role': role, 'content': content})}\n\n"
                    except Exception as e:
                        print(f"Error processing event: {str(e)}")
                        continue
                
                yield "event: done\ndata: {}\n\n"
            except Exception as e:
                print(f"Error in stream: {str(e)}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'role': 'assistant', 'content': f'Error: {str(e)}'})}\n\n"
                yield "event: done\ndata: {}\n\n"
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
