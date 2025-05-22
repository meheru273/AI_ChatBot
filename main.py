from typing import Annotated
from typing_extensions import TypedDict
import os
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

#working with tools
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.prebuilt import ToolNode,tools_condition

from dotenv import load_dotenv
load_dotenv()


#arxiv and wikipedia wrapper
arvix_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arvix_tool = ArxivQueryRun(api_wrapper=arvix_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# res = wiki_tool.invoke({"query": "What is the capital of USA?"})
# print(res) 

tools = [wiki_tool,arvix_tool]

#langgraph applications 

class State(TypedDict):
    messages:Annotated[list,add_messages]

graph_builder = StateGraph(State)

llm = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"),model_name="Gemma2-9b-It")

llm_with_tools=llm.bind_tools(tools=tools)

def chatbot(state: State):
   return {"messages":[llm_with_tools.invoke(state["messages"])]} 

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools",tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)
graph_builder.add_edge("tools","chatbot")
graph_builder.add_edge("chatbot",END)

graph = graph_builder.compile()

user_input = "What is the name of the capital of France"

events = graph.stream(
    {"messages":[("user",user_input)]},
    stream_mode="values"
)

for event in events:
    event["messages"][-1].pretty_print()