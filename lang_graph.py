from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# 1. Define State - the data that flows through the graph.
#
# In LangGraph, every node receives the current state and returns updates to it.
# This TypedDict describes what keys are allowed in that state.
class AgentState(TypedDict):
    # messages is a list of LangChain message objects.
    #
    # Annotated[..., add_messages] tells LangGraph to append/merge new messages
    # instead of replacing the whole messages list.
    messages: Annotated[list[BaseMessage], add_messages]

    # current_step is a normal string field.
    # Since it does not have a reducer like add_messages, each node replaces it.
    current_step: str

    # result is also a normal string field.
    # step_two will replace the value written by step_one.
    result: str


# 2. Define Nodes - functions that process and update state.
#
# A node function receives the current state and returns a dictionary
# containing only the fields it wants to update.
def step_one(state: AgentState) -> dict:
    # This writes current_step and result into the graph state.
    return {"current_step": "step_one", "result": "Processed step one"}


def step_two(state: AgentState) -> dict:
    # This runs after step_one and replaces the previous result value.
    return {"current_step": "step_two", "result": "Processed step two"}


# 3. Build the Graph - connect nodes with edges.
#
# StateGraph creates a graph that uses AgentState as its shared state shape.
graph = StateGraph(AgentState)

# Register each Python function as a named graph node.
graph.add_node("step_one", step_one)
graph.add_node("step_two", step_two)

# Edges define the execution order:
# START -> step_one -> step_two -> END
graph.add_edge(START, "step_one")
graph.add_edge("step_one", "step_two")
graph.add_edge("step_two", END)

# 4. Compile and Run.
#
# compile() turns the graph definition into something executable.
app = graph.compile()

# invoke() starts the graph with the initial state below.
result = app.invoke({
    "messages": [],
    "current_step": "",
    "result": ""
})

# This prints "Processed step two" because step_two is the last node
# to update the result field.
print(result["result"])  # "Processed step two"
