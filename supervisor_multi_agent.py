from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from typing_extensions import TypedDict, Annotated

load_dotenv()

llm=ChatOpenAI(model_name="gpt-4o")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    research: str
    draft: str
    final_output: str



# --- Define Worker Agents ---

def research_agent(state: AgentState) -> dict:
    """Research specialist that gathers information."""
    messages = state["messages"]
    last_message = messages[-1].content

    response = llm.invoke([
        SystemMessage(content="""You are an expert researcher. Your job is to:
1. Identify the key aspects of the topic
2. Gather relevant facts, statistics, and examples
3. Organize your findings clearly

Output your research as structured notes."""),
        HumanMessage(content=f"Research the following topic thoroughly:\n\n{last_message}")
    ])

    return {"research": response.content}


def writing_agent(state: AgentState) -> dict:
    """Writing specialist that creates content."""
    research = state.get("research", "")

    response = llm.invoke([
        SystemMessage(content="""You are a skilled technical writer. Your job is to:
1. Transform research notes into a polished article
2. Use clear headings and logical structure
3. Make complex topics accessible
4. Include an introduction and conclusion"""),
        HumanMessage(content=f"Write a comprehensive article based on this research:\n\n{research}")
    ])

    return {"draft": response.content}


def editing_agent(state: AgentState) -> dict:
    """Editing specialist that reviews and improves content."""
    draft = state.get("draft", "")

    response = llm.invoke([
        SystemMessage(content="""You are a meticulous editor. Your job is to:
1. Fix any grammatical or stylistic issues
2. Improve clarity and readability
3. Ensure factual consistency
4. Enhance engagement and flow
5. Return the final polished version"""),
        HumanMessage(content=f"Review and improve this article:\n\n{draft}")
    ])

    return {"final_output": response.content}

def supervisor(state: AgentState) -> dict:
    """Supervisor that decides which agent should work next."""
    research = state.get("research", "")
    draft = state.get("draft", "")
    final_output = state.get("final_output", "")

    if not research:
        return {"next_agent": "researcher"}
    elif not draft:
        return {"next_agent": "writer"}
    elif not final_output:
        return {"next_agent": "editor"}
    else:
        return {"next_agent": "done"}


def route_to_agent(state: AgentState) -> Literal["researcher", "writer", "editor", "__end__"]:
    """Route to the next agent based on supervisor decision."""
    next_agent = state.get("next_agent", "")
    if next_agent == "done":
        return "__end__"
    return next_agent


# --- Build the Graph ---

workflow = StateGraph(AgentState)

workflow.add_node("supervisor",supervisor)
workflow.add_node("researcher", research_agent)
workflow.add_node("writer", writing_agent)
workflow.add_node("editor", editing_agent)

# Define edges
workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "researcher": "researcher",
        "writer": "writer",
        "editor": "editor",
        "__end__": END,
    }
)

# Workers always return to supervisor
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", "supervisor")
workflow.add_edge("editor", "supervisor")

# Compile
multi_agent = workflow.compile()

result = multi_agent.invoke({
    "messages": [HumanMessage(content="The future of AI agents in enterprise software")],
    "next_agent": "",
    "research": "",
    "draft": "",
    "final_output": "",
})

print(result["final_output"])
