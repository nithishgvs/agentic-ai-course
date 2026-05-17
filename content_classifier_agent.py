import os

# Literal: restricts a type hint to specific string values.
# TypedDict: describes the shape of a dictionary.
# Annotated: attaches extra metadata to a type, used by LangGraph reducers.
from typing import Literal, TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Message classes are the standard LangChain chat message types.
# HumanMessage = user input, SystemMessage = instructions, AIMessage = model output.
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

# StateGraph is the LangGraph builder.
# START and END are special markers for where the graph begins and finishes.
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Reads values from the .env file and puts them into environment variables.
# Example: OPENAI_API_KEY=...
load_dotenv()

# Main model used for the actual customer-facing response.
llm = ChatOpenAI(model="gpt-4o")

# Smaller/faster model used only for classification.
# Classification is simple, so it does not need the larger model.
fast_llm = ChatOpenAI(model="gpt-4o-mini")


# RouterState is the shared data object passed between graph nodes.
# Java-ish mental model: this is similar to defining a DTO shape.
class RouterState(TypedDict):
    # Annotated lets us attach extra instructions to a type.
    # The real type is list[BaseMessage].
    # The extra instruction is add_messages, which tells LangGraph to append
    # new messages to the existing list instead of replacing the whole list.
    messages: Annotated[list[BaseMessage], add_messages]

    # These are normal string fields. Since they do not use Annotated with a
    # reducer, the latest node update replaces the previous value.
    category: str
    response: str


# --- Nodes ---

def classify_node(state: RouterState) -> dict:
    """Classify the incoming message."""
    # Get the latest user message from the messages list.
    # [-1] means "last item in the list".
    last_message = state["messages"][-1].content

    # Ask the fast model to classify the message into exactly one category.
    response = fast_llm.invoke([
        SystemMessage(content=(
            "Classify this message into ONE category. "
            "Respond with ONLY the category name.\n"
            "Categories: question, complaint, feature_request, praise"
        )),
        HumanMessage(content=last_message)
    ])

    # Normalize the model output so routing is easier.
    # strip() removes spaces/newlines, lower() makes it lowercase.
    category = response.content.strip().lower()

    # Safety check: if the model returns something unexpected,
    # default to "question" so the graph still has a valid route.
    valid_categories = {"question", "complaint", "feature_request", "praise"}
    if category not in valid_categories:
        category = "question"

    print(f"[Classifier] Category: {category}")

    # Return only the field this node updates.
    # LangGraph merges this into the overall RouterState.
    return {"category": category}


def handle_question(state: RouterState) -> dict:
    """Handle a question with a helpful answer."""
    # This node is chosen when category == "question".
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Answer the question clearly and concisely."),
        # *state["messages"] unpacks the list of messages into this list.
        # Example: [SystemMessage(...), *[HumanMessage(...)]]
        # becomes: [SystemMessage(...), HumanMessage(...)]
        # Without *, it would create a nested list, which llm.invoke() does not want.
        *state["messages"]
    ])

    # response is easy to return to the caller.
    # messages appends the AI reply to the conversation history.
    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)]
    }


def handle_complaint(state: RouterState) -> dict:
    """Handle a complaint with empathy and a solution."""
    # This node is chosen when category == "complaint".
    response = llm.invoke([
        SystemMessage(content=(
            "You are a customer service agent. The user has a complaint. "
            "Respond with empathy, acknowledge the issue, and offer a solution or next steps."
        )),
        # Unpack the existing conversation messages into the list passed to the LLM.
        *state["messages"]
    ])

    # Same return pattern as the other handlers:
    # save final text in response and append the AI message to messages.
    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)]
    }


def handle_feature_request(state: RouterState) -> dict:
    """Handle a feature request by acknowledging and logging it."""
    # This node is chosen when category == "feature_request".
    response = llm.invoke([
        SystemMessage(content=(
            "You are a product manager assistant. The user is requesting a feature. "
            "Thank them for the suggestion, ask clarifying questions if needed, "
            "and let them know the feedback has been logged."
        )),
        # Unpack the existing conversation messages into the list passed to the LLM.
        *state["messages"]
    ])

    # The returned messages list is appended because RouterState.messages
    # uses Annotated[..., add_messages].
    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)]
    }


def handle_praise(state: RouterState) -> dict:
    """Handle positive feedback graciously."""
    # This node is chosen when category == "praise".
    response = llm.invoke([
        SystemMessage(content=(
            "The user is giving positive feedback. Thank them warmly "
            "and let them know their feedback motivates the team."
        )),
        # Unpack the existing conversation messages into the list passed to the LLM.
        *state["messages"]
    ])

    # Store the final answer and add it to the message history.
    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)]
    }


# --- Routing ---

def route_by_category(state: RouterState) -> Literal[
    "handle_question", "handle_complaint",
    "handle_feature_request", "handle_praise"
]:
    """Route to the appropriate handler based on classification."""
    # Literal is a type hint that says this function should only return one
    # of the exact string values listed above.
    # Java-ish mental model: it is like saying the return value must be one of
    # a small fixed set of allowed constants.
    category = state.get("category", "question")

    # If category is "complaint", this returns "handle_complaint".
    # That returned string must match one of the keys in add_conditional_edges().
    return f"handle_{category}"


# --- Build Graph ---

# Graph picture:
#
#                         +-------------------+
#                         |       START       |
#                         +---------+---------+
#                                   |
#                                   v
#                         +-------------------+
#                         |     classify      |
#                         | classify_node()   |
#                         +---------+---------+
#                                   |
#                    route_by_category(state["category"])
#                                   |
#        +--------------------------+--------------------------+
#        |                          |                          |
#        v                          v                          v
# +-----------------+      +------------------+      +------------------------+
# | handle_question |      | handle_complaint |      | handle_feature_request |
# +--------+--------+      +---------+--------+      +-----------+------------+
#          |                         |                           |
#          v                         v                           v
#       +-----+                   +-----+                     +-----+
#       | END |                   | END |                     | END |
#       +-----+                   +-----+                     +-----+
#
#        +----------------+
#        | handle_praise  |
#        +-------+--------+
#                |
#                v
#              +-----+
#              | END |
#              +-----+
#
# Execution path example:
# User message -> classify -> one matching handler -> END

workflow = StateGraph(RouterState)

# Register node names with the Python functions they should run.
# The string names are what edges refer to.
workflow.add_node("classify", classify_node)
workflow.add_node("handle_question", handle_question)
workflow.add_node("handle_complaint", handle_complaint)
workflow.add_node("handle_feature_request", handle_feature_request)
workflow.add_node("handle_praise", handle_praise)

# The first real node after START is always classify.
workflow.add_edge(START, "classify")

# add_conditional_edges() creates dynamic routing.
#
# After the "classify" node runs, LangGraph calls route_by_category(state).
# That function reads state["category"] and returns a route name such as
# "handle_question" or "handle_complaint".
#
# The dictionary below maps each possible route name to the actual graph node
# that should run next.
workflow.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "handle_question": "handle_question",
        "handle_complaint": "handle_complaint",
        "handle_feature_request": "handle_feature_request",
        "handle_praise": "handle_praise",
    }
)

workflow.add_edge("handle_question", END)
workflow.add_edge("handle_complaint", END)
workflow.add_edge("handle_feature_request", END)
workflow.add_edge("handle_praise", END)

# compile() freezes/builds the graph so it can be executed with invoke().
router_agent = workflow.compile()


# --- Test It ---

def process_message(message: str) -> str:
    # Start the graph with one user message and empty fields for values
    # that will be filled by the classifier and handler nodes.
    result = router_agent.invoke({
        "messages": [HumanMessage(content=message)],
        "category": "",
        "response": "",
    })

    # The selected handler writes the final answer into result["response"].
    return result["response"]


if __name__ == "__main__":
    # These examples exercise all four routes:
    # question, complaint, feature_request, and praise.
    test_messages = [
        "How do I reset my password?",
        "Your app crashed three times today and I lost my work!",
        "It would be great if you added dark mode.",
        "I love this product! Best tool I've used.",
    ]

    for msg in test_messages:
        print(f"\nUser: {msg}")
        response = process_message(msg)
        print(f"Agent: {response}")
        print("-" * 60)
