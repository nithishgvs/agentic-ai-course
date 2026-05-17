from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Main LLM used to draft the email.
llm = ChatOpenAI(model="gpt-4o")


# ApprovalState is the shared state passed between all graph nodes.
# Java-ish mental model: this is like a DTO that each step can read/update.
class ApprovalState(TypedDict):
    # messages is the conversation history.
    # Annotated[..., add_messages] tells LangGraph to append new messages
    # instead of replacing the whole list.
    messages: Annotated[list[BaseMessage], add_messages]

    # Normal fields below are replaced by the latest node that updates them.
    draft_email: str
    approved: bool
    sent: bool


def draft_email_node(state: ApprovalState) -> dict:
    """Draft an email based on the user's request."""
    # Ask the LLM to create an email from the user's message.
    response = llm.invoke([
        SystemMessage(content=(
            "Draft a professional email based on the user's request. "
            "Return ONLY the email content (subject and body)."
        )),
        # Unpack existing messages into the list passed to the LLM.
        # Java-ish mental model: addAll(state.messages).
        *state["messages"]
    ])

    draft = response.content
    print(f"\n--- DRAFT EMAIL ---\n{draft}\n-------------------\n")

    # This node only updates draft_email.
    return {"draft_email": draft}


def review_node(state: ApprovalState) -> dict:
    """Present the draft for human review."""
    # This node uses an interrupt to pause execution
    # In a real app, this would present a UI for approval
    print("\nThis email requires your approval before sending.")
    print(f"\nDraft:\n{state['draft_email']}\n")

    # Simple command-line human approval step.
    approval = input("Approve this email? (yes/no): ").strip().lower()
    approved = approval == "yes"

    # The next route depends on this approved value.
    return {"approved": approved}


def send_email_node(state: ApprovalState) -> dict:
    """Send the approved email."""
    print("[Email] Sending email...")
    # In production, this would call an email API

    # Mark the email as sent and append an AI message to the conversation.
    return {
        "sent": True,
        "messages": [AIMessage(content="Email sent successfully.")]
    }


def reject_node(state: ApprovalState) -> dict:
    """Handle rejected email."""
    # Mark the email as not sent and append an AI message explaining the outcome.
    return {
        "sent": False,
        "messages": [AIMessage(content="Email was not sent. Let me know if you'd like to revise it.")]
    }


def route_after_review(state: ApprovalState) -> Literal["send", "reject"]:
    """Route based on approval status."""
    # If the human approved the draft, go to the send node.
    # Otherwise, go to the reject node.
    if state.get("approved", False):
        return "send"
    return "reject"


# Build the graph.
# Flow:
#
# START -> draft -> review -> approved? -> send   -> END
#                              |
#                              +-> reject -> END
workflow = StateGraph(ApprovalState)

# Register graph node names with their Python functions.
workflow.add_node("draft", draft_email_node)
workflow.add_node("review", review_node)
workflow.add_node("send", send_email_node)
workflow.add_node("reject", reject_node)

# Fixed part of the flow: always draft first, then review.
workflow.add_edge(START, "draft")
workflow.add_edge("draft", "review")

# Conditional routing:
# after review, call route_after_review(state).
# If it returns "send", run the send node.
# If it returns "reject", run the reject node.
workflow.add_conditional_edges(
    "review",
    route_after_review,
    {"send": "send", "reject": "reject"}
)

# Both final paths end the graph.
workflow.add_edge("send", END)
workflow.add_edge("reject", END)

# A checkpointer saves graph state between pauses/resumes.
# MemorySaver keeps that state in memory for this Python process.
checkpointer = MemorySaver()

# Compile with interrupt_before=["send"].
# This tells LangGraph: if the next node is "send", pause before running it.
# That gives a human/system a chance to inspect state before the email is sent.
email_agent = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["send"]
)

# config contains a thread_id so the checkpointer knows which saved state
# belongs to this run. Think of it like a conversation/session id.
config = {"configurable": {"thread_id": "email-1"}}

# First invocation:
# START -> draft -> review -> maybe pause before send.
result = email_agent.invoke(
    {
        "messages": [HumanMessage(content="Send a follow-up email to the client about the project deadline")],
        "draft_email": "",
        "approved": False,
        "sent": False,
    },
    config=config,
)

# If approved=True, the graph pauses before the send node because of interrupt_before.
# At this point, the draft is available in the saved state.
print(f"Draft: {result['draft_email']}")

# Resume from the checkpoint.
# Passing None means "continue from the saved state for this thread_id".
# If the graph was paused before send, this will continue into send_email_node().
email_agent.invoke(None, config=config)  # Resumes from checkpoint

# To reject in a more advanced app, you would update the saved state so
# approved=False, then resume through the reject path.
