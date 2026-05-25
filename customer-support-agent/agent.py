from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from typing_extensions import TypedDict

load_dotenv()

from knowledge_base import get_knowledge_base
from tools import *

llm = ChatOpenAI(model="gpt-4o")
fast_llm = ChatOpenAI(model="gpt-4o-mini")  # For classification (cheaper)


class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    classification: str  # "general", "technical", "sensitive", "escalate"
    retrieved_context: str
    ticket_id: str
    escalated: bool
    response: str


# Initialize knowledge base
kb = get_knowledge_base()


# --- Node Definitions ---

def classify_node(state: SupportState) -> dict:
    """Classify the user's message to determine routing."""
    messages = state["messages"]
    last_message = messages[-1].content

    response = fast_llm.invoke([
        SystemMessage(content="""Classify this customer support message into ONE category.
Respond with ONLY the category name, nothing else.

Categories:
- general: Product questions, pricing, features, how-to questions
- technical: Bug reports, API errors, integration issues, troubleshooting
- sensitive: Billing disputes, account deletion, legal concerns, complaints about service
- escalate: Threats, harassment, requests to speak to a manager, urgent safety issues

Message to classify:"""),
        HumanMessage(content=last_message)
    ])

    # Validate classification
    classification = response.content.strip().lower()
    valid = {"general", "technical", "sensitive", "escalate"}
    if classification not in valid:
        classification = "general"

    print(f"[Classifier] Category: {classification}")
    return {"classification": classification}


def rag_search_node(state: SupportState) -> dict:
    """Search the knowledge base for relevant information."""
    messages = state["messages"]
    last_message = messages[-1].content
    print("[RAG] Searching knowledge base...")

    results = kb.similarity_search(last_message, k=4)

    if results:
        context = "\n\n".join(
            f"- {doc.page_content}" for doc in results
        )
        print(f"[RAG] Found {len(results)} relevant documents")
    else:
        context = "No relevant documentation found."
        print("[RAG] No relevant documents found")

    return {"retrieved_context": context}


def respond_node(state: SupportState) -> dict:
    """Generate a helpful response using the retrieved context."""
    messages = state["messages"]
    context = state.get("retrieved_context", "")
    classification = state.get("classification", "general")
    ticket_id = state.get("ticket_id", "")

    system_content = f"""You are a friendly, professional customer support agent for TechCorp.

Use the following knowledge base context to answer the customer's question:

{context}

Rules:
1. Be helpful, empathetic, and professional.
2. Only provide information that is supported by the context above.
3. If you don't know the answer, say so honestly and suggest contacting support.
4. Keep responses concise but thorough.
5. If a support ticket was created, mention the ticket ID: {ticket_id}."""

    response = llm.invoke([
        SystemMessage(content=system_content),
        # *messages unpacks the existing list into this new list.
        # Java-ish mental model: finalMessages.addAll(messages).
        # Without *, this would create a nested list, which llm.invoke() does not want.
        *messages
    ])

    print("[Response] Generated reply")
    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def create_ticket_node(state: SupportState) -> dict:
    """Create a support ticket for technical issues."""
    messages = state["messages"]
    last_message = messages[-1].content
    classification = state.get("classification", "technical")
    context = state.get("retrieved_context", "")
    ticket_id = create_ticket(customer_message=last_message,
                              context=context,
                              classification=classification)
    print(f"[Ticket] Created: {ticket_id}")
    return {"ticket_id": ticket_id}


def escalation_node(state: SupportState) -> dict:
    """Escalate to a human agent for sensitive or urgent issues."""
    messages = state["messages"]
    last_message = messages[-1].content
    classification = state.get("classification", "")

    ticket_id = escalate_to_human(
        customer_message=last_message,
        reason=f"Classified as: {classification}",
    )

    escalation_message = (
        "I understand this is important to you. I've escalated your case to a "
        f"senior support specialist (Reference: {ticket_id}). A human agent will "
        "reach out to you within the next 2 hours. Is there anything else I can "
        "help with in the meantime?"
    )

    print(f"[Escalation] Escalated with ticket: {ticket_id}")
    return {
        "ticket_id": ticket_id,
        "escalated": True,
        "response": escalation_message,
        "messages": [AIMessage(content=escalation_message)],
    }


# --- Routing Logic ---

def route_by_classification(state: SupportState) -> Literal["rag_search", "escalation"]:
    """Route based on message classification."""
    classification = state.get("classification", "general")

    if classification in ("sensitive", "escalate"):
        return "escalation"
    return "rag_search"


def route_after_rag(state: SupportState) -> Literal["create_ticket", "respond"]:
    """After RAG search, decide if we need a ticket."""
    classification = state.get("classification", "general")

    if classification == "technical":
        return "create_ticket"
    return "respond"


# --- Build the Graph ---

workflow = StateGraph(SupportState)

# Add nodes
workflow.add_node("classify", classify_node)
workflow.add_node("rag_search", rag_search_node)
workflow.add_node("respond", respond_node)
workflow.add_node("create_ticket", create_ticket_node)
workflow.add_node("escalation", escalation_node)

# Define edges
workflow.add_edge(START, "classify")

workflow.add_conditional_edges(
    "classify",
    route_by_classification,
    {
        "rag_search": "rag_search",
        "escalation": "escalation",
    }
)

workflow.add_conditional_edges(
    "rag_search",
    route_after_rag,
    {
        "create_ticket": "create_ticket",
        "respond": "respond",
    }
)

workflow.add_edge("create_ticket", "respond")
workflow.add_edge("respond", END)
workflow.add_edge("escalation", END)

# Compile the agent
support_agent = workflow.compile()

# Session memory store
session_store: dict[str, list[BaseMessage]] = {}

def chat(session_id: str, user_message: str) -> str:
    """Handle a chat message with session memory."""

    # Get or create session history
    if session_id not in session_store:
        session_store[session_id] = []

    history = session_store[session_id]
    history.append(HumanMessage(content=user_message))

    # Invoke the agent with full history
    result = support_agent.invoke({
        "messages": history,
        "classification": "",
        "retrieved_context": "",
        "ticket_id": "",
        "escalated": False,
        "response": "",
    })

    # Extract response and update history
    response = result["response"]
    history.append(AIMessage(content=response))

    # Trim history to prevent unbounded growth
    max_messages = 20
    if len(history) > max_messages:
        session_store[session_id] = history[-max_messages:]

    return response