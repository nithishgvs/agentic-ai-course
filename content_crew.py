from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from content_classifier_agent import workflow

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")


class ContentState(TypedDict):
    topic: str
    research_notes: str
    first_draft: str
    editor_feedback: str
    final_article: str
    revision_count: int
    status: str


# ---- Agent Definitions---------


def research_node(state: ContentState) -> dict:
    """Researcher agent: gathers comprehensive information on the topic."""
    topic = state["topic"]
    print(f"\n[Researcher] Investigating: {topic}")

    response = llm.invoke([
        SystemMessage(content="""You are an expert research analyst. Your task is to:

    1. Break down the topic into key subtopics
    2. Provide specific facts, statistics, and recent developments
    3. Identify expert opinions and notable quotes
    4. Note any controversies or different perspectives
    5. List potential sources and references

    Structure your output as clear, organized research notes with headers.
    Be thorough but focused. Aim for depth over breadth."""),
        HumanMessage(content=f"Conduct thorough research on: {topic}")
    ])

    print("[Researcher] Research complete.")
    return {"research_notes": response.content, "status": "researched"}


def writer_node(state: ContentState) -> dict:
    """Writer agent: creates a polished article from research notes."""

    research = state["research_notes"]
    topic = state["topic"]
    feedback = state.get("editor_feedback", "")
    print(f"\n[Writer] {'Revising' if feedback else 'Drafting'} article...")

    prompt_parts = [f"Topic: {topic}\n\nResearch Notes:\n{research}"]
    if feedback:
        prompt_parts.append(f"\n\nEditor Feedback to Address:\n{feedback}")
        prompt_parts.append(f"\n\nPrevious Draft:\n{state.get('first_draft', '')}")

    response = llm.invoke([
        SystemMessage(content="""You are a talented technical writer. Create a compelling article that:

    1. Opens with a strong hook that draws readers in
    2. Uses clear headings and subheadings for structure
    3. Explains complex concepts with relatable analogies
    4. Includes specific examples and data points from the research
    5. Maintains a professional yet engaging tone
    6. Ends with a thought-provoking conclusion

    If editor feedback is provided, revise the article to address all feedback points.
    The article should be 600-800 words."""),
        HumanMessage(content="\n".join(prompt_parts))
    ])

    print("[Writer] Draft complete.")
    return {"first_draft": response.content, "status": "drafted"}


def editor_node(state: ContentState) -> dict:
    """Editor agent: reviews the draft and provides feedback or approves."""
    draft = state["first_draft"]
    revision_count = state.get("revision_count", 0)

    print(f"\n[Editor] Reviewing draft (revision #{revision_count + 1})...")

    response = llm.invoke([
        SystemMessage(content=f"""You are a senior editor with high standards. Review this article carefully.

This is revision #{revision_count + 1}. If this is revision 2 or higher, be more lenient.

Evaluate:
1. Factual accuracy and consistency
2. Writing quality and clarity
3. Structure and flow
4. Engagement and readability
5. Grammar and style

If the article meets your standards (or this is revision 2+), respond with:
APPROVED

Followed by the final version with any minor corrections.

If it needs significant revision (only on first review), respond with:
NEEDS_REVISION

Followed by specific, actionable feedback."""),
        HumanMessage(content=f"Review this article:\n\n{draft}")
    ])

    content = response.content

    if "APPROVED" in content:
        # Extract the final version (everything after APPROVED)
        final = content.split("APPROVED", 1)[1].strip()
        print("[Editor] Article approved!")
        return {
            "final_article": final if final else draft,
            "editor_feedback": "",
            "revision_count": revision_count + 1,
            "status": "approved",
        }
    else:
        # Extract feedback
        feedback = content.split("NEEDS_REVISION", 1)[1].strip() if "NEEDS_REVISION" in content else content
        print("[Editor] Requesting revisions.")
        return {
            "editor_feedback": feedback,
            "revision_count": revision_count + 1,
            "status": "needs_revision",
        }


# --- Routing Logic ---
def should_continue(state: ContentState) -> Literal["writer", "__end__"]:
    """Decide if the article needs more revision or is ready."""
    if state.get("status") == "approved":
        return "__end__"
    return "writer"


# --- Build the Graph ---

workflow = StateGraph(ContentState)

# Add nodes
workflow.add_node("researcher", research_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

# Define flow
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "editor")

# Conditional: editor either approves or sends back for revision
workflow.add_conditional_edges("editor", should_continue, {
    "writer": "writer",
    "__end__": END,
})

# Compile
content_crew = workflow.compile()

# --- Run ---

def main():
    print("=" * 60)
    print("   Content Creation Crew")
    print("   Researcher -> Writer -> Editor")
    print("=" * 60)

    topic = input("\nEnter a topic for the article: ").strip()
    if not topic:
        topic = "How AI agents are transforming customer service"

    print(f"\nCreating article about: {topic}")
    print("-" * 60)

    result = content_crew.invoke({
        "messages": [],
        "topic": topic,
        "research_notes": "",
        "first_draft": "",
        "editor_feedback": "",
        "final_article": "",
        "revision_count": 0,
        "status": "started",
    })

    print("\n" + "=" * 60)
    print("   FINAL ARTICLE")
    print("=" * 60)
    print(result["final_article"])
    print("\n" + "=" * 60)
    print(f"Completed after {result['revision_count']} revision(s)")

if __name__ == "__main__":
    main()

