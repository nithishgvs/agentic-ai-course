from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o")

def researcher(topic: str) -> str:
    """Research agent gathers information."""
    response = llm.invoke([
        SystemMessage(content="You are a research assistant. Gather key facts and data about the topic. Be thorough and cite specific details."),
        HumanMessage(content=f"Research this topic: {topic}")
    ])
    return response.content

def writer(research: str, topic: str) -> str:
    """Writer agent creates content from research."""
    response = llm.invoke([
        SystemMessage(content="You are a professional writer. Create a well-structured article from the research provided. Use clear language and logical flow."),
        HumanMessage(content=f"Write an article about '{topic}' using this research:\n\n{research}")
    ])
    return response.content

def editor(article: str) -> str:
    """Editor agent reviews and improves the content."""
    response = llm.invoke([
        SystemMessage(content="You are a senior editor. Review this article for clarity, accuracy, grammar, and engagement. Return the improved version."),
        HumanMessage(content=f"Edit and improve this article:\n\n{article}")
    ])
    return response.content

# Sequential pipeline
topic = "The impact of AI agents on software development"
research = researcher(topic)
draft = writer(research, topic)
final_article = editor(draft)

print(final_article)