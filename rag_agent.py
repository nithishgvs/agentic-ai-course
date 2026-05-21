from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Purpose of this file:
# Build a small RAG-powered product support agent.
#
# RAG flow:
# 1. Store product knowledge in Chroma as embeddings.
# 2. Expose a search tool that retrieves relevant product facts.
# 3. Give that tool to an agent.
# 4. The agent decides when to call the tool before answering the user.

# Load API keys from .env.
# Example: OPENAI_API_KEY=...
load_dotenv()

# Chat model used by the agent for reasoning and final answers.
llm = ChatOpenAI(model="gpt-4o")

# Embedding model converts product docs into vectors for semantic search.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Product knowledge base.
# In a real app, these could come from docs, PDFs, a website, or a database.
product_docs = [
    "The Pro Plan costs $29/month and includes unlimited API calls.",
    "The Starter Plan is free and includes 1000 API calls per month.",
    "The Enterprise Plan has custom pricing and dedicated support.",
    "All plans include email support with 24-hour response time.",
    "Pro and Enterprise plans include priority chat support.",
    "API rate limits: Starter 10 req/s, Pro 100 req/s, Enterprise unlimited.",
]

# Create a Chroma vector store from the product docs.
# Chroma stores:
# - the original text
# - the embedding vector for each text
vectorstore = Chroma.from_texts(product_docs, embeddings)


@tool
def search_knowledge_base(query: str) -> str:
    """Search based on the product knowledge base for the relevant information"""
    # Search Chroma for the 3 product docs most similar to the user's query.
    docs = vectorstore.similarity_search(query, k=3)

    # Convert the retrieved Document objects into one readable string.
    #
    # Instead of manually doing:
    # result += doc.page_content
    # result += "\n"
    #
    # join() builds the final string in one clean step.
    return "\n".join(doc.page_content for doc in docs)


@tool
def get_current_date() -> str:
    """Get the current date."""
    # Local import keeps datetime usage close to the small tool that needs it.
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")


# Create a RAG-powered agent.
#
# tools gives the agent capabilities beyond plain chat:
# - search_knowledge_base: search product docs
# - get_current_date: answer date-related questions
#
# system_prompt sets the agent's role and tells it to use the knowledge base.
agent = create_agent(
    model=llm,
    tools=[search_knowledge_base, get_current_date],
    system_prompt=(
        "You are a helpful product support agent. "
        "Use the knowledge base to answer questions accurately."
    ),
)

# Ask the agent a product question.
# The agent should decide to call search_knowledge_base because the answer
# lives in the product docs, not in the user's message.
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What's the rate limit on the Pro plan?"}
        ]
    }
)

# Print the last message, which is the agent's final answer.
print(response["messages"][-1].content)
# The agent searches the KB and responds: "The Pro Plan has a rate limit of 100 requests per second."
