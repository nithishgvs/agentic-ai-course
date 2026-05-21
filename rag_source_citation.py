from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load API keys from .env.
# Example: OPENAI_API_KEY=...
load_dotenv()

# Chat model used to generate the final answer.
llm = ChatOpenAI(model="gpt-4o")

# Embedding model used to convert text into vectors for Chroma.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a small Chroma vector store.
#
# texts are the document contents.
# metadatas are extra details attached to each document.
# Here, metadata stores where each document came from.
vectorstore = Chroma.from_texts(
    texts=["Returns accepted within 30 days.", "Free shipping over $50."],
    metadatas=[
        {"source": "returns-policy.pdf", "page": 1},
        {"source": "shipping-guide.pdf", "page": 3},
    ],
    embedding=embeddings,
)

# Convert Chroma into a retriever.
# k=3 means return the top 3 most similar documents for a question.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def retrieve_with_sources(question: str):
    # Search Chroma for documents related to the user's question.
    docs = retriever.invoke(question)

    # Build the context that will be sent to the LLM.
    #
    # Each document is formatted with its source metadata so the LLM can cite it.
    # Example:
    # [Source: returns-policy.pdf]
    # Returns accepted within 30 days.
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

    # Prompt tells the LLM to answer from context and cite sources.
    template = """Answer based on the context below. Cite your sources.

Context:
{context}

Question: {question}"""

    # Build a small chain:
    # prompt variables -> LLM -> plain string output
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # Send both the retrieved context and original question to the chain.
    answer = chain.invoke({"context": context, "question": question})

    # Also return sources programmatically, so the app does not depend only
    # on whether the LLM mentioned them correctly in text.
    sources = [doc.metadata.get("source") for doc in docs]

    return {"answer": answer, "sources": sources}


# Example run.
result = retrieve_with_sources("What is your shipping policy?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
