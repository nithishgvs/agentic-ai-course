from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load API keys from .env.
# Example: OPENAI_API_KEY=...
load_dotenv()

# Chat model used to generate the final answer.
llm = ChatOpenAI(model="gpt-4o")

# Embedding model converts text into vectors.
# Vector databases compare these vectors to find semantically similar text.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# RAG = Retrieval Augmented Generation.
# Instead of asking the LLM from memory only, we first retrieve relevant context
# from our own documents, then give that context to the LLM.

# These strings are the small knowledge base for this example.
# In a real app, these might come from PDFs, websites, docs, or database rows.
documents = [
    "LangGraph is a library for building stateful multi-agent applications.",
    "LangGraph uses a graph-based approach where nodes are functions and edges define flow.",
    "State in LangGraph is a TypedDict that flows through the graph.",
    "LangGraph supports conditional edges for dynamic routing between nodes.",
    "Human-in-the-loop can be implemented using interrupt_before in LangGraph.",
]

# Create a Chroma vector store from the text documents.
# Chroma stores each document plus its embedding vector.
vectorstore = Chroma.from_texts(documents, embeddings)

# Convert the vector store into a retriever.
# k=3 means: for each question, return the top 3 most similar documents.
# search_kwargs → dictionary of extra search settings
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG prompt template.
# The final prompt will contain:
# - context: documents retrieved from Chroma
# - question: the user's question
template = """Answer the question using only the following context.
If the context contains a relevant fact, answer with that fact in a concise way.
If the context has no relevant facts for the question, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

# ChatPromptTemplate turns the template string into a LangChain prompt object.
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    # The retriever returns Document objects.
    # doc.page_content contains the text for each retrieved document.
    # This joins the retrieved docs into one context string for the prompt.
    return "\n\n".join(doc.page_content for doc in docs)


def debug_retrieval(question: str) -> None:
    # Debug helper:
    # Before blaming the LLM answer, first check what documents were retrieved.
    # If the right document is missing here, the issue is retrieval.
    # If the right document is present here, the issue is likely the prompt/LLM step.
    docs = retriever.invoke(question)

    print("\n--- Retrieved Documents ---")
    if not docs:
        print("No documents were retrieved.")

    for index, doc in enumerate(docs, start=1):
        print(f"\nDocument {index}:")
        print(doc.page_content)

    print("\n--- Formatted Context Sent To Prompt ---")
    print(format_docs(docs))
    print("--- End Context ---\n")


# Build the RAG chain.
#
# Java-ish mental model:
# 1. Take the user's question.
# 2. Send it to retriever to get matching docs.
# 3. Format those docs into a context string.
# 4. Put context + question into the prompt.
# 5. Send the prompt to the LLM.
# 6. Convert the LLM message response into a plain string.
rag_chain = (
    # This dictionary creates the input variables needed by the prompt:
    # - "context" comes from retriever | format_docs
    # - "question" is the original user input passed through unchanged
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Ask a question.
# The chain will retrieve relevant LangGraph documents before answering.
question = "How does state work in LangGraph?"
debug_retrieval(question)

answer = rag_chain.invoke(question)
print("--- Final Answer ---")
print(answer)
# => "State in LangGraph is a TypedDict that flows through the graph..."
