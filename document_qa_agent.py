# Document Q&A agent that loads PDFs, creates embeddings, and answers questions
# with citations.
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API keys from .env.
# Example: OPENAI_API_KEY=...
load_dotenv()

# Chat model used by the agent to reason and answer questions.
llm = ChatOpenAI(model="gpt-4o")

# Embedding model used to convert PDF chunks into vectors for Chroma.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# --- Step 1: Load and index documents ---


def build_knowledge_base(pdf_paths: list[str], persist_dir: str = "./qa_db") -> Chroma:
    """Load PDFs and create a searchable vector store."""
    all_chunks = []

    # Split long PDF pages into smaller chunks.
    #
    # chunk_size=1000 means each chunk is about 1000 characters.
    # chunk_overlap=200 means the next chunk repeats the last 200 characters
    # from the previous chunk. This helps avoid losing context at boundaries.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for pdf_path in pdf_paths:
        print(f"Loading pdf path: {pdf_path}")

        # PyPDFLoader reads the PDF and returns one Document per page.
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Split pages into smaller chunks that are easier to retrieve.
        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks created")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Building vector store...")

    # Create a persistent Chroma vector store.
    # Chroma stores each chunk, its embedding, and metadata like source/page.
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print("Knowledge base ready!\n")
    return vectorstore


# --- Step 2: Define tools ---

# Global vectorstore used by the tools below.
# main() initializes this after loading PDFs.
vectorstore = None  # Will be initialized in main()


@tool
def search_documents(query: str) -> str:
    """Search the loaded documents for information relevant to the query.

    Returns the most relevant passages with source information.
    """

    if vectorstore is None:
        return "No Documents are loaded"

    # Search for the 4 chunks most similar to the user's query.
    results = vectorstore.similarity_search(query, k=4)

    if not results:
        return "No relevant information found."

    # Format each retrieved chunk with citation metadata.
    # The agent sees this text and can cite source/page in its answer.
    output = []
    for index, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        output.append(f"[Result {index} | Source: {source}, Page: {page}]")
        output.append(doc.page_content)
        output.append("")

    return "\n".join(output)


@tool
def list_loaded_documents() -> str:
    """List all documents that have been loaded into the knowledge base."""
    if vectorstore is None:
        return "No documents loaded."

    # vectorstore.get() returns stored docs and metadata.
    # We only need metadata here so we can list unique source filenames.
    docs = vectorstore.get()
    sources = set()
    for metadata in docs["metadatas"]:
        sources.add(metadata.get("source", "unknown"))

    return f"Loaded documents: {', '.join(sorted(sources))}"


# --- Step 3: Create the agent ---


def create_qa_agent():
    """Create a document Q&A agent with RAG capabilities."""
    # This system prompt controls agent behavior.
    # The most important rule is: search documents before answering.
    system_prompt = """You are a helpful document Q&A assistant. Your job is to answer
questions based on the loaded documents.

Rules:
1. ALWAYS search the documents before answering a question.
2. Base your answers ONLY on information found in the documents.
3. If the documents don't contain the answer, say so clearly.
4. Always cite which document and page the information came from.
5. If a question is ambiguous, search for multiple interpretations.

Be thorough, accurate, and helpful."""
    agent = create_agent(
        model=llm,
        tools=[search_documents, list_loaded_documents],
        system_prompt=system_prompt,
    )
    return agent


# --- Step 4: Interactive chat loop ---


def main():
    global vectorstore

    # Load your PDFs here.
    # This list is filled automatically from ./docs below.
    pdf_files = []

    # Check for PDFs in a ./docs directory.
    docs_dir = "./docs"
    if os.path.exists(docs_dir):
        pdf_files = [
            os.path.join(docs_dir, f)
            for f in os.listdir(docs_dir)
            if f.endswith(".pdf")
        ]

    if not pdf_files:
        print("No PDF files found in ./docs directory.")
        print("Create a ./docs folder and add PDF files, then restart.")
        print("\nStarting with an empty knowledge base for demo purposes.\n")

        # Create an empty persistent Chroma DB so the tools still have a
        # vectorstore object to use.
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory="./qa_db",
        )
    else:
        # Build Chroma from all PDFs found in ./docs.
        vectorstore = build_knowledge_base(pdf_files)

    agent = create_qa_agent()

    print("Document Q&A Agent Ready!")
    print("Ask questions about your documents. Type 'quit' to exit.\n")

    message_history = []

    # Simple terminal chat loop.
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to chat history.
        message_history.append({"role": "user", "content": user_input})

        # Send the whole message history to the agent.
        # The agent can call search_documents or list_loaded_documents as needed.
        response = agent.invoke({"messages": message_history})

        # The final message is the agent's answer.
        assistant_message = response["messages"][-1].content
        print(f"\nAgent: {assistant_message}\n")

        # Save the assistant response so the next turn has conversation context.
        message_history.append({"role": "assistant", "content": assistant_message})


if __name__ == "__main__":
    main()
