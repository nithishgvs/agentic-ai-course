"""
Knowledge Base Module

This module manages the vector database for customer support documentation.
It uses ChromaDB for vector storage and OpenAI embeddings for semantic search.

The knowledge base contains:
- Product pricing and plans
- Feature documentation
- Technical API information
- Company policies
- Troubleshooting guides
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize OpenAI embeddings for vector similarity search
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample product knowledge base - in production, load from files or CMS
PRODUCT_DOCS = [
    # --- Pricing Information ---
    "The Starter plan is free and includes 1,000 API calls per month with community support.",
    "The Pro plan costs $49/month and includes 50,000 API calls with priority email support.",
    "The Enterprise plan has custom pricing with unlimited API calls and dedicated account manager.",
    "All paid plans come with a 14-day free trial. No credit card required for the trial.",
    "Annual billing gives a 20% discount on Pro and Enterprise plans.",

    # --- Product Features ---
    "Our API supports REST and GraphQL endpoints for all plan levels.",
    "Webhook notifications are available on Pro and Enterprise plans.",
    "Rate limits: Starter 10 req/s, Pro 100 req/s, Enterprise 1000 req/s.",
    "Data export is available in CSV, JSON, and Parquet formats on all plans.",
    "Custom integrations with Salesforce, HubSpot, and Slack are Enterprise-only features.",

    # --- Technical Documentation ---
    "API authentication uses Bearer tokens. Generate tokens in the dashboard under Settings > API Keys.",
    "The API base URL is https://api.techcorp.io/v2/ for all endpoints.",
    "Rate limit errors return HTTP 429. Implement exponential backoff for retries.",
    "Webhook payloads are signed with HMAC-SHA256. Verify the X-Signature header.",
    "SDK libraries are available for Python, JavaScript, Go, and Ruby.",

    # --- Company Policies ---
    "Refunds are available within 30 days of purchase for annual plans.",
    "Monthly plans can be cancelled at any time with no cancellation fee.",
    "Data retention: We store customer data for 90 days after account deletion.",
    "Our SLA guarantees 99.9% uptime for Pro plans and 99.99% for Enterprise.",
    "GDPR and SOC 2 Type II compliance is maintained across all plans.",

    # --- Troubleshooting Guides ---
    "If you receive a 401 error, check that your API key is valid and not expired.",
    "Connection timeouts usually indicate a network issue. Check your firewall settings.",
    "For slow response times, try using our regional endpoints: us.api.techcorp.io or eu.api.techcorp.io.",
    "If webhooks are not firing, verify the endpoint URL is publicly accessible and returns a 200 status.",
    "Dashboard login issues: Clear your browser cache or try an incognito window.",
]


def create_knowledge_base(persist_dir: str = "./support_kb") -> Chroma:
    """
    Create a new vector database from product documentation.

    Splits documentation into chunks, generates embeddings, and stores
    them in a ChromaDB vector database for semantic search.

    Args:
        persist_dir: Directory path to persist the vector database

    Returns:
        A ChromaDB vector store instance
    """
    # Split documents into chunks with overlap for better context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Maximum chunk size in characters
        chunk_overlap=200  # Overlap between chunks to maintain context
    )

    # Join all docs and split into chunks
    texts = text_splitter.split_text("\n\n".join(PRODUCT_DOCS))

    # Create vector store with embeddings
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"Knowledge base created with {len(texts)} chunks")
    return vector_store


def get_knowledge_base(persist_dir: str = "./support_kb") -> Chroma:
    """
    Load or create the knowledge base.

    Attempts to load an existing vector database from disk. If it doesn't
    exist or is empty, creates a new one from PRODUCT_DOCS.

    Args:
        persist_dir: Directory path where the vector database is stored

    Returns:
        A ChromaDB vector store instance ready for similarity search
    """
    try:
        # Try to load existing vector store
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        # Verify it contains documents
        if vectorstore._collection.count() > 0:
            return vectorstore
    except Exception:
        # Directory doesn't exist or is corrupted
        pass

    # Create new knowledge base if loading failed
    return create_knowledge_base(persist_dir)
