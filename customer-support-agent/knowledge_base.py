from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample product knowledge base
PRODUCT_DOCS = [
    # Pricing
    "The Starter plan is free and includes 1,000 API calls per month with community support.",
    "The Pro plan costs $49/month and includes 50,000 API calls with priority email support.",
    "The Enterprise plan has custom pricing with unlimited API calls and dedicated account manager.",
    "All paid plans come with a 14-day free trial. No credit card required for the trial.",
    "Annual billing gives a 20% discount on Pro and Enterprise plans.",

    # Features
    "Our API supports REST and GraphQL endpoints for all plan levels.",
    "Webhook notifications are available on Pro and Enterprise plans.",
    "Rate limits: Starter 10 req/s, Pro 100 req/s, Enterprise 1000 req/s.",
    "Data export is available in CSV, JSON, and Parquet formats on all plans.",
    "Custom integrations with Salesforce, HubSpot, and Slack are Enterprise-only features.",

    # Technical
    "API authentication uses Bearer tokens. Generate tokens in the dashboard under Settings > API Keys.",
    "The API base URL is https://api.techcorp.io/v2/ for all endpoints.",
    "Rate limit errors return HTTP 429. Implement exponential backoff for retries.",
    "Webhook payloads are signed with HMAC-SHA256. Verify the X-Signature header.",
    "SDK libraries are available for Python, JavaScript, Go, and Ruby.",

    # Policies
    "Refunds are available within 30 days of purchase for annual plans.",
    "Monthly plans can be cancelled at any time with no cancellation fee.",
    "Data retention: We store customer data for 90 days after account deletion.",
    "Our SLA guarantees 99.9% uptime for Pro plans and 99.99% for Enterprise.",
    "GDPR and SOC 2 Type II compliance is maintained across all plans.",

    # Troubleshooting
    "If you receive a 401 error, check that your API key is valid and not expired.",
    "Connection timeouts usually indicate a network issue. Check your firewall settings.",
    "For slow response times, try using our regional endpoints: us.api.techcorp.io or eu.api.techcorp.io.",
    "If webhooks are not firing, verify the endpoint URL is publicly accessible and returns a 200 status.",
    "Dashboard login issues: Clear your browser cache or try an incognito window.",
]


def create_knowledge_base(persist_dir: str = "./support_kb") -> Chroma:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    texts = text_splitter.split_text("\n\n".join(PRODUCT_DOCS))

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"Knowledge base created with {len(texts)} chunks")
    return vector_store


def get_knowledge_base(persist_dir: str = "./support_kb") -> Chroma:
    """Load or create the knowledge base."""
    try:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        # Check if it has documents
        if vectorstore._collection.count() > 0:
            return vectorstore
    except Exception:
        pass

    return create_knowledge_base(persist_dir)
