from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

documents = [
    "Python was created by Guido van Rossum in 1991.",
    "LangChain is a framework for building LLM applications.",
    "Vector databases store embeddings for similarity search.",
    "RAG combines retrieval with generation for accurate answers.",
]

# Create FAISS index from documents
vectorstore = FAISS.from_texts(documents, embeddings)

# Save to disk
vectorstore.save_local("./faiss_index")

# Load from disk later
loaded_store = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Search
results = loaded_store.similarity_search("What is LangChain?", k=2)
for doc in results:
    print(doc.page_content)
