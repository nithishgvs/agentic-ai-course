from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a persistent vector store
vectorstore = Chroma(
    collection_name="my_knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add documents
documents = [
    "Our return policy allows returns within 30 days of purchase.",
    "Shipping is free for orders over $50.",
    "Premium members get 20% off all products.",
    "Customer support is available Monday through Friday, 9 AM to 5 PM.",
    "We accept Visa, Mastercard, and PayPal.",
]

vectorstore.add_texts(documents)

# Search for relevant documents
results = vectorstore.similarity_search("How do I return an item?", k=2)

for doc in results:
    print(doc.page_content)
# => "Our return policy allows returns within 30 days of purchase."
# => "Customer support is available Monday through Friday, 9 AM to 5 PM."
