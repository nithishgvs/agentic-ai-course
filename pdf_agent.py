from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# Load a PDF
loader = PyPDFLoader("Google_Interview_Guide_1726941745.pdf")
pages = loader.load()


print(f"Loaded {len(pages)} pages")

# Split into chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# Create vector store from chunks
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./handbook_db"
)

print("Knowledge base ready!")