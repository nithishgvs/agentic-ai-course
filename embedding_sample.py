
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Generate embeddings
vec1 = embeddings.embed_query("The cat sat on the mat")
vec2 = embeddings.embed_query("A feline rested on the rug")
vec3 = embeddings.embed_query("Stock markets closed higher today")

print(f"Vector dimensions: {len(vec1)}")
print(f"Vector dimensions: {vec1}")
