import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Check that the API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file")
    exit(1)

print("API key loaded successfully.")

# Test the connection
llm = ChatOpenAI(model="gpt-4", temperature=0)
response = llm.invoke("Say 'Hello, Agent!' and nothing else.")
print(f"LLM response: {response.content}")
print("\nSetup complete! You are ready to build agents.")