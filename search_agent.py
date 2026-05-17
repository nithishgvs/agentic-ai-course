import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tavily import TavilyClient

# Load environment variables from the .env file.
# This is where API keys like OPENAI_API_KEY and TAVILY_API_KEY usually live.
load_dotenv()


# The @tool decorator turns this normal Python function into a LangChain tool.
# That means the agent can decide to call web_search when it needs current web data.
@tool
def web_search(query: str) -> str:
    """
    Search the Web for current information on the specified topic
    Args:
        query: The search query to lookup
    """
    try:
        # Create a Tavily client using the API key from .env.
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        # Ask Tavily for up to 5 search results for the user's query.
        response = client.search(query, max_results=5)
        results = []

        # Convert each Tavily result into a readable text block.
        for item in response.get("results", []):
            results.append(
                f"Title: {item['title']}\n"
                f"URL: {item['url']}\n"
                # Only keep the first 300 characters so the tool output stays short.
                f"Content: {item['content'][:300]}\n"
            )
        if not results:
            return "No results found for the query."

        # Join all result blocks with a separator line between them.
        return "\n---\n".join(results)

    except Exception as e:
        # Return the error as text so the agent can see what went wrong.
        return f"Search error: {str(e)}"


# This is another LangChain tool. The agent can call it when it needs to
# condense a long piece of text into a shorter answer.
@tool
def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Summarize a long text into a shorter version.

    Args:
        text: The text to summarize.
        max_sentences: Maximum number of sentences in the summary.
    """
    # Create a chat model for summarization.
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # Send the summarization instruction and text to the LLM.
    response = llm.invoke(
        f"Summarize the following text in {max_sentences} sentences:\n\n{text}"
    )

    # ChatOpenAI returns a message object; .content contains the actual text.
    return response.content


# This is the main LLM the agent uses to reason and decide which tools to call.
# temperature=0 makes the output more consistent and less random.
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Give the agent the tools it is allowed to use.
tools = [summarize_text, web_search]

# Create the LangChain agent from the model and tools.
agent = create_agent(tools=tools, model=llm)


# The system message sets the agent's role and behavior.
# It is included with every user question in the research() function.
system_message = SystemMessage(
    content=(
        "You are a research assistant. When asked a question:\n"
        "1. Search the web for relevant, current information\n"
        "2. Analyze and synthesize the results\n"
        "3. Provide a clear, well-structured answer with sources\n"
        "4. If the search results are insufficient, search again with different terms\n"
        "Always cite your sources."
    )
)


def research(question: str) -> str:
    """Run a research query through the agent."""
    print(f"\nResearching: {question}\n")

    # Send the system instructions and the user's question to the agent.
    result = agent.invoke({
        "messages": [system_message, HumanMessage(content=question)]
    })

    # The agent returns a list of messages. The last message is the final answer.
    final_message = result["messages"][-1]
    return final_message.content


def main():
    print("Web Search Agent")
    print("Type your research question (or 'exit' to quit).\n")

    # Keep asking questions until the user types "exit".
    while True:
        question = input("Question: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer = research(question)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)


# This makes main() run only when this file is executed directly.
# It will not run automatically if another Python file imports search_agent.py.
if __name__ == "__main__":
    main()
