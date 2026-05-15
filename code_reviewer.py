import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()


# Define the structured output format
class CodeReview(BaseModel):
    language: str = Field(description="The programming language of the code")
    issues: list[str] = Field(description="List of issues found in the code")
    suggestions: list[str] = Field(description="List of improvement suggestions")
    rating: int = Field(description="Code quality rating from 1 to 10")
    summary: str = Field(description="One-paragraph summary of the review")


# Set up the output parser
parser = JsonOutputParser(pydantic_object=CodeReview)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert code reviewer. Analyze the provided code snippet "
        "and provide a detailed review. Be constructive and specific.\n\n"
        "{format_instructions}"
    ),
    (
        "user",
        "Please review this code:\n\n```{language}\n{code}\n```"
    )
])

# Create the model
model = ChatOpenAI(model="gpt-4", temperature=0)

# Compose the chain
chain = prompt | model | parser


def review_code(code: str, language: str = "python") -> dict:
    """Review a code snippet and return structured feedback."""
    result = chain.invoke({
        "code": code,
        "language": language,
        "format_instructions": parser.get_format_instructions()
    })
    return result


def display_review(review: dict) -> None:
    """Display the review in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"CODE REVIEW — {review['language'].upper()}")
    print(f"Rating: {review['rating']}/10")
    print(f"{'=' * 60}")

    print(f"\nSummary:\n{review['summary']}")

    if review["issues"]:
        print(f"\nIssues Found ({len(review['issues'])}):")
        for i, issue in enumerate(review["issues"], 1):
            print(f"  {i}. {issue}")

    if review["suggestions"]:
        print(f"\nSuggestions ({len(review['suggestions'])}):")
        for i, suggestion in enumerate(review["suggestions"], 1):
            print(f"  {i}. {suggestion}")

    print(f"\n{'=' * 60}")


def main():
    # Example code to review
    sample_code = """
def get_user_data(id):
    import requests
    r = requests.get(f"http://api.example.com/users/{id}")
    data = r.json()
    name = data['name']
    email = data['email']
    age = data['age']
    return {"name": name, "email": email, "age": age, "raw": data}
"""

    print("Reviewing code snippet...")
    review = review_code(sample_code, "python")
    display_review(review)

    # Interactive mode
    print("\nPaste your own code to review (type 'done' on a new line to submit):")
    lines = []
    while True:
        line = input()
        if line.strip().lower() == "done":
            break
        lines.append(line)

    if lines:
        user_code = "\n".join(lines)
        print("\nReviewing your code...")
        review = review_code(user_code)
        display_review(review)


if __name__ == "__main__":
    main()
