from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, conint

load_dotenv()


class StockReview(BaseModel):
    company_name: str = Field(description="Company or ticker being reviewed")
    rating: conint(ge=1, le=5) = Field(description="Stock rating from 1 to 5")
    summary: str = Field(description="Brief stock review and current buying considerations")
    alternatives: list[str] = Field(
        description="Similar companies the user may want to research",
    )


parser = JsonOutputParser(pydantic_object=StockReview)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful stock market research assistant. Analyze the provided "
            "company or ticker and return balanced research, not personalized financial "
            "advice. Mention key risks, avoid guarantees, and suggest comparable "
            "companies the user can research.\n\n{format_instructions}",
        ),
        (
            "user",
            "Please review this company's stock:\n\n```{company_name}\n```",
        ),
    ]
)

model = ChatOpenAI(model="gpt-4", temperature=0)

chain = prompt | model | parser


def review_stock(company_name: str) -> dict:
    """Review a company stock and return structured feedback."""
    return chain.invoke(
        {
            "company_name": company_name,
            "format_instructions": parser.get_format_instructions(),
        }
    )


def display_review(review: dict) -> None:
    """Display the review in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"Company: {review['company_name']}")
    print(f"Rating: {review['rating']}/5")
    print(f"{'=' * 60}")
    print(f"\nSummary:\n{review['summary']}")

    alternatives = review.get("alternatives", [])
    if alternatives:
        print("\nSimilar companies to research:")
        for company in alternatives:
            print(f"- {company}")


def main():
    print("\nEnter a company name or ticker to review. Type 'done' on a new line to submit:")
    lines = []

    while True:
        line = input()
        if line.strip().lower() == "done":
            break
        if line.strip():
            lines.append(line.strip())

    if not lines:
        print("No company provided.")
        return

    company_name = " ".join(lines)
    print("\nReviewing stock...")
    review = review_stock(company_name)
    display_review(review)


if __name__ == "__main__":
    main()
