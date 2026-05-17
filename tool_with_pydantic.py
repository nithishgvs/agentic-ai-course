from langchain_core.tools import tool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    """Input for the calculator tool."""
    expression: str = Field(
        description="A mathematical expression to evaluate (e.g., '2 + 2', '100 * 0.15')"
    )
    precision: int = Field(
        default=2,
        description="Number of decimal places in the result",
        ge=0,
        le=10
    )


@tool(args_schema=CalculatorInput)
def calculator(expression: str, precision: int = 2) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        # Safety check: only allow math characters
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Expression contains invalid characters."

        result = eval(expression)
        return f"{expression} = {result:.{precision}f}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: {str(e)}"


print(calculator.invoke({"expression": "100 * 0.15", "precision": 4}))
