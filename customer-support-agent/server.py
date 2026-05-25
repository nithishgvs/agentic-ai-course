from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# chat is the main function that talks to the LangGraph support agent.
# session_store keeps conversation history by session_id.
from agent import chat, session_store

# tickets is an in-memory dictionary created by the ticket tool.
from tools import tickets

# Load environment variables from .env before the app handles requests.
# Example: OPENAI_API_KEY=...
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI calls this when the server starts.
    print("Customer Support Agent API is starting...")  # startup

    # yield means: keep the app running here.
    yield  # app runs here

    # FastAPI calls this when the server shuts down.
    print("Shutting down...")  # shutdown


# Create the FastAPI application.
# Java-ish mental model: this is similar to creating a Spring Boot app object
# and registering controllers/routes on it.
app = FastAPI(
    title="Customer Support Agent",
    description="AI-powered customer support with RAG, tickets, and escalation",
    lifespan=lifespan,
)


# Request body for POST /chat.
# FastAPI uses this Pydantic model to validate incoming JSON.
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        # Reject blank messages.
        if len(v.strip()) == 0:
            raise ValueError("Message cannot be empty")

        # Keep requests reasonably small.
        if len(v) > 4000:
            raise ValueError("Message too long (max 4000 characters)")

        # Return the trimmed message that should be stored in the model.
        return v.strip()


# Response body for POST /chat.
class ChatResponse(BaseModel):
    response: str
    session_id: str


# Response body for GET /tickets/{ticket_id}.
class TicketResponse(BaseModel):
    ticket_id: str
    status: str
    classification: str
    created_at: str


def format_response_for_json(text: str) -> str:
    # JSON displays newline characters as "\n" in raw output.
    # Replace line breaks with spaces so API clients see one clean text string.
    return " ".join(text.split())


# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Send a message to the support agent."""
    try:
        # Pass the user's session_id and message into the LangGraph agent.
        # The chat() function handles session memory and returns final text.
        response = chat(request.session_id, request.message)

        # FastAPI converts this Pydantic model into JSON.
        return ChatResponse(
            response=format_response_for_json(response),
            session_id=request.session_id,
        )
    except Exception as e:
        # Convert unexpected Python errors into HTTP 500 responses.
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
async def get_ticket(ticket_id: str):
    """Look up a support ticket by ID."""
    # tickets is an in-memory dictionary.
    # If the ID is missing, return HTTP 404.
    if ticket_id not in tickets:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # Convert the stored ticket dictionary into a typed response model.
    ticket = tickets[ticket_id]
    return TicketResponse(
        ticket_id=ticket["id"],
        status=ticket["status"],
        classification=ticket["classification"],
        created_at=ticket["created_at"],
    )


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a conversation session."""
    # Remove saved chat history for this session, if it exists.
    if session_id in session_store:
        del session_store[session_id]

    return {"status": "cleared"}


@app.get("/health")
async def health():
    # Simple health check endpoint.
    # Useful for confirming the API server is running.
    return {"status": "healthy"}
