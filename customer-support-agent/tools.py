import uuid
from datetime import datetime

# In-memory ticket store (use a database in production)
tickets = {}

def create_ticket(
    customer_message: str,
    classification: str,
    context: str,
) -> str:
    """Create a support ticket and return the ticket ID."""
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    tickets[ticket_id] = {
        "id": ticket_id,
        "status": "open",
        "classification": classification,
        "customer_message": customer_message,
        "context": context,
        "created_at": datetime.now().isoformat(),
    }

    return ticket_id


def escalate_to_human(
    customer_message: str,
    reason: str,
) -> str:
    """Escalate the conversation to a human agent."""
    ticket_id = create_ticket(
        customer_message=customer_message,
        classification="escalated",
        context=f"Escalation reason: {reason}",
    )

    # In production, this would notify a human agent via
    # Slack, PagerDuty, email, or a ticketing system
    return ticket_id
