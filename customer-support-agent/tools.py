"""
Support Ticket Tools

This module provides functions for creating and managing support tickets.
Tickets are stored in-memory for demonstration purposes. In production,
these would be stored in a database and integrated with ticketing systems
like Zendesk, Jira, or Salesforce Service Cloud.
"""

import uuid
from datetime import datetime

# In-memory ticket store (use a database in production)
# Maps ticket_id -> ticket dictionary with metadata
tickets = {}


def create_ticket(
    customer_message: str,
    classification: str,
    context: str,
) -> str:
    """
    Create a support ticket and return the ticket ID.

    Generates a unique ticket ID and stores the ticket with metadata
    for tracking and follow-up.

    Args:
        customer_message: The customer's original message/question
        classification: Category of the issue (e.g., "technical", "general")
        context: Retrieved knowledge base context for reference

    Returns:
        A unique ticket ID string (format: TKT-XXXXXXXX)
    """
    # Generate unique 8-character hex ID
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    # Store ticket with metadata
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
    """
    Escalate the conversation to a human agent.

    Creates a high-priority escalation ticket for cases that require
    human judgment, such as sensitive issues, complaints, or urgent requests.

    Args:
        customer_message: The customer's message requiring escalation
        reason: Why this case is being escalated (e.g., "Classified as: sensitive")

    Returns:
        A unique ticket ID for the escalation

    Note:
        In production, this function would also:
        - Send notifications via Slack, PagerDuty, or email
        - Update a ticketing system (Zendesk, Jira, etc.)
        - Log the escalation for analytics
    """
    # Create escalation ticket
    ticket_id = create_ticket(
        customer_message=customer_message,
        classification="escalated",
        context=f"Escalation reason: {reason}",
    )

    # TODO: In production, add notification logic here
    # Example: send_slack_notification(ticket_id, customer_message)
    # Example: create_pagerduty_incident(ticket_id)

    return ticket_id
