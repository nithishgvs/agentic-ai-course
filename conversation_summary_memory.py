from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Main chat model used for both summarizing and answering the user.
llm = ChatOpenAI(model="gpt-4o")

# Stores the recent conversation messages.
# This list keeps HumanMessage and AIMessage objects.
conversation_history = []

# Stores a compressed summary of older conversation messages.
# This lets the app remember older context without sending every old message
# to the LLM each time.
summary = ""


def summarize_conversation(messages: list, existing_summary: str) -> str:
    """Use the LLM to summarize conversation history."""
    # Build a prompt that asks the model to update the old summary with
    # the older messages we are about to remove from conversation_history.
    #
    # chr(10) is a newline character, same as "\n".
    # The generator expression turns each message into text like:
    # human: hello
    # ai: hi, how can I help?
    # Convert the list of message objects into one readable string.
    #
    # Java-ish mental model:
    # List<String> lines = new ArrayList<>();
    # for (Message message : messages) {
    #     lines.add(message.getType() + ": " + message.getContent());
    # }
    # String formattedMessages = String.join("\n", lines);
    formatted_messages = chr(10).join(
        f"{message.type}: {message.content}" for message in messages
    )
    summary_prompt = (
        "Progressively summarize the conversation, adding to the existing summary.\n\n"
        f"Current summary: {existing_summary}\n\n"
        "New messages:\n"
        f"{formatted_messages}\n\n"
        "Updated summary:"
    )

    # Ask the LLM to produce the updated summary.
    response = llm.invoke([HumanMessage(content=summary_prompt)])

    # ChatOpenAI returns a message object; .content contains the text.
    return response.content


def chat(user_input: str, window_size: int = 6) -> str:
    # Save the latest user message in conversation history.
    conversation_history.append(HumanMessage(content=user_input))

    # We update the module-level summary variable inside this function,
    # so Python needs the global keyword.
    global summary

    # If history exceeds the window size, summarize older messages.
    # Example:
    # - window_size = 6
    # - history has 10 messages
    # - summarize the oldest 4
    # - keep the newest 6 in conversation_history
    if len(conversation_history) > window_size:
        older_messages = conversation_history[:-window_size]
        summary = summarize_conversation(older_messages, summary)

        # Delete the older messages from the list and keep only recent ones.
        del conversation_history[:-window_size]

    # The system message controls assistant behavior.
    system_content = "You are a helpful assistant."

    # If there is an existing summary, include it in the system message so
    # the LLM still has context from older messages.
    if summary:
        system_content += f"\n\nConversation summary so far: {summary}"

    # Final message list sent to the LLM:
    # 1. System instructions, including summary if available
    # 2. Recent conversation history
    messages = [SystemMessage(content=system_content)] + conversation_history

    # Ask the LLM for the next assistant response.
    response = llm.invoke(messages)

    # Save the assistant response so future calls remember it.
    conversation_history.append(AIMessage(content=response.content))

    # Return only the assistant's text to the caller.
    return response.content
