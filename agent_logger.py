import logging
import time
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent")

class AgentLogger(BaseCallbackHandler):
    """Custom callback handler for agent observability."""

    def __init__(self):
        self.step_count = 0
        self.start_time = None
        self.total_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        self.step_count += 1
        logger.info(f"LLM call #{self.step_count} started")

    def on_llm_end(self, response, **kwargs):
        elapsed = time.time() - self.start_time
        tokens = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        total = tokens.get("total_tokens", 0)
        self.total_tokens += total

        logger.info(
            f"LLM call #{self.step_count} completed | "
            f"Latency: {elapsed:.2f}s | "
            f"Tokens: {total} | "
            f"Total session tokens: {self.total_tokens}"
        )

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        logger.info(f"Tool called: {tool_name} | Input: {input_str[:100]}")

    def on_tool_end(self, output, **kwargs):
        logger.info(f"Tool result: {str(output)[:200]}")

    def on_llm_error(self, error, **kwargs):
        logger.error(f"LLM error: {error}")

    def on_tool_error(self, error, **kwargs):
        logger.error(f"Tool error: {error}")