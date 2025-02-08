from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from react_agent.agents.base_agent import BaseAgent

CODER_PROMPT = """You are the coder agent. Your responsibilities are:
1. Create a new Git branch
2. Implement the proposed solution
3. Commit the changes
4. Use route_to_orchestrator tool when complete

IMPORTANT: When using MCP tools:
- Start your response with a <tool_result> block for each tool call
- Each tool result must be acknowledged separately
- Format your response like this:
  <tool_result>Acknowledging result from tool X</tool_result>
  <tool_result>Acknowledging result from tool Y</tool_result>
  [rest of your response]

CRITICAL:
- Always create a new branch before making changes
- Follow Git best practices for commits
- Test your changes thoroughly
- When implementation is complete, use route_to_orchestrator tool
- If you encounter errors, fix them and try again
- Do your best to complete the task on your own -- the orchestrator and planner don't know how to code

WORKFLOW:
1. Create a new branch with a descriptive name
2. Make the necessary code changes
3. Test the changes
4. Commit with a clear message
5. Use route_to_orchestrator tool to return to orchestrator
"""

class Coder(BaseAgent):
    def __init__(self, llm: BaseChatModel, tools: list):
        super().__init__("coder", CODER_PROMPT, llm, tools)

def get_coder(llm: BaseChatModel, tools: list) -> Coder:
    return Coder(llm, tools)
