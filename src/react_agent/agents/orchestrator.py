from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from react_agent.agents.base_agent import BaseAgent

ORCHESTRATOR_PROMPT = """You are the orchestrator agent. 

Your responsibilities are:

1. Initially: Use MCP tools to gather context about the task and codebase
   - For local repositories: Use the git MCP server
   - For remote repositories: Use the github MCP server
2. When context is complete: Use the route_to_planner tool to call the planner agent
3. When plans are ready: Use the route_to_coder tool to send them to the coder agent
4. When all variations are coded: Use the route_to_end tool to terminate the workflow

Explain your thinking briefly as you go.

CRITICAL: Do not make plans, create branches or write code. That's the job of the planner and coder agents.
Please send those tasks to them or they will be sad and you will be fired.

CRITICAL: The planner agent may suggest multiple solutions -- if it does, the coder agent will implement each variation
in separate branches and then tell you about them. This is why it's important that you don't create branches or write code yourself.

"""

#IMPORTANT: When using MCP tools:
#- Start your response with a <tool_result> block for each tool call
#- Each tool result must be acknowledged separately
#- Format your response like this:
#  <tool_result>Acknowledging result from tool X</tool_result>
#  <tool_result>Acknowledging result from tool Y</tool_result>
#  [rest of your response]
#"""

class Orchestrator(BaseAgent):
    def __init__(self, llm: BaseChatModel, tools: list):
        super().__init__("orchestrator", ORCHESTRATOR_PROMPT, llm, tools)

def get_orchestrator(llm: BaseChatModel, tools: list) -> Orchestrator:
    return Orchestrator(llm, tools)
