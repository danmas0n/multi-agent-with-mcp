from typing import Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from react_agent.agents.base_agent import BaseAgent

PLANNER_PROMPT = """You are the planner agent. Your responsibility is to:
1. Analyze the context provided by the orchestrator
2. Create 3 different technical variations to solve the problem
3. When complete, use the route_to_coder tool to send the plans to the coder

CRITICAL:
- Create exactly 3 different technical approaches
- Each approach should be clearly labeled and explained
- Focus on technical implementation details
- Do not implement the solutions - that's the coder's job
- When done, use the route_to_coder tool to send your plans

RESPONSE FORMAT:
Approach 1: [Description of first technical approach]
Technical Details:
- Implementation steps
- Key considerations
- Potential challenges

Approach 2: [Description of second technical approach]
Technical Details:
- Implementation steps
- Key considerations
- Potential challenges

Approach 3: [Description of third technical approach]
Technical Details:
- Implementation steps
- Key considerations
- Potential challenges

[After presenting the plans, use the route_to_coder tool to send them to the coder]
"""

#IMPORTANT: When using MCP tools:
#- Start your response with a <tool_result> block for each tool call
#- Each tool result must be acknowledged separately
#- Format your response like this:
#  <tool_result>Acknowledging result from tool X</tool_result>
#  <tool_result>Acknowledging result from tool Y</tool_result>
#  [rest of your response]


class Planner(BaseAgent):
    def __init__(self, llm: BaseChatModel, tools: list = None):
        super().__init__("planner", PLANNER_PROMPT, llm, tools)

def get_planner(llm: BaseChatModel, tools: list = None) -> Planner:
    return Planner(llm, tools)
