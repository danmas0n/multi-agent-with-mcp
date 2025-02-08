"""Define a custom multi-agent workflow for implementing coding solutions."""
from datetime import datetime, timezone
from typing import Dict, List, Literal, cast, Union, Any
from langgraph.types import Command
import asyncio
import json
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS, initialize_tools
from react_agent.utils import load_chat_model
from react_agent.agents.orchestrator import get_orchestrator
from react_agent.agents.planner import get_planner
from react_agent.agents.coder import get_coder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoderWorkflow:
    def __init__(self):
        # Initialize MCP tools and shared model
        self.config = Configuration.load_from_langgraph_json()
        asyncio.run(initialize_tools(self.config))

        # Create shared model instance
        self.llm = load_chat_model(
            self.config.model,
            self.config.openrouter_base_url
        )

        # Initialize agents
        self.orchestrator = get_orchestrator(self.llm, TOOLS)
        self.planner = get_planner(self.llm, TOOLS)
        self.coder = get_coder(self.llm, TOOLS)

    def has_tool_calls(self, message: AIMessage) -> bool:
        """Check if a message has any tool calls."""
        # Check traditional tool_calls attribute
        if hasattr(message, 'tool_calls') and message.tool_calls:
            return True
            
        # Check content list for tool_use items
        if isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, dict):
                    if item.get('type') == 'tool_use':
                        logger.info(f"Found tool_use in content: {item}")
                        return True
        
        return False

    def extract_content(self, message: AIMessage) -> str:
        """Extract text content from a message."""
        if isinstance(message.content, list):
            content = ""
            for item in message.content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    content += item.get('text', '')
            return content
        return message.content

    def parse_tool_input(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
        """Parse tool input from tool_use block."""
        # Get the raw input
        input_data = tool_use.get('input', {})
        
        # If input is a string, try to parse it as JSON
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                input_data = {"query": input_data}  # Fallback for string inputs
        
        # If we have partial_json, parse and merge it
        partial_json = tool_use.get('partial_json')
        if partial_json:
            if isinstance(partial_json, str):
                try:
                    partial_json = json.loads(partial_json)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse partial_json: {partial_json}")
                else:
                    input_data.update(partial_json)
            elif isinstance(partial_json, dict):
                input_data.update(partial_json)
        
        logger.info(f"Parsed tool input: {input_data}")
        return input_data

    async def execute_tool(self, state: MessagesState) -> Command:
        """Execute tool with logging."""
        last_message = state['messages'][-1]
        logger.info("Executing tool...")
        
        if isinstance(last_message.content, list):
            for item in last_message.content:
                if isinstance(item, dict) and item.get('type') == 'tool_use':
                    tool_name = item.get('name')
                    logger.info(f"Processing tool call: {tool_name}")
                    logger.info(f"Raw tool use block: {item}")
                    
                    # Parse tool input
                    tool_input = self.parse_tool_input(item)
                    logger.info(f"Parsed tool input: {tool_input}")
                    
                    # Find and execute the tool
                    for tool in TOOLS:
                        if tool.name == tool_name:
                            try:
                                result = await tool.ainvoke(tool_input)
                                logger.info(f"Tool result: {result}")
                                # If the tool returns a Command object (our routing tools), return it directly
                                if isinstance(result, Command):
                                    return result
                                # For MCP tools, create a Command to return to the calling agent
                                calling_agent = state.get('current_agent', 'orchestrator')
                                return Command(
                                    goto=calling_agent,
                                    update={"messages": [HumanMessage(content=str(result))]}
                                )
                            except Exception as e:
                                logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
                                calling_agent = state.get('current_agent', 'orchestrator')
                                return Command(
                                    goto=calling_agent,
                                    update={"messages": [HumanMessage(content=f"Error: {str(e)}")]}
                                )
        
        logger.warning("No tool call found in message")
        return Command(
            goto='orchestrator',
            update={"messages": []}
        )

    def route_orchestrator(self, state: MessagesState) -> Literal["MCP", "__end__"]:
        """Route next steps for orchestrator agent."""
        last_message = state['messages'][-1]
        logger.info(f"Orchestrator routing - Message type: {type(last_message)}")
        
        # Check for tool calls
        has_tools = self.has_tool_calls(last_message)
        logger.info(f"Message has tool calls: {has_tools}")
        if has_tools:
            logger.info("Found tool calls - Routing to MCP")
            return "MCP"
        
        # If no tool calls, end the workflow
        logger.info("No tool calls - Ending workflow")
        return "__end__"

    def setup_workflow(self):
        """Set up the workflow graph."""
        workflow = StateGraph(MessagesState)

        # Add nodes for each agent
        workflow.add_node("orchestrator", self.orchestrator.run)
        workflow.add_node("planner", self.planner.run)
        workflow.add_node("coder", self.coder.run)
        workflow.add_node("MCP", self.execute_tool)

        # Set orchestrator as the entrypoint
        workflow.add_edge("__start__", "orchestrator")

        # Add conditional edges for orchestrator routing
        workflow.add_conditional_edges(
            "orchestrator",
            self.route_orchestrator,
            {
                "MCP": "MCP",
                "__end__": "__end__"
            }
        )

        # Add edges from other agents to MCP
        workflow.add_edge("planner", "MCP")
        workflow.add_edge("coder", "MCP")

        # Add edges from MCP to all agents
        workflow.add_edge("MCP", "orchestrator")
        workflow.add_edge("MCP", "planner")
        workflow.add_edge("MCP", "coder")

        return workflow.compile()

    async def execute(self, task: str):
        """Execute the workflow."""
        logger.info("Initiating workflow...")
        workflow = self.setup_workflow()

        logger.info(f"Initial task: {task}")

        # Create proper initial state with HumanMessage
        initial_state = MessagesState(
            messages=[HumanMessage(content=task)]
        )

        config = {"recursion_limit": 50}
        async for output in workflow.astream(initial_state, stream_mode="updates", config=config):
            logger.info(f"Agent message: {str(output)}")

# For LangGraph Studio support
coder_workflow = CoderWorkflow()
graph = coder_workflow.setup_workflow()
