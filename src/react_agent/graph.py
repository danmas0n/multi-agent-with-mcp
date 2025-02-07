"""Define a custom multi-agent workflow for implementing Linear stories."""
from datetime import datetime, timezone
from typing import Dict, List, Literal, cast, Union, Any
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

class LinearWorkflow:
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
        self.planner = get_planner(self.llm)
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

    async def execute_tool(self, state: MessagesState) -> Dict[str, List[Dict[str, Any]]]:
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
                                return {"messages": [HumanMessage(content=str(result))]}
                            except Exception as e:
                                logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
                                return {"messages": [HumanMessage(content=f"Error: {str(e)}")]}
        
        logger.warning("No tool call found in message")
        return {"messages": []}

    def route_orchestrator(self, state: MessagesState) -> Literal["orchestrator", "planner", "coder", "MCP1", "done"]:
        """Route next steps for orchestrator agent."""
        last_message = state['messages'][-1]
        logger.info(f"Orchestrator routing - Message type: {type(last_message)}")
        
        # Only process AIMessages
        if not isinstance(last_message, AIMessage):
            logger.info("Not an AIMessage - Staying with orchestrator")
            return "orchestrator"
            
        # Log the full message for debugging
        logger.info(f"Orchestrator routing - Message: {last_message}")
        
        # Check for tool calls in both traditional and content list formats
        has_tools = self.has_tool_calls(last_message)
        logger.info(f"Message has tool calls: {has_tools}")
        if has_tools:
            logger.info("Found tool calls - Routing to MCP1")
            return "MCP1"
        
        # Extract and process content
        content = self.extract_content(last_message)
        logger.info(f"Processed message content: {content}")
        
        # Check for routing signals
        if "CONTEXT COMPLETE" in content:
            logger.info("Found CONTEXT COMPLETE - Routing to planner")
            return "planner"
        elif "CODE THIS MFER" in content:
            logger.info("Found CODE THIS MFER - Routing to coder")
            return "coder"
        elif "WORKFLOW COMPLETE" in content:
            logger.info("Found WORKFLOW COMPLETE - Routing to done")
            return "done"
        
        logger.info("No special conditions met - send to tool node (which will route back)")
        return "MCP1"

    def route_coder(self, state: MessagesState) -> Literal["orchestrator", "coder", "MCP2"]:
        """Route next steps for coder agent."""
        last_message = state['messages'][-1]
        
        # Only process AIMessages
        if not isinstance(last_message, AIMessage):
            return "coder"
            
        # Check for tool calls in both formats
        if self.has_tool_calls(last_message):
            return "MCP2"
        
        # Extract and process content
        content = self.extract_content(last_message)
        
        if "I CODED IT MFER" in content:
            return "orchestrator"
        
        # else, send to tool node (which will send it back)
        return "MCP2"

    def setup_workflow(self):
        """Set up the workflow graph."""
        workflow = StateGraph(MessagesState)

        # Add nodes for each agent
        workflow.add_node("orchestrator", self.orchestrator.run)
        workflow.add_node("planner", self.planner.run)
        workflow.add_node("coder", self.coder.run)
        workflow.add_node("MCP1", self.execute_tool)
        workflow.add_node("MCP2", self.execute_tool)

        # Set orchestrator as the entrypoint
        workflow.add_edge("__start__", "orchestrator")

        # Add conditional edges for routing between agents
        workflow.add_conditional_edges(
            "orchestrator",
            self.route_orchestrator,
            {
                "planner": "planner",
                "coder": "coder",
                "MCP1": "MCP1",
                "done": "__end__"
            }
        )

        # Planner always returns to orchestrator
        workflow.add_edge("planner", "orchestrator")

        # Add conditional edges for coder
        workflow.add_conditional_edges(
            "coder",
            self.route_coder,
            {
                "orchestrator": "orchestrator",
                "MCP2": "MCP2",
            }
        )

        # Tools go back to whoever called them
        workflow.add_edge("MCP1", "orchestrator")
        workflow.add_edge("MCP2", "coder")

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
linear_workflow = LinearWorkflow()
graph = linear_workflow.setup_workflow()
