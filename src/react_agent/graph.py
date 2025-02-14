"""Define a custom multi-agent workflow for implementing coding solutions."""
from datetime import datetime, timezone
from typing import Dict, List, Literal, cast, Union, Any
from langgraph.types import Command
import asyncio
import json
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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
        logger.info(f"Checking for tool calls in message: {message}")
        logger.info(f"Message content type: {type(message.content)}")
        
        # Check tool_calls in additional_kwargs
        if hasattr(message, 'additional_kwargs'):
            tool_calls = message.additional_kwargs.get('tool_calls', [])
            logger.info(f"Found tool_calls in additional_kwargs: {tool_calls}")
            if tool_calls:
                return True

        # Check traditional tool_calls attribute
        if hasattr(message, 'tool_calls'):
            logger.info(f"Found tool_calls attribute: {message.tool_calls}")
            if message.tool_calls:
                return True
            
        # Check content list for tool_use items
        if isinstance(message.content, list):
            logger.info("Message content is a list, checking items...")
            for item in message.content:
                logger.info(f"Checking item type: {type(item)}")
                if isinstance(item, dict):
                    logger.info(f"Dict item: {item}")
                    if item.get('type') == 'tool_use':
                        logger.info(f"Found tool_use in content: {item}")
                        return True
        else:
            logger.info(f"Message content is not a list: {message.content}")
        
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
        
        # Add tool_call_id from the tool use block
        input_data['tool_call_id'] = tool_use.get('id')
        
        logger.info(f"Parsed tool input: {input_data}")
        return input_data

    async def execute_tool(self, state: MessagesState) -> Command:
        """Execute tool with logging."""
        last_message = state['messages'][-1]
        logger.info("Executing tool...")
        logger.info(f"Message in execute_tool: {last_message}")
        logger.info(f"Message type: {type(last_message)}")
        logger.info(f"Current agent: {state.get('current_agent', 'unknown')}")
        if hasattr(last_message, 'additional_kwargs'):
            logger.info(f"Additional kwargs in execute_tool: {last_message.additional_kwargs}")
        
        # First check direct tool_calls attribute
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]  # Take the first tool call
            logger.info(f"Processing tool call from tool_calls attribute: {tool_call}")
            
            # Handle both dict and object formats
            if isinstance(tool_call, dict):
                tool_name = tool_call['name']
                tool_args = tool_call['args']
            else:
                # Assume it's an object with attributes
                tool_name = tool_call.name if hasattr(tool_call, 'name') else tool_call.function.name
                if hasattr(tool_call, 'args'):
                    tool_args = tool_call.args
                else:
                    # Parse arguments from function.arguments if needed
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except (AttributeError, json.JSONDecodeError):
                        logger.error(f"Failed to parse tool arguments from: {tool_call}")
                        tool_args = {}

            logger.info(f"Tool name: {tool_name}, arguments: {tool_args}")
            
            # Find and execute the tool
            for tool in TOOLS:
                if tool.name == tool_name:
                    try:
                        result = await tool.ainvoke(tool_args)
                        logger.info(f"Tool result: {result}")
                        
                        # Check if result has Command attributes
                        if hasattr(result, 'goto') and hasattr(result, 'update'):
                            logger.info(f"Command-like result detected: {result}")
                            return result
                        
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
        
        # Then check for tool_calls in additional_kwargs
        if hasattr(last_message, 'additional_kwargs'):
            tool_calls = last_message.additional_kwargs.get('tool_calls', [])
            if tool_calls:
                tool_call = tool_calls[0]  # Take the first tool call
                logger.info(f"Processing tool call from additional_kwargs: {tool_call}")
                
                tool_name = tool_call['function']['name']
                try:
                    tool_args = json.loads(tool_call['function']['arguments'])
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse tool arguments: {tool_call['function']['arguments']}")
                    tool_args = {}
                
                logger.info(f"Tool name: {tool_name}, arguments: {tool_args}")
                
                # Find and execute the tool
                for tool in TOOLS:
                    if tool.name == tool_name:
                        try:
                            result = await tool.ainvoke(tool_args)
                            logger.info(f"Tool result: {result}")
                            
                            # Check if result has Command attributes
                            if hasattr(result, 'goto') and hasattr(result, 'update'):
                                logger.info(f"Command-like result detected: {result}")
                                return result
                            
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
        
        # Then check for tool_use in content list
        if isinstance(last_message.content, list):
            for item in last_message.content:
                if isinstance(item, dict) and item.get('type') == 'tool_use':
                    tool_name = item.get('name')
                    logger.info(f"Processing tool call from content list: {item}")
                    
                    tool_input = self.parse_tool_input(item)
                    logger.info(f"Parsed tool input: {tool_input}")
                    
                    # Find and execute the tool
                    for tool in TOOLS:
                        if tool.name == tool_name:
                            try:
                                result = await tool.ainvoke(tool_input)
                                logger.info(f"Tool result: {result}")
                                
                                # Check if result has Command attributes
                                if hasattr(result, 'goto') and hasattr(result, 'update'):
                                    logger.info(f"Command-like result detected: {result}")
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
        logger.info(f"Full message content: {last_message.content}")
        if hasattr(last_message, 'additional_kwargs'):
            logger.info(f"Additional kwargs: {last_message.additional_kwargs}")
        
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

        # Add conditional edges from MCP to all agents
        def route_mcp(state: MessagesState) -> Literal["orchestrator", "planner", "coder"]:
            """Route based on the Command's goto value."""
            # Get the last message which should contain our Command result
            last_message = state['messages'][-1]
            logger.info(f"MCP routing - Message: {last_message}")
            logger.info(f"MCP routing - Current agent: {state.get('current_agent', 'unknown')}")
            
            # For routing tools, use their goto value
            if isinstance(last_message, ToolMessage):
                if "Routing to planner" in last_message.content:
                    return "planner"
                elif "Routing to coder" in last_message.content:
                    return "coder"
            
            # For non-routing tools, return to the calling agent
            return state.get('current_agent', 'orchestrator')

        workflow.add_conditional_edges(
            "MCP",
            route_mcp,
            {
                "orchestrator": "orchestrator",
                "planner": "planner",
                "coder": "coder"
            }
        )

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
