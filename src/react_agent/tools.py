"""This module provides tools for the ReAct Agent using MCP servers.

Tools are dynamically loaded from MCP servers through the gateway.
"""

import asyncio
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Type, Annotated

from langchain_core.tools import BaseTool, Tool, StructuredTool, tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, create_model

from react_agent import mcp_client

logger = logging.getLogger(__name__)


def get_schema(tool_def: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get the input schema from a tool definition, handling both naming conventions."""
    # Try both input_schema and inputSchema
    schema = tool_def.get("input_schema") or tool_def.get("inputSchema")
    if schema:
        logger.info(f"Found schema for tool {tool_def['name']}: {json.dumps(schema, indent=2)}")
    else:
        logger.info(f"No schema found for tool {tool_def['name']}")
    return schema


def create_schema_model(tool_def: Dict[str, Any]) -> Optional[Type[BaseModel]]:
    """Create a Pydantic model from the tool's schema."""
    schema = get_schema(tool_def)
    if not schema or not isinstance(schema, dict):
        return None
        
    properties = schema.get("properties", {})
    if not properties:
        return None
        
    # Convert JSON schema types to Python types
    field_definitions = {}
    for name, prop in properties.items():
        logger.info(f"Adding field {name} to schema model for {tool_def['name']}")
        python_type = str if prop.get("type") == "string" else Any
        required = name in schema.get("required", [])
        field_definitions[name] = (python_type, ... if required else None)
    
    # Create the model
    model = create_model(
        f"{tool_def['name']}Args",
        **field_definitions
    )
    logger.info(f"Created schema model for {tool_def['name']}: {model}")
    return model


def _create_tool_wrapper(tool_def: Dict[str, Any]) -> BaseTool:
    """Create a wrapper function for an MCP tool.
    
    Args:
        tool_def: Tool definition from the MCP server
        
    Returns:
        A LangChain Tool
    """
    async def wrapper(*args, **kwargs) -> Any:
        """Wrapper function that calls the MCP tool."""
        # Check if tool has no parameters
        schema = get_schema(tool_def)
        has_params = schema and schema.get("properties")
        
        if not has_params:
            # If tool has no parameters, ignore any passed arguments
            logger.info(f"Tool {tool_def['name']} has no parameters, ignoring arguments")
            result = mcp_client.call_tool(tool_def["name"], {})
        else:
            # Convert args to kwargs if needed
            if args:
                logger.info(f"Converting args to kwargs: {args}")
                if len(args) == 1 and isinstance(args[0], str):
                    # If we get a single string argument, treat it as the first schema property
                    first_prop = next(iter(schema["properties"]))
                    kwargs[first_prop] = args[0]
                    logger.info(f"Converted string arg to {first_prop}: {args[0]}")
                elif len(args) == 1 and isinstance(args[0], dict):
                    # If we get a dict argument, merge it with kwargs, excluding __arg1
                    filtered_args = {k: v for k, v in args[0].items() if k != '__arg1'}
                    kwargs.update(filtered_args)
                    logger.info(f"Merged filtered dict arg with kwargs: {filtered_args}")
            
            logger.info(f"Tool wrapper calling with kwargs: {kwargs}")
            result = mcp_client.call_tool(tool_def["name"], kwargs)
        return result
    
    # Create Pydantic model for schema validation
    args_schema = create_schema_model(tool_def)
    
    # Check if we need a structured tool (multiple parameters) or simple tool
    schema = get_schema(tool_def)
    if schema and len(schema.get("properties", {})) > 1:
        # Use StructuredTool for multiple parameters
        tool = StructuredTool(
            name=tool_def["name"],
            description=tool_def.get("description", ""),
            func=wrapper,
            coroutine=wrapper,
            args_schema=args_schema
        )
    else:
        # Use regular Tool for single or no parameters
        tool = Tool(
            name=tool_def["name"],
            description=tool_def.get("description", ""),
            func=wrapper,
            coroutine=wrapper,
            args_schema=args_schema
        )
    
    logger.info(f"Created tool: {tool}")
    return tool


def _load_tools() -> List[BaseTool]:
    """Load all available tools from the MCP gateway.
    
    Returns:
        List of LangChain tools
    """
    logger.info("Loading tools from gateway")
    tools = []
    tool_names = []
    for tool_def in mcp_client.list_tools():
        logger.info(f"Loading tool: {tool_def['name']}")
        if tool_def['name'] in tool_names:
            continue

        tool_names.append(tool_def['name'])
        tool = _create_tool_wrapper(tool_def)
        tools.append(tool)

    logger.info(tool_names)
    return tools


# Routing tools
@tool
async def route_to_planner(
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Route the workflow to the planner agent."""
    return Command(
        goto="planner",
        update={"messages": [ToolMessage(content="Routing to planner", tool_call_id=tool_call_id)]}
    )

@tool
async def route_to_coder(
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Route the workflow to the coder agent."""
    return Command(
        goto="coder",
        update={"messages": [ToolMessage(content="Routing to coder", tool_call_id=tool_call_id)]}
    )

@tool
async def route_to_end(
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """End the workflow."""
    return Command(
        goto="__end__",
        update={"messages": [ToolMessage(content="Ending workflow", tool_call_id=tool_call_id)]}
    )

# Initial routing tools list
ROUTING_TOOLS: List[BaseTool] = [
    route_to_planner,
    route_to_coder,
    route_to_end
]

# Initial empty tools list - will be populated during startup
TOOLS: List[BaseTool] = []


async def initialize_tools(config) -> List[BaseTool]:
    """Initialize connection to MCP gateway and get available tools.
    
    This should be called during application startup.
    
    Args:
        config: Application configuration
        
    Returns:
        List of available tools
    """
    global TOOLS
    
    logger.info("Initializing tools")
    
    # Configure MCP client with gateway URL from config
    if hasattr(config, "mcp_gateway_url"):
        mcp_client.get_client(config.mcp_gateway_url)
    
    # Load tools from gateway and combine with routing tools
    mcp_tools = _load_tools()
    TOOLS = ROUTING_TOOLS + mcp_tools

    logger.info(f"Initialized {len(TOOLS)} tools")
    return TOOLS
