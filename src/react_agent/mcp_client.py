"""MCP client for ReAct Agent.

This module handles communication with the MCP gateway server.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class MCPGatewayClient:
    """Client for communicating with the MCP gateway server."""
    
    def __init__(self, gateway_url: str = "http://localhost:8808"):
        """Initialize the client.
        
        Args:
            gateway_url: URL of the MCP gateway server
        """
        self.gateway_url = gateway_url
        self.client = httpx.Client()
        self._tools: Optional[List[Dict[str, Any]]] = None
    
    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Send a request to the gateway server.
        
        Args:
            method: The method to call (e.g., "tools/list", "tools/call")
            params: Optional parameters for the method
            
        Returns:
            The response from the server
            
        Raises:
            Exception: If the request fails
        """
        request = {
            "method": method,
            "params": params or {}
        }
        
        # Log the request being sent
        logger.info(f"Sending request to gateway: {json.dumps(request, indent=2)}")
        
        response = self.client.post(
            f"{self.gateway_url}/message",
            json=request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")
            
        return response.json()
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from the gateway.
        
        Returns:
            List of tool definitions
        """
        if self._tools is None:
            response = self._send_request("tools/list")
            self._tools = response.get("tools", [])
        return self._tools
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool through the gateway.
        
        Args:
            name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            The tool's response
            
        Raises:
            Exception: If the tool call fails
        """
        # Log the incoming arguments
        logger.info(f"call_tool received arguments: {json.dumps(arguments, indent=2)}")
        
        # If arguments is a string, try to parse it as JSON
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
                logger.info(f"Parsed string arguments into: {json.dumps(arguments, indent=2)}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse arguments string: {e}")
                raise
        
        # Ensure arguments is a dictionary
        if not isinstance(arguments, dict):
            logger.error(f"Arguments must be a dictionary, got {type(arguments)}")
            raise TypeError("Arguments must be a dictionary")
        
        params = {
            "name": name,
            "arguments": arguments
        }
        
        # Log the actual parameters being sent
        logger.info(f"Sending parameters to gateway: {json.dumps(params, indent=2)}")
        
        response = self._send_request("tools/call", params)
        
        # Extract text content from response
        if isinstance(response, dict):
            content = response.get("content", [])
            if content and isinstance(content, list):
                first_content = content[0]
                if isinstance(first_content, dict) and first_content.get("type") == "text":
                    return first_content.get("text")
        
        return response


# Global client instance
_client: Optional[MCPGatewayClient] = None


def get_client(gateway_url: Optional[str] = None) -> MCPGatewayClient:
    """Get or create the global client instance.
    
    Args:
        gateway_url: Optional URL for the gateway server
        
    Returns:
        The global client instance
    """
    global _client
    if _client is None:
        _client = MCPGatewayClient(gateway_url or "http://localhost:8808")
    return _client


def list_tools() -> List[Dict[str, Any]]:
    """Get list of available tools.
    
    Returns:
        List of tool definitions
    """
    return get_client().list_tools()


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Call a tool through the gateway.
    
    Args:
        name: Name of the tool to call
        arguments: Arguments to pass to the tool
        
    Returns:
        The tool's response
    """
    return get_client().call_tool(name, arguments)
