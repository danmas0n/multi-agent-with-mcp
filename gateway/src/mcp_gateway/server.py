"""MCP Gateway Server.

This module implements a gateway server that:
1. Exposes an SSE endpoint for clients to connect
2. Reads MCP server configurations
3. Forwards requests to appropriate MCP servers
4. Aggregates responses back to clients
"""

import asyncio
import json
import os
import logging
import signal
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.types import Tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    command: str
    args: List[str]
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class MCPServer:
    """Represents a running MCP server."""
    name: str
    config: MCPServerConfig
    process: asyncio.subprocess.Process
    tools: List[Dict] = field(default_factory=list)


def get_schema(tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get the input schema from a tool definition, handling both naming conventions."""
    # Try both input_schema and inputSchema
    schema = tool.get("input_schema") or tool.get("inputSchema")
    if schema:
        logger.info(f"Found schema for tool {tool['name']}: {json.dumps(schema, indent=2)}")
    else:
        logger.info(f"No schema found for tool {tool['name']}")
    return schema


class Gateway:
    """MCP Gateway that manages server connections and forwards requests."""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        
    async def _communicate_with_server(self, server: MCPServer, method: str, params: dict = None) -> Any:
        """Send a request to a server and get the response."""
        if not server.process.stdin or not server.process.stdout:
            raise Exception("Server process pipes not available")
            
        try:
            # Prepare request
            request = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
                "id": 1
            }
            request_str = json.dumps(request) + "\n"
            logger.info(f"Sending request to {server.name}: {request_str.strip()}")
            
            # Send request
            server.process.stdin.write(request_str.encode())
            await server.process.stdin.drain()
            
            # Read response
            response_line = await server.process.stdout.readline()
            if not response_line:
                raise Exception("Empty response")
                
            response_str = response_line.decode().strip()
            logger.info(f"Received response from {server.name}: {response_str}")
            
            response = json.loads(response_str)
            if "error" in response:
                raise Exception(response["error"])
                
            return response.get("result")
            
        except Exception as e:
            logger.error(f"Error communicating with {server.name}: {str(e)}")
            raise
        
    async def start_server(self, name: str, config: MCPServerConfig) -> MCPServer:
        """Start an MCP server and initialize its client session."""
        try:
            logger.info(f"Starting MCP server: {name}")
            logger.info(f"Server config: command={config.command}, args={config.args}, env={config.env}")
            
            # Construct command
            cmd = f"{config.command} {' '.join(config.args)}"
            logger.info(f"Running command: {cmd}")
            
            # Get current environment and update with server-specific env vars
            env = os.environ.copy()
            env.update(config.env)
            
            # Start the server process in the background
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Create server object
            server = MCPServer(
                name=name,
                config=config,
                process=process
            )
            
            # Wait a bit for server to initialize
            await asyncio.sleep(2)
            
            # Query available tools
            try:
                logger.info(f"Querying tools from {name}")
                result = await self._communicate_with_server(server, "tools/list")
                server.tools = result.get("tools", [])
                logger.info(f"Server {name} tools response: {json.dumps(result, indent=2)}")
                logger.info(f"Server {name} provides tools: {[t['name'] for t in server.tools]}")
                for tool in server.tools:
                    logger.info(f"Tool details for {tool['name']}:")
                    logger.info(f"  Description: {tool.get('description', 'No description')}")
                    schema = get_schema(tool)
                    if schema:
                        logger.info(f"  Schema: {json.dumps(schema, indent=2)}")
            except Exception as e:
                logger.error(f"Error querying tools from {name}: {str(e)}")
                server.tools = []
            
            self.servers[name] = server
            
            # Start monitoring stderr in background
            asyncio.create_task(self._monitor_stderr(server))
            
            return server
            
        except Exception as e:
            logger.error(f"Error starting server {name}: {str(e)}")
            raise
    
    async def _monitor_stderr(self, server: MCPServer):
        """Monitor server's stderr output."""
        while True:
            if server.process.stderr:
                try:
                    line = await server.process.stderr.readline()
                    if line:
                        logger.info(f"[{server.name}] {line.decode().strip()}")
                    else:
                        break
                except Exception as e:
                    logger.error(f"Error reading stderr from {server.name}: {str(e)}")
                    break
    
    async def start_all_servers(self, config_path: str) -> None:
        """Start all configured MCP servers."""
        try:
            logger.info(f"Loading config from: {config_path}")
            # Read config file
            with open(config_path) as f:
                config = json.load(f)
            logger.info(f"Loaded config: {json.dumps(config, indent=2)}")
            
            if not config.get('mcp', {}).get('servers'):
                raise ValueError("No MCP servers configured in config file")
                
            # Start each configured server in parallel
            tasks = []
            for name, server_config in config['mcp']['servers'].items():
                logger.info(f"Creating start task for server: {name}")
                task = asyncio.create_task(
                    self.start_server(
                        name,
                        MCPServerConfig(**server_config)
                    )
                )
                tasks.append(task)
            
            # Wait for all servers to start
            logger.info("Waiting for all servers to start")
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("All servers started")
            
        except Exception as e:
            logger.error(f"Error starting servers: {str(e)}")
            raise
    
    async def list_all_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools from all servers."""
        tools = []
        for server in self.servers.values():
            for tool in server.tools:
                tool_dict = {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "server": server.name
                }
                schema = get_schema(tool)
                if schema:
                    tool_dict["input_schema"] = schema
                tools.append(tool_dict)
        logger.info(f"All available tools: {json.dumps(tools, indent=2)}")
        return tools
    
    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call a tool on the appropriate server."""
        # Find server that has this tool
        for server in self.servers.values():
            if any(t["name"] == tool_name for t in server.tools):
                try:
                    logger.info(f"Calling tool {tool_name} on server {server.name}")
                    logger.info(f"Tool arguments: {json.dumps(arguments, indent=2)}")
                    
                    result = await self._communicate_with_server(
                        server,
                        "tools/call",
                        {
                            "name": tool_name,
                            "arguments": arguments
                        }
                    )
                    
                    logger.info(f"Tool call result: {json.dumps(result, indent=2)}")
                    return result
                except Exception as e:
                    logger.error(f"Error calling tool {tool_name}: {str(e)}")
                    raise
        raise ValueError(f"Tool {tool_name} not found")
    
    async def shutdown(self) -> None:
        """Shutdown all MCP servers."""
        for server in self.servers.values():
            if server.process:
                try:
                    # Kill entire process group
                    os.killpg(os.getpgid(server.process.pid), signal.SIGTERM)
                    await server.process.wait()
                except Exception as e:
                    logger.error(f"Error shutting down server {server.name}: {str(e)}")
                    try:
                        os.killpg(os.getpgid(server.process.pid), signal.SIGKILL)
                    except:
                        pass
        self.servers.clear()


# Global gateway instance
gateway = Gateway()


@app.on_event("startup")
async def startup():
    """Initialize the gateway on startup."""
    config_path = os.environ.get("MCP_CONFIG", "config.json")
    logger.info("Starting MCP Gateway Server")
    await gateway.start_all_servers(config_path)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down MCP Gateway Server")
    await gateway.shutdown()


@app.post("/message")
async def message_endpoint(request: Request):
    """Handle incoming messages from clients."""
    try:
        msg = await request.json()
        logger.info(f"Received message: {json.dumps(msg, indent=2)}")
        
        if msg.get("method") == "tools/list":
            tools = await gateway.list_all_tools()
            response = {"tools": tools}
            logger.info(f"Returning tools: {json.dumps(response, indent=2)}")
            return JSONResponse(response)
        
        elif msg.get("method") == "tools/call":
            params = msg.get("params", {})
            logger.info(f"Tool call parameters: {json.dumps(params, indent=2)}")
            
            result = await gateway.call_tool(
                params.get("name"),
                params.get("arguments", {})
            )
            logger.info(f"Tool call result: {json.dumps(result, indent=2)}")
            return JSONResponse(result)
        
        return JSONResponse({"error": "Unknown method"}, status_code=400)
    except Exception as e:
        logger.error(f"Error handling message: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("MCP_PORT", "8808"))
    uvicorn.run(app, host="0.0.0.0", port=port)
