"""Define the configurable parameters for the agent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default=os.getenv("LLM_MODEL", "anthropic/claude-3-5-sonnet-20240620"),
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    openrouter_base_url: str = field(
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        metadata={
            "description": "Base URL for OpenRouter API when using OpenRouter models."
        },
    )

    mcp_gateway_url: str = field(
        default="http://localhost:8808",
        metadata={
            "description": "URL of the MCP gateway server that provides tools."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})

    @classmethod
    def load_from_langgraph_json(cls) -> Configuration:
        """Load configuration from langgraph.json file."""
        # Find langgraph.json in current directory or parent directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir != '/':
            config_path = os.path.join(current_dir, 'langgraph.json')
            if os.path.exists(config_path):
                break
            current_dir = os.path.dirname(current_dir)
        else:
            raise FileNotFoundError("Could not find langgraph.json")

        # Load and parse the config file
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Create configuration instance
        config = cls()

        # Load MCP gateway URL if present
        if 'mcp' in config_data:
            config.mcp_gateway_url = config_data['mcp'].get('gateway_url', config.mcp_gateway_url)

        return config
