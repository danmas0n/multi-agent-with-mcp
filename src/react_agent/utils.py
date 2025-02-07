"""Utility functions for the ReAct agent."""

import logging
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_chat_model(model_name: str, openrouter_base_url: str | None = None) -> Any:
    """Load a chat model based on the provider prefix.

    Args:
        model_name: Name of the model in format "provider/model-name"
        openrouter_base_url: Base URL for OpenRouter API if using OpenRouter

    Returns:
        A chat model instance

    Raises:
        ValueError: If the provider is not supported
    """
    logger.info(f"Loading chat model with name: {model_name}")
    provider, *model_parts = model_name.split("/")
    model = "/".join(model_parts)
    
    logger.info(f"Parsed provider: {provider}")
    logger.info(f"Parsed model: {model}")
    
    if provider == "anthropic":
        logger.info("Using Anthropic client")
        return ChatAnthropic(model=model)
    elif provider == "openai":
        logger.info("Using OpenAI client")
        return ChatOpenAI(model=model)
    elif provider == "openrouter":
        logger.info("Using OpenRouter via OpenAI client")
        logger.info(f"OpenRouter base URL: {openrouter_base_url}")
        
        if not openrouter_base_url:
            raise ValueError("openrouter_base_url is required for OpenRouter models")
            
        # For OpenRouter, we pass just the model part without the openrouter prefix
        api_key = os.getenv("OPENROUTER_API_KEY")
        logger.info(f"OpenRouter API key present: {bool(api_key)}")
        logger.info(f"Using model identifier: {model}")
        
        return ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=openrouter_base_url,
            model=model,  # Use just the model part (e.g., "anthropic/claude-3-5-sonnet-20241022")
        )
    elif provider == "google":
        logger.info("Using Google Gemini client")
        api_key = os.getenv("GOOGLE_API_KEY")
        logger.info(f"Google API key present: {bool(api_key)}")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Google models")
            
        return ChatGoogleGenerativeAI(
            model=model,  # e.g., "gemini-1.5-pro"
            google_api_key=api_key,
            temperature=0,  # Match default behavior of other providers
            convert_system_message_to_human=True  # Gemini doesn't support system messages natively
        )
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            "Must be one of: anthropic, openai, openrouter, google"
        )
