"""
Philosophical Assistants Module

This module provides functionality for managing philosophical assistants
across different worldviews using various backends including custom implementations
and hybrid DeepSeek + Pinecone approach.
"""

# Existing custom assistant functionality
from .common_instructions import compose_instructions, extract_worldview_instructions
from .pinecone_integration import AssistantManager, PineconeClient, EmbeddingClient
from .template_processor import TemplateProcessor

# Hybrid DeepSeek + Pinecone Assistant functionality (cost-effective)
from .deepseek_assistant_manager import DeepSeekAssistantManager as PineconeAssistantManager

__all__ = [
    # Common instructions
    "compose_instructions",
    "extract_worldview_instructions",
    
    # Custom assistant system
    "AssistantManager", 
    "PineconeClient", 
    "EmbeddingClient",
    "TemplateProcessor",
    
    # Hybrid Assistant Manager (DeepSeek + Pinecone)
    "PineconeAssistantManager"
] 