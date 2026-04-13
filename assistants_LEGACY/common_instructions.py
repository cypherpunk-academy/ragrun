"""
Common Instructions Module

This module contains the common instructions shared across all philosophical assistants,
regardless of their specific worldview. It helps eliminate duplication in assistant configurations
and provides a central place to manage shared instruction components.
"""

import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Common instructions shared by all philosophical assistants
COMMON_INSTRUCTIONS = """
Sprich immer deutsch.

Umgang mit Quellen (Vector-Store): Bei jeder Anfrage ziehst du Erkenntnisse aus dem verfügbaren Wissen, das in deinem Vector-Store vorliegt. Greife stets auf diese Quellen zurück, um deine Antworten zu vertiefen und an die Tradition deiner Weltanschauung anzuknüpfen.

Nutze IMMER die Dateisuche (File Search / Vector Store), um Fragen zu beantworten – selbst wenn du die Antwort zu kennen glaubst. Gehe nicht von Informationen außerhalb der bereitgestellten Dateien aus.

Sprich aus dir selbst heraus, zitiere nicht, verweise nicht auf andere. Bemühre dich, in deinem ureigenen Sound zu sprechen.
""".strip()

def compose_instructions(worldview_instructions: str, worldview: str) -> str:
    """Compose full instructions by combining worldview-specific and common instructions.
    
    Args:
        worldview_instructions: The specific instructions for the worldview
        worldview: The name of the philosophical worldview
        
    Returns:
        Combined instructions string
    """
    # Combine worldview-specific instructions with common instructions
    full_instructions = f"{worldview_instructions}\n\n{COMMON_INSTRUCTIONS}"
    
    # Replace any worldview placeholders
    full_instructions = full_instructions.replace("{{weltanschauung}}", worldview)
    
    return full_instructions

def extract_worldview_instructions(full_instructions: str) -> str:
    """Extract worldview-specific instructions from full instructions.
    
    Args:
        full_instructions: The complete instructions including common parts
        
    Returns:
        Worldview-specific instructions with common parts removed
    """
    # Split by the common instructions and take the first part
    if COMMON_INSTRUCTIONS in full_instructions:
        return full_instructions.split(COMMON_INSTRUCTIONS)[0].strip()
    else:
        # If common instructions are not found as-is, return the original
        # This handles cases where there might be slight variations
        logger.warning("Common instructions not found in the full instructions")
        return full_instructions

def update_assistant_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Update an assistant configuration to use the common instructions pattern.
    
    Args:
        config: Original assistant configuration dictionary
        
    Returns:
        Updated configuration with separated worldview instructions
    """
    if "instructions" in config and "worldview_instructions" not in config:
        # Extract worldview-specific instructions and store them
        worldview = config.get("weltanschauung", "")
        worldview_instructions = extract_worldview_instructions(config["instructions"])
        config["worldview_instructions"] = worldview_instructions
        
        # Recompose full instructions
        config["instructions"] = compose_instructions(worldview_instructions, worldview)
    
    return config 