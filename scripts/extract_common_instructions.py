#!/usr/bin/env python3
"""
Extract Common Instructions Script

This script processes all assistant configuration files to extract common instructions
and update them to use the new format with separated worldview-specific instructions.
"""

import os
import json
import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistants.common_instructions import (
    update_assistant_config, 
    extract_worldview_instructions
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Assistant configuration directory
CONFIG_DIR = os.path.join("assistants", "config")

# Worldviews
WORLDVIEWS = ["Idealismus", "Materialismus", "Realismus", "Spiritualismus"]

def process_assistant_config(config_path: str) -> None:
    """Process an assistant configuration file to extract common instructions.
    
    Args:
        config_path: Path to the configuration file
    """
    try:
        # Read the configuration file
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract worldview-specific instructions
        original_instructions = config.get("instructions", "")
        worldview = config.get("weltanschauung", "")
        
        if not original_instructions or not worldview:
            logger.warning(f"Missing instructions or worldview in {config_path}")
            return
        
        # Update the configuration with worldview instructions
        updated_config = update_assistant_config(config)
        
        # Write the updated configuration back to the file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(updated_config, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Updated {config_path}")
        
        # Print a summary of changes
        worldview_instructions = updated_config.get("worldview_instructions", "")
        if worldview_instructions:
            logger.info(f"Extracted {len(worldview_instructions.split())} words of worldview-specific instructions")
            logger.info(f"Original instructions had {len(original_instructions.split())} words")
            logger.info(f"Removed approximately {len(original_instructions.split()) - len(worldview_instructions.split())} words of common instructions")
        
    except Exception as e:
        logger.error(f"Error processing {config_path}: {e}")

def main():
    """Main function to process all assistant configuration files."""
    logger.info("Starting to process assistant configurations")
    
    # Process each worldview configuration
    for worldview in WORLDVIEWS:
        config_path = os.path.join(CONFIG_DIR, f"{worldview.lower()}.json")
        if os.path.exists(config_path):
            logger.info(f"Processing {worldview} configuration")
            process_assistant_config(config_path)
        else:
            logger.warning(f"Configuration file not found for {worldview}")
    
    logger.info("Finished processing assistant configurations")

if __name__ == "__main__":
    main() 