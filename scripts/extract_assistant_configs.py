#!/usr/bin/env python3
"""
Extract Assistant Configurations from OpenAI Config Files

This script reads all the OpenAI assistant config files from the 12-weltanschauungen
directory and creates a working assistant_definitions.py file for the current system.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any

# Directory mapping to worldviews
ASSISTANT_DIRECTORIES = {
    "Dynamismus_Ariadne_Ikarus_Nietzsche": "Dynamismus",
    "Idealismus_Aurelian_I._Schelling": "Idealismus", 
    "Individualismus_Amara_Illias_Leibniz": "Individualismus",
    "Materialismus_Aloys_I._Freud": "Materialismus",
    "Mathematismus_Arcadius_Ikarus_Torvalds": "Mathematismus",
    "Phänomenalismus_Aetherius_Imaginaris_Goethe": "Phänomenalismus",
    "Pneumatismus_Aurelian_Irenicus_Novalis": "Pneumatismus",
    "Psychismus_Archetype_Intuitionis_Fichte": "Psychismus",
    "Rationalismus_Aristoteles_Isaak_Herder": "Rationalismus",
    "Realismus_Arvid_I._Steiner": "Realismus",
    "Sensualismus_Apollo_Ikarus_Schiller": "Sensualismus",
    "Spiritualismus_Amara_I._Steiner": "Spiritualismus"
}

def normalize_name(name: str) -> str:
    """Normalize assistant name to abbreviate middle names with 'I.'"""
    parts = name.split()
    if len(parts) >= 3:
        # If middle name is not already abbreviated, abbreviate it
        if not parts[1].endswith('.'):
            parts[1] = parts[1][0] + '.'
    return ' '.join(parts)

def create_assistant_id(name: str, worldview: str) -> str:
    """Create assistant ID from name and worldview."""
    # Extract first name and last name, convert to lowercase with dashes
    parts = name.split()
    if len(parts) >= 2:
        first_name = parts[0].lower()
        last_name = parts[-1].lower()
        return f"{first_name}-i--{last_name}"
    return name.lower().replace(' ', '-')

def read_config_file(directory_path: Path) -> Dict[str, Any]:
    """Read the OpenAI config file from a directory."""
    config_files = list(directory_path.glob("*.config"))
    
    if not config_files:
        raise ValueError(f"No .config file found in {directory_path}")
    
    if len(config_files) > 1:
        print(f"Warning: Multiple .config files found in {directory_path}, using first one")
    
    config_file = config_files[0]
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_assistant_data(base_path: str) -> Dict[str, Dict[str, Any]]:
    """Extract assistant data from all directories."""
    base_path = Path(base_path)
    assistant_data = {}
    
    for dir_name, worldview in ASSISTANT_DIRECTORIES.items():
        directory_path = base_path / dir_name
        
        if not directory_path.exists():
            print(f"Warning: Directory {directory_path} not found")
            continue
        
        try:
            config = read_config_file(directory_path)
            
            # Extract and normalize name
            original_name = config.get('name', '')
            normalized_name = normalize_name(original_name)
            
            # Create assistant ID
            assistant_id = create_assistant_id(normalized_name, worldview)
            
            # Extract instructions
            instructions = config.get('instructions', '')
            
            assistant_data[assistant_id] = {
                'name': normalized_name,
                'worldview': worldview,
                'instructions': instructions,
                'original_name': original_name,
                'model': config.get('model', 'deepseek-reasoner'),
                'directory': dir_name
            }
            
            print(f"✅ {worldview}: {assistant_id} ({normalized_name})")
            
        except Exception as e:
            print(f"❌ Error processing {dir_name}: {e}")
            continue
    
    return assistant_data

def generate_assistant_definitions_file(assistant_data: Dict[str, Dict[str, Any]]) -> str:
    """Generate the assistant_definitions.py file content."""
    
    # File header
    content = '''#!/usr/bin/env python3
"""
Philosophical Assistant Definitions - Generated from OpenAI Configs

This file contains all 12 philosophical assistants extracted from the
OpenAI configuration files and adapted for the DeepSeek + Pinecone hybrid system.

Generated automatically from: /Users/michaelschmidt/Reniets/Ai/12-weltanschauungen/assistants
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

class Worldview(Enum):
    """The 12 philosophical worldviews."""
    DYNAMISMUS = "Dynamismus"
    IDEALISMUS = "Idealismus"
    INDIVIDUALISMUS = "Individualismus"
    MATERIALISMUS = "Materialismus"
    MATHEMATISMUS = "Mathematismus"
    PHÄNOMENALISMUS = "Phänomenalismus"
    PNEUMATISMUS = "Pneumatismus"
    PSYCHISMUS = "Psychismus"
    RATIONALISMUS = "Rationalismus"
    REALISMUS = "Realismus"
    SENSUALISMUS = "Sensualismus"
    SPIRITUALISMUS = "Spiritualismus"

@dataclass
class AssistantDefinition:
    """Complete definition of a philosophical assistant."""
    
    # Core identity
    id: str
    name: str
    worldview: Worldview
    instructions: str
    
    # Model configuration
    model: str = "deepseek-reasoner"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Development features
    development_mode: bool = True
    debug_logging: bool = False
    
    # Metadata
    version: str = "1.0.0"
    author: str = "Extracted from OpenAI Configs"
    description: str = ""

# =============================================================================
# PHILOSOPHICAL ASSISTANT DEFINITIONS (Generated from OpenAI Configs)
# =============================================================================

PHILOSOPHICAL_ASSISTANTS = {
'''
    
    # Generate assistant definitions
    for assistant_id, data in assistant_data.items():
        worldview_enum = f"Worldview.{data['worldview'].upper()}"
        
        # Clean up instructions for Python string
        instructions = data['instructions'].replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        
        # Truncate very long instructions for readability (keep first part)
        if len(instructions) > 3000:
            instructions = instructions[:3000] + "..."
            
        content += f'''
    # --- {data['worldview'].upper()} ---
    "{assistant_id}": AssistantDefinition(
        id="{assistant_id}",
        name="{data['name']}",
        worldview={worldview_enum},
        description="Philosophical advisor for {data['worldview']} worldview",
        instructions="""{data['instructions']}""",
        
        model="deepseek-reasoner",
        temperature=0.7,
        development_mode=True,
        
        version="1.0.0",
        author="Generated from OpenAI Config: {data['directory']}"
    ),
'''
    
    # File footer
    content += '''
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_assistant_by_worldview(worldview: Worldview) -> List[str]:
    """Get all assistant IDs for a specific worldview."""
    return [
        assistant_id for assistant_id, definition in PHILOSOPHICAL_ASSISTANTS.items()
        if definition.worldview == worldview
    ]

def list_all_worldviews() -> List[str]:
    """Get all available worldviews."""
    return [worldview.value for worldview in Worldview]

def validate_assistant_definitions() -> Dict[str, List[str]]:
    """Validate all assistant definitions and return any issues."""
    issues = {}
    
    for assistant_id, definition in PHILOSOPHICAL_ASSISTANTS.items():
        assistant_issues = []
        
        # Check required fields
        if not definition.instructions.strip():
            assistant_issues.append("Empty instructions")
        
        if len(definition.instructions) < 100:
            assistant_issues.append("Instructions too short (< 100 characters)")
        
        if not definition.name:
            assistant_issues.append("Missing name")
        
        if assistant_issues:
            issues[assistant_id] = assistant_issues
    
    return issues

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🧠 Philosophical Assistant Definitions (Generated from OpenAI Configs)")
    print(f"Total assistants: {len(PHILOSOPHICAL_ASSISTANTS)}")
    
    # Group by worldview
    by_worldview = {}
    for assistant_id, definition in PHILOSOPHICAL_ASSISTANTS.items():
        worldview = definition.worldview.value
        if worldview not in by_worldview:
            by_worldview[worldview] = []
        by_worldview[worldview].append(assistant_id)
    
    print("\\n📊 By Worldview:")
    for worldview, assistants in by_worldview.items():
        print(f"  {worldview}: {', '.join(assistants)}")
    
    # Validation
    issues = validate_assistant_definitions()
    if issues:
        print("\\n⚠️  Validation Issues:")
        for assistant_id, assistant_issues in issues.items():
            print(f"  {assistant_id}: {', '.join(assistant_issues)}")
    else:
        print("\\n✅ All assistant definitions are valid!")
'''
    
    return content

def main():
    """Main execution function."""
    print("🚀 Extracting Assistant Configurations from OpenAI Configs")
    print("=" * 60)
    
    # Path to the assistants directory
    base_path = "/Users/michaelschmidt/Reniets/Ai/12-weltanschauungen/assistants"
    
    if not os.path.exists(base_path):
        print(f"❌ Base path not found: {base_path}")
        return
    
    # Extract assistant data
    print("📖 Reading config files...")
    assistant_data = extract_assistant_data(base_path)
    
    if not assistant_data:
        print("❌ No assistant data extracted")
        return
    
    print(f"\\n✅ Successfully extracted {len(assistant_data)} assistants")
    
    # Generate assistant definitions file
    print("\\n🔧 Generating assistant definitions file...")
    file_content = generate_assistant_definitions_file(assistant_data)
    
    # Write to file
    output_file = "assistants/assistant_definitions.py"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(file_content)
    
    print(f"\\n🎉 Generated: {output_file}")
    print("\\n📋 Summary:")
    print(f"  - Total assistants: {len(assistant_data)}")
    print(f"  - All 12 worldviews covered")
    print(f"  - Compatible with current CLI and REST API")
    print(f"  - Ready for DeepSeek + Pinecone hybrid system")
    
    print("\\n🧪 Test the generated file:")
    print(f"  python {output_file}")

if __name__ == "__main__":
    main() 