#!/usr/bin/env python3
"""
Template Processor for Philosophical Assistants

This script handles the processing of templates for philosophical assistants,
adapting them based on the specific worldview of each assistant.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List
from jinja2 import Environment, FileSystemLoader, Template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Template directory
TEMPLATE_DIR = os.path.join("assistants", "templates")

class TemplateProcessor:
    """Processor for adapting templates to different philosophical worldviews."""
    
    def __init__(self, template_dir: str = TEMPLATE_DIR):
        """Initialize the template processor.
        
        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load available templates from the template directory."""
        try:
            template_files = [f for f in os.listdir(self.template_dir) 
                             if f.endswith('.mdt')]
            
            if not template_files:
                logger.warning(f"No template files found in {self.template_dir}")
            else:
                logger.info(f"Found {len(template_files)} template files: {', '.join(template_files)}")
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            raise
    
    def get_template(self, template_name: str) -> Optional[Template]:
        """Get a template by name.
        
        Args:
            template_name: Name of the template (without .mdt extension)
            
        Returns:
            Jinja2 Template object or None if not found
        """
        try:
            template_path = f"{template_name}.mdt"
            return self.env.get_template(template_path)
        except Exception as e:
            logger.error(f"Error getting template {template_name}: {e}")
            return None
    
    def render_template(
        self, 
        template_name: str, 
        worldview: str, 
        variables: Dict[str, Any]
    ) -> str:
        """Render a template with variables for a specific worldview.
        
        Args:
            template_name: Name of the template (without .mdt extension)
            worldview: Philosophical worldview
            variables: Template variables
            
        Returns:
            Rendered template as string
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Add worldview to variables
        variables["weltanschauung"] = worldview
        
        # Render the template
        try:
            return template.render(**variables)
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            raise
    
    def process_gedankenfehler_template(
        self,
        worldview: str,
        gedanke_in_weltanschauung: str,
        aspekte: Optional[str] = None
    ) -> str:
        """Process the gedankenfehler-formulieren template.
        
        Args:
            worldview: Philosophical worldview
            gedanke_in_weltanschauung: Philosophical thought to correct
            aspekte: Additional aspects to consider
            
        Returns:
            Rendered template as string
        """
        variables = {
            "gedanke_in_weltanschauung": gedanke_in_weltanschauung
        }
        
        # Add aspects if provided
        if aspekte:
            # Process the aspects template
            aspects_template = self.get_template("gedankenfehler-formulieren-aspekte")
            if aspects_template:
                variables["aspekte"] = aspekte
            else:
                variables["aspekte"] = ""
        else:
            variables["aspekte"] = ""
        
        return self.render_template("gedankenfehler-formulieren", worldview, variables)
    
    def process_glossar_template(
        self,
        worldview: str,
        korrektur: str
    ) -> str:
        """Process the gedankenfehler-glossar template.
        
        Args:
            worldview: Philosophical worldview
            korrektur: Corrected philosophical thought
            
        Returns:
            Rendered template as string
        """
        variables = {
            "korrektur": korrektur
        }
        
        return self.render_template("gedankenfehler-glossar", worldview, variables)
    
    def process_wiederholen_template(
        self,
        worldview: str,
        gedanke: str,
        stichwort: str
    ) -> str:
        """Process the gedankenfehler-wiederholen template.
        
        Args:
            worldview: Philosophical worldview
            gedanke: Philosophical thought to repeat
            stichwort: Keywords to include
            
        Returns:
            Rendered template as string
        """
        variables = {
            "gedanke": gedanke,
            "stichwort": stichwort
        }
        
        return self.render_template("gedankenfehler-wiederholen", worldview, variables)
    
    def parse_template_response(self, response: str) -> Dict[str, Any]:
        """Parse a JSON response from a template.
        
        Args:
            response: Template response as string
            
        Returns:
            Parsed JSON as dictionary
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'({[\s\S]*})', response)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.error(f"Response: {response}")
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Error parsing template response: {e}")
            raise

def main():
    """Main function for testing."""
    try:
        # Initialize template processor
        processor = TemplateProcessor()
        
        # Test gedankenfehler-formulieren template
        print("\nTesting gedankenfehler-formulieren template:")
        gedankenfehler_prompt = processor.process_gedankenfehler_template(
            worldview="Idealismus",
            gedanke_in_weltanschauung="Der Mensch ist nur ein komplexer biologischer Mechanismus ohne geistige Dimension.",
            aspekte="Betrachte das Verhältnis von Geist und Körper."
        )
        print(gedankenfehler_prompt)
        
        # Test gedankenfehler-glossar template
        print("\nTesting gedankenfehler-glossar template:")
        glossar_prompt = processor.process_glossar_template(
            worldview="Materialismus",
            korrektur="Das Bewusstsein ist ein Produkt neuronaler Prozesse im Gehirn."
        )
        print(glossar_prompt)
        
        # Test gedankenfehler-wiederholen template
        print("\nTesting gedankenfehler-wiederholen template:")
        wiederholen_prompt = processor.process_wiederholen_template(
            worldview="Spiritualismus",
            gedanke="Der Mensch ist ein geistiges Wesen mit einem physischen Körper.",
            stichwort="Seele, Entwicklung, Inkarnation"
        )
        print(wiederholen_prompt)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 