import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class KnowledgeBaseScanner:
    """Service for scanning and cataloging knowledge base files."""
    
    def __init__(self):
        # Only process .txt files, exclude .quantify and .csv files
        self.supported_extensions = [".txt"]
    
    def scan_kb(self, base_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan the knowledge base directory (alias for scan_directory).
        
        Args:
            base_path: Path to the knowledge base root directory
            
        Returns:
            Dict mapping categories to file information lists
        """
        return self.scan_directory(base_path)
    
    def scan_directory(self, base_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan the knowledge base directory and catalog all files by category.
        
        Args:
            base_path: Path to the knowledge base root directory
            
        Returns:
            Dict mapping categories to file information lists
        """
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Knowledge base directory not found: {base_path}")
        
        logger.info(f"Scanning knowledge base at: {base_path}")
        
        result = {}
        
        # Get all subdirectories (categories)
        categories = [d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')]
        
        for category in categories:
            category_path = os.path.join(base_path, category)
            result[category] = self._scan_category(category_path, category)
            
        logger.info(f"Found {len(categories)} categories with {sum(len(files) for files in result.values())} files")
        return result
    
    def _scan_category(self, category_path: str, category_name: str) -> List[Dict[str, Any]]:
        """
        Scan a single category directory.
        
        Args:
            category_path: Path to the category directory
            category_name: Name of the category
            
        Returns:
            List of file information dictionaries
        """
        files = []
        
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            
            # Skip directories and unsupported file types
            if os.path.isdir(file_path):
                continue
                
            # Get file extension
            _, ext = os.path.splitext(filename)
            if ext not in self.supported_extensions:
                continue
            
            # Extract metadata from filename
            metadata = self._extract_metadata_from_filename(filename, category_name)
            
            # Add file info
            file_info = {
                "path": file_path,
                "filename": filename,
                "extension": ext,
                "category": category_name,
                "size": os.path.getsize(file_path),
                "last_modified": os.path.getmtime(file_path),
                **metadata
            }
            
            files.append(file_info)
            
        return files
    
    def _extract_metadata_from_filename(self, filename: str, category: str) -> Dict[str, Any]:
        """
        Extract metadata from filename.
        
        Handles patterns like:
        - Author#Title.extension
        - ©Author@Title.extension
        
        Args:
            filename: The filename to process
            category: The category name
            
        Returns:
            Dict with extracted metadata
        """
        metadata = {
            "author": "",
            "title": "",
            "category": category
        }
        
        # Remove extension
        name_part = os.path.splitext(filename)[0]
        
        # Check for Author#Title pattern
        hash_match = re.match(r'([^#]+)#(.+)', name_part)
        if hash_match:
            metadata["author"] = hash_match.group(1).strip().replace('_', ' ')
            metadata["title"] = hash_match.group(2).strip().replace('_', ' ')
            return metadata
        
        # Check for ©Author@Title pattern
        at_match = re.match(r'©([^@]+)@(.+)', name_part)
        if at_match:
            metadata["author"] = at_match.group(1).strip().replace('_', ' ')
            metadata["title"] = at_match.group(2).strip().replace('_', ' ')
            return metadata
        
        # Default case: use filename as title
        if name_part != category:  # Avoid using category name as title
            metadata["title"] = name_part.replace('_', ' ')
            
        return metadata

    def get_related_files(self, files: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group related files by their base name (without extension).
        
        Args:
            files: List of file information dictionaries
            
        Returns:
            Dict mapping base names to sets of related files
        """
        related_files = {}
        
        for file_info in files:
            # Get base name without extension
            base_name = os.path.splitext(file_info["filename"])[0]
            
            if base_name not in related_files:
                related_files[base_name] = {}
                
            # Group by extension
            ext = file_info["extension"]
            related_files[base_name][ext] = file_info
            
        return related_files


# Create a singleton instance
kb_scanner = KnowledgeBaseScanner() 