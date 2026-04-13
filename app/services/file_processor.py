import os
import logging
import json
import csv
from typing import Dict, List, Any, Tuple, Optional
import ast

logger = logging.getLogger(__name__)

class FileProcessor:
    """Service for processing different file types in the knowledge base."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a file based on its extension.
        
        Args:
            file_info: Dictionary with file information
            
        Returns:
            Processed file data
        """
        extension = file_info["extension"]
        
        try:
            if extension == ".txt":
                content, chunks = self.process_text_file(file_info["path"])
                return {
                    "type": "text",
                    "content": content,
                    "chunks": chunks,
                    **file_info
                }
            elif extension == ".quantify":
                word_stats = self.process_quantify_file(file_info["path"])
                return {
                    "type": "quantify",
                    "word_stats": word_stats,
                    **file_info
                }
            elif extension == ".csv":
                word_stats = self.process_csv_file(file_info["path"])
                return {
                    "type": "csv",
                    "word_stats": word_stats,
                    **file_info
                }
            else:
                logger.warning(f"Unsupported file extension: {extension}")
                return {
                    "type": "unknown",
                    **file_info
                }
        except Exception as e:
            logger.error(f"Error processing file {file_info['path']}: {str(e)}")
            return {
                "type": "error",
                "error": str(e),
                **file_info
            }
    
    def process_text_file(self, file_path: str) -> Tuple[str, List[str]]:
        """
        Process a text file and chunk its content.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Tuple of (full content, list of chunks)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create chunks
            chunks = self.chunk_text(content, self.chunk_size, self.chunk_overlap)
            
            return content, chunks
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            
            chunks = self.chunk_text(content, self.chunk_size, self.chunk_overlap)
            return content, chunks
    
    def process_quantify_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a quantify file and extract word statistics.
        
        Args:
            file_path: Path to the quantify file
            
        Returns:
            List of word statistics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # The quantify files appear to be Python dict literals
            # Using ast.literal_eval for safe parsing
            try:
                # Clean the content if needed
                content = content.strip()
                data = ast.literal_eval(content)
                
                # Extract word statistics
                if "word_stats" in data:
                    return data["word_stats"]
                else:
                    return []
            except:
                logger.warning(f"Failed to parse quantify file: {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error reading quantify file {file_path}: {str(e)}")
            return []
    
    def process_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a CSV file with word frequencies.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of word statistics
        """
        word_stats = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if all(k in row for k in ["word", "type", "count", "percentage"]):
                        word_stats.append({
                            "word": row["word"],
                            "type": row["type"],
                            "count": int(row["count"]) if row["count"].isdigit() else 0,
                            "percentage": row["percentage"].strip("%") if "%" in row["percentage"] else row["percentage"]
                        })
            
            return word_stats
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {str(e)}")
            return []
    
    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # Simple character-based chunking
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            # If we're not at the end, try to break at a paragraph or sentence
            if end < len(text):
                # Try to find paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Try to find sentence break
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end)
                    )
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 2
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position for next chunk
            start = end - chunk_overlap if end < len(text) else len(text)
        
        return chunks
    
    def enrich_document_with_metadata(self, document: Dict[str, Any], 
                                       related_files: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enrich a document with metadata from related files.
        
        Args:
            document: Document data
            related_files: Dictionary of related files by extension
            
        Returns:
            Enriched document
        """
        # If this is a text document, look for related quantify file
        if document["type"] == "text" and ".quantify" in related_files:
            quantify_info = related_files[".quantify"]
            if "word_stats" in quantify_info:
                document["word_stats"] = quantify_info["word_stats"]
        
        # If this is a text document, look for related CSV file
        if document["type"] == "text" and ".csv" in related_files:
            csv_info = related_files[".csv"]
            if "word_stats" in csv_info:
                document["csv_stats"] = csv_info["word_stats"]
        
        return document


# Create a singleton instance
file_processor = FileProcessor() 