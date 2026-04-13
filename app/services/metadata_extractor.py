import re
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

from app.models.metadata import (
    DocumentMetadata, 
    MetadataExtractorResult, 
    Weltanschauung, 
    SourceType,
    PATTERNS,
    DEFAULT_LANGUAGE,
    DEFAULT_SOURCE_TYPE
)

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Automatic metadata extraction from filenames, paths, and content"""
    
    def __init__(self):
        """Initialize the metadata extractor"""
        self.weltanschauungen = [w.value for w in Weltanschauung]
        
    def extract_metadata(self, 
                        file_path: str, 
                        content: str, 
                        manual_metadata: Optional[Dict[str, Any]] = None) -> MetadataExtractorResult:
        """
        Extract metadata from file path, filename, and content
        
        Args:
            file_path: Full path to the file
            content: Text content of the document
            manual_metadata: Any manually provided metadata to override automatic extraction
            
        Returns:
            MetadataExtractorResult with extracted metadata
        """
        try:
            # Start with empty metadata
            extracted_metadata = {}
            extraction_details = {}
            warnings = []
            errors = []
            confidence_score = 1.0
            
            path_obj = Path(file_path)
            filename = path_obj.name
            
            # 1. Extract from filename (author and title using @ or # delimiters)
            filename_metadata = self.extract_from_filename(filename)
            extracted_metadata.update(filename_metadata)
            extraction_details["filename"] = filename_metadata
            
            # 2. Extract worldview from directory path
            path_metadata = self.extract_from_path(file_path)
            extracted_metadata.update(path_metadata)
            extraction_details["path"] = path_metadata
            
            # 3. Extract from document content (year and page numbers)
            content_metadata = self.extract_from_content(content)
            extracted_metadata.update(content_metadata)
            extraction_details["content"] = content_metadata
            
            # 4. Add file system metadata
            file_metadata = self.extract_file_metadata(file_path, content)
            extracted_metadata.update(file_metadata)
            extraction_details["file_system"] = file_metadata
            
            # 5. Apply manual overrides if provided
            if manual_metadata:
                extracted_metadata.update(manual_metadata)
                extraction_details["manual_overrides"] = manual_metadata
                
            # 6. Validate and clean extracted metadata
            cleaned_metadata, validation_warnings = self.clean_and_validate_metadata(extracted_metadata)
            warnings.extend(validation_warnings)
            
            # 7. Calculate confidence score based on extraction success
            confidence_score = self.calculate_confidence_score(extracted_metadata, extraction_details)
            
            # 8. Create DocumentMetadata object
            try:
                document_metadata = DocumentMetadata(
                    **cleaned_metadata,
                    extraction_method="automatic",
                    confidence_score=confidence_score,
                    text=content
                )
            except Exception as e:
                errors.append(f"Failed to create DocumentMetadata: {str(e)}")
                # Create minimal valid metadata
                document_metadata = DocumentMetadata(
                    text=content,
                    filename=filename,
                    file_path=file_path,
                    extraction_method="automatic_fallback"
                )
                confidence_score = 0.3
            
            return MetadataExtractorResult(
                metadata=document_metadata,
                extraction_method="automatic",
                confidence_score=confidence_score,
                extraction_details=extraction_details,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {str(e)}")
            # Return minimal metadata
            return MetadataExtractorResult(
                metadata=DocumentMetadata(
                    text=content,
                    filename=Path(file_path).name,
                    file_path=file_path,
                    extraction_method="error_fallback"
                ),
                extraction_method="error_fallback",
                confidence_score=0.1,
                extraction_details={},
                warnings=[],
                errors=[f"Extraction failed: {str(e)}"]
            )
    
    def extract_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Extract author and title from filename patterns
        
        Supports patterns:
        - Author@Title.txt
        - Author#Title.txt
        """
        metadata = {"filename": filename}
        
        try:
            # Remove file extension
            name_without_ext = Path(filename).stem
            
            # Try @ delimiter first
            if "@" in name_without_ext:
                parts = name_without_ext.split("@", 1)
                if len(parts) == 2:
                    author = parts[0].strip().replace("_", " ")
                    title = parts[1].strip().replace("_", " ")
                    metadata["author"] = author
                    metadata["title"] = title
                    metadata["book_title"] = title  # Use title as book_title by default
                    logger.debug(f"Extracted from filename (@): author='{author}', title='{title}'")
            
            # Try # delimiter if @ didn't work
            elif "#" in name_without_ext:
                parts = name_without_ext.split("#", 1)
                if len(parts) == 2:
                    author = parts[0].strip().replace("_", " ")
                    title = parts[1].strip().replace("_", " ")
                    metadata["author"] = author
                    metadata["title"] = title
                    metadata["book_title"] = title  # Use title as book_title by default
                    logger.debug(f"Extracted from filename (#): author='{author}', title='{title}'")
            
            else:
                # No delimiter found, use filename as title
                title = name_without_ext.replace("_", " ")
                metadata["title"] = title
                metadata["book_title"] = title
                logger.debug(f"No delimiter in filename, using as title: '{title}'")
                
        except Exception as e:
            logger.warning(f"Failed to extract from filename '{filename}': {str(e)}")
        
        return metadata
    
    def extract_from_path(self, file_path: str) -> Dict[str, str]:
        """
        Extract worldview from directory structure
        
        Looks for Weltanschauung folder names in the path
        """
        metadata = {}
        
        try:
            path_parts = Path(file_path).parts
            
            # Look for Weltanschauung folder names
            for part in path_parts:
                if part in self.weltanschauungen:
                    metadata["worldview"] = part
                    metadata["category"] = part  # Set both for compatibility
                    logger.debug(f"Extracted worldview from path: '{part}'")
                    break
            
            # Also store the relative path
            if len(path_parts) > 1:
                # Create relative path from the first non-absolute part
                relative_parts = [p for p in path_parts if not p.startswith('/') and p != '']
                if relative_parts:
                    metadata["file_path"] = "/".join(relative_parts)
                    
        except Exception as e:
            logger.warning(f"Failed to extract from path '{file_path}': {str(e)}")
        
        return metadata
    
    def extract_from_content(self, content: str) -> Dict[str, str]:
        """
        Extract year and page numbers from document content
        
        Patterns:
        - Year: [1894] or [GA 4, 1894] at document beginning
        - Page: [S. 123] or [GA123, S. 456] anywhere in content
        """
        metadata = {}
        
        if not content or not content.strip():
            return metadata
        
        try:
            # Extract year from beginning of document
            # Look at first 500 characters for year patterns
            content_start = content.strip()[:500]
            
            year_match = PATTERNS["year_in_content"].search(content_start)
            if year_match:
                year = year_match.group(2)  # The year part from the pattern
                metadata["year"] = year
                logger.debug(f"Extracted year from content: '{year}'")
            
            # Extract page numbers from content
            # Find all page number patterns and take the first one
            page_matches = PATTERNS["page_in_content"].findall(content)
            if page_matches:
                page_number = page_matches[0]  # Take first occurrence
                metadata["page_number"] = page_number
                logger.debug(f"Extracted page number from content: '{page_number}'")
            
            # Extract additional content metadata
            content_length = len(content)
            metadata["content_length"] = str(content_length)
            
            # Simple language detection (basic heuristic for German)
            if self._detect_german_content(content):
                metadata["language"] = "de"
            else:
                metadata["language"] = "en"  # Default fallback
                
        except Exception as e:
            logger.warning(f"Failed to extract from content: {str(e)}")
        
        return metadata
    
    def extract_file_metadata(self, file_path: str, content: str) -> Dict[str, str]:
        """Extract file system metadata"""
        metadata = {}
        
        try:
            path_obj = Path(file_path)
            
            # Basic file info
            metadata["filename"] = path_obj.name
            
            # File size (estimate from content if file doesn't exist)
            try:
                if path_obj.exists():
                    metadata["file_size"] = str(path_obj.stat().st_size)
                else:
                    # Estimate from content
                    metadata["file_size"] = str(len(content.encode('utf-8')))
            except:
                metadata["file_size"] = str(len(content.encode('utf-8')))
            
            # File hash for integrity checking
            metadata["file_hash"] = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Source type (default to book)
            metadata["source_type"] = DEFAULT_SOURCE_TYPE.value
            
            # Default language
            metadata["language"] = DEFAULT_LANGUAGE
            
        except Exception as e:
            logger.warning(f"Failed to extract file metadata from '{file_path}': {str(e)}")
        
        return metadata
    
    def clean_and_validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Clean and validate extracted metadata"""
        cleaned = {}
        warnings = []
        
        for key, value in metadata.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                continue  # Skip empty values
                
            try:
                # Clean string values
                if isinstance(value, str):
                    cleaned_value = value.strip()
                    
                    # Special cleaning for specific fields
                    if key in ["author", "title", "book_title"]:
                        # Clean up common filename artifacts
                        cleaned_value = cleaned_value.replace("_", " ")
                        cleaned_value = re.sub(r'\s+', ' ', cleaned_value)  # Multiple spaces to single
                        
                    # Validate year format
                    elif key == "year":
                        if not PATTERNS["year"].match(cleaned_value):
                            warnings.append(f"Invalid year format: '{cleaned_value}', expected YYYY")
                            continue
                    
                    # Validate page number
                    elif key == "page_number":
                        if not PATTERNS["page_number"].match(cleaned_value):
                            warnings.append(f"Invalid page number: '{cleaned_value}'")
                            continue
                    
                    cleaned[key] = cleaned_value
                else:
                    cleaned[key] = value
                    
            except Exception as e:
                warnings.append(f"Failed to clean field '{key}': {str(e)}")
        
        return cleaned, warnings
    
    def calculate_confidence_score(self, metadata: Dict[str, Any], extraction_details: Dict[str, Any]) -> float:
        """Calculate confidence score based on extraction success"""
        score = 0.0
        max_score = 0.0
        
        # Scoring weights for different extraction methods
        weights = {
            "filename": {"author": 0.3, "title": 0.3, "book_title": 0.2},
            "path": {"worldview": 0.4, "category": 0.3},
            "content": {"year": 0.2, "page_number": 0.1},
            "file_system": {"filename": 0.1, "file_hash": 0.1}
        }
        
        for extraction_type, fields in weights.items():
            if extraction_type in extraction_details:
                extracted_data = extraction_details[extraction_type]
                for field, weight in fields.items():
                    max_score += weight
                    if field in extracted_data and extracted_data[field]:
                        score += weight
        
        # Normalize to 0-1 range
        confidence = score / max_score if max_score > 0 else 0.0
        
        # Minimum confidence threshold
        return max(0.1, min(1.0, confidence))
    
    def extract_page_number_from_chunk(self, chunk_text: str) -> Optional[str]:
        """
        Extract page number from a specific chunk of text
        
        This looks for patterns like:
        - [S. 123]
        - [GA4, S. 123]  
        - [–– GA4, S.123 ––]
        
        Args:
            chunk_text: The specific chunk text to search
            
        Returns:
            Page number string if found, None otherwise
        """
        if not chunk_text or not chunk_text.strip():
            return None
            
        try:
            # First try the enhanced GA pattern
            ga_match = PATTERNS["page_with_ga"].search(chunk_text)
            if ga_match:
                page_num = ga_match.group(1)
                logger.debug(f"Extracted page number from chunk (GA pattern): '{page_num}'")
                return page_num
            
            # Fallback to simple page pattern
            page_match = PATTERNS["page_in_content"].search(chunk_text)
            if page_match:
                page_num = page_match.group(1)
                logger.debug(f"Extracted page number from chunk (simple pattern): '{page_num}'")
                return page_num
                
        except Exception as e:
            logger.warning(f"Failed to extract page number from chunk: {str(e)}")
        
        return None
    
    def _detect_german_content(self, content: str) -> bool:
        """Simple heuristic to detect German content"""
        german_indicators = [
            "der", "die", "das", "und", "oder", "aber", "mit", "von", "zu", "im", "am",
            "ist", "war", "wird", "sein", "haben", "werden", "können", "müssen",
            "ß", "ä", "ö", "ü"  # German special characters
        ]
        
        content_lower = content.lower()
        german_count = sum(1 for indicator in german_indicators if indicator in content_lower)
        
        # If we find multiple German indicators, assume it's German
        return german_count >= 3


class MetadataValidator:
    """Validates metadata completeness and consistency"""
    
    def __init__(self):
        self.required_fields = {"document_id", "text"}
        self.recommended_fields = {"author", "title", "book_title", "category", "filename"}
        self.searchable_fields = {"author", "book_title", "title", "category", "worldview", "year"}
    
    def validate_metadata(self, metadata: DocumentMetadata) -> "MetadataValidationResult":
        """Validate metadata completeness and consistency"""
        from app.models.metadata import MetadataValidationResult
        
        errors = []
        warnings = []
        suggestions = []
        
        # Check required fields
        metadata_dict = metadata.dict()
        for field in self.required_fields:
            if field not in metadata_dict or not metadata_dict[field]:
                errors.append(f"Required field '{field}' is missing or empty")
        
        # Check recommended fields
        missing_recommended = []
        for field in self.recommended_fields:
            if field not in metadata_dict or not metadata_dict[field]:
                missing_recommended.append(field)
        
        if missing_recommended:
            warnings.append(f"Recommended fields missing: {', '.join(missing_recommended)}")
            suggestions.extend([f"Consider adding '{field}'" for field in missing_recommended])
        
        # Check searchable fields for search optimization
        missing_searchable = []
        for field in self.searchable_fields:
            if field not in metadata_dict or not metadata_dict[field]:
                missing_searchable.append(field)
        
        if len(missing_searchable) > len(self.searchable_fields) // 2:
            warnings.append("Many searchable fields are missing, search functionality may be limited")
        
        # Validate consistency
        if metadata.category and metadata.worldview and metadata.category != metadata.worldview:
            warnings.append(f"Category '{metadata.category}' and worldview '{metadata.worldview}' don't match")
        
        if metadata.chunk_index is not None and metadata.total_chunks is not None:
            if metadata.chunk_index >= metadata.total_chunks:
                errors.append(f"chunk_index ({metadata.chunk_index}) must be less than total_chunks ({metadata.total_chunks})")
        
        # Calculate completeness score
        total_fields = len(self.required_fields) + len(self.recommended_fields) + len(self.searchable_fields)
        filled_fields = sum(1 for field in (self.required_fields | self.recommended_fields | self.searchable_fields) 
                           if field in metadata_dict and metadata_dict[field])
        completeness_score = filled_fields / total_fields
        
        is_valid = len(errors) == 0
        
        return MetadataValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            completeness_score=completeness_score
        ) 