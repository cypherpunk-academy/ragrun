"""
Metadata validation utilities for ensuring data consistency and completeness.

This module provides comprehensive validation functions for DocumentMetadata
to ensure data quality, consistency, and completeness across the RAG system.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime
from pathlib import Path

from app.models.metadata import (
    DocumentMetadata, 
    Weltanschauung, 
    SourceType,
    MetadataValidationResult,
    REQUIRED_FIELDS,
    RECOMMENDED_FIELDS,
    SEARCHABLE_FIELDS,
    PATTERNS
)

logger = logging.getLogger(__name__)


class MetadataValidationError(Exception):
    """Custom exception for metadata validation errors"""
    pass


class MetadataValidator:
    """Comprehensive metadata validation with configurable rules"""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the metadata validator
        
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        self.validation_rules = self._init_validation_rules()
        
    def _init_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules configuration"""
        return {
            "required_fields": REQUIRED_FIELDS,
            "recommended_fields": RECOMMENDED_FIELDS,
            "searchable_fields": SEARCHABLE_FIELDS,
            "max_lengths": {
                "title": 500,
                "book_title": 500,
                "author": 200,
                "translator": 200,
                "editor": 200,
                "publisher": 200,
                "edition": 50,
                "isbn": 20,
                "chapter": 200,
                "section": 200,
                "filename": 255,
                "language": 10
            },
            "allowed_languages": {"de", "en", "fr", "es", "it", "nl", "pt", "ru"},
            "allowed_embedding_dimensions": {384, 512, 768, 1024, 1536, 2048},
            "min_confidence_score": 0.0,
            "max_confidence_score": 1.0,
            "min_content_length": 10,
            "max_content_length": 1000000,  # 1MB of text
            "valid_file_extensions": {".txt", ".md", ".pdf", ".docx", ".html", ".xml"},
            "min_year": 1000,
            "max_year": datetime.now().year + 10
        }
    
    def validate_metadata(self, metadata: DocumentMetadata) -> MetadataValidationResult:
        """
        Perform comprehensive metadata validation
        
        Args:
            metadata: DocumentMetadata object to validate
            
        Returns:
            MetadataValidationResult with validation details
        """
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # 1. Required fields validation
            req_errors = self._validate_required_fields(metadata)
            errors.extend(req_errors)
            
            # 2. Field type validation
            type_errors, type_warnings = self._validate_field_types(metadata)
            errors.extend(type_errors)
            warnings.extend(type_warnings)
            
            # 3. Field length validation
            length_errors, length_warnings = self._validate_field_lengths(metadata)
            errors.extend(length_errors)
            warnings.extend(length_warnings)
            
            # 4. Field format validation
            format_errors, format_warnings = self._validate_field_formats(metadata)
            errors.extend(format_errors)
            warnings.extend(format_warnings)
            
            # 5. Cross-field consistency validation
            consistency_errors, consistency_warnings = self._validate_consistency(metadata)
            errors.extend(consistency_errors)
            warnings.extend(consistency_warnings)
            
            # 6. Business logic validation
            business_errors, business_warnings = self._validate_business_logic(metadata)
            errors.extend(business_errors)
            warnings.extend(business_warnings)
            
            # 7. Completeness validation
            completeness_warnings, completeness_suggestions = self._validate_completeness(metadata)
            warnings.extend(completeness_warnings)
            suggestions.extend(completeness_suggestions)
            
            # 8. Calculate completeness score
            completeness_score = self._calculate_completeness_score(metadata)
            
            # 9. Determine if validation passed
            is_valid = len(errors) == 0
            if self.strict_mode:
                is_valid = is_valid and len(warnings) == 0
            
            return MetadataValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                completeness_score=completeness_score
            )
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {str(e)}")
            return MetadataValidationResult(
                is_valid=False,
                errors=[f"Validation exception: {str(e)}"],
                warnings=[],
                suggestions=[],
                completeness_score=0.0
            )
    
    def _validate_required_fields(self, metadata: DocumentMetadata) -> List[str]:
        """Validate required fields are present and non-empty"""
        errors = []
        metadata_dict = metadata.dict()
        
        for field in self.validation_rules["required_fields"]:
            if field not in metadata_dict:
                errors.append(f"Required field '{field}' is missing")
            elif metadata_dict[field] is None:
                errors.append(f"Required field '{field}' is None")
            elif isinstance(metadata_dict[field], str) and not metadata_dict[field].strip():
                errors.append(f"Required field '{field}' is empty")
        
        return errors
    
    def _validate_field_types(self, metadata: DocumentMetadata) -> Tuple[List[str], List[str]]:
        """Validate field types are correct"""
        errors = []
        warnings = []
        metadata_dict = metadata.dict()
        
        # Type validation rules
        type_rules = {
            "document_id": str,
            "chunk_id": (str, type(None)),
            "title": (str, type(None)),
            "book_title": (str, type(None)),
            "author": (str, type(None)),
            "co_authors": (list, type(None)),
            "year": (str, type(None)),
            "chunk_index": (int, type(None)),
            "total_chunks": (int, type(None)),
            "chunk_size": (int, type(None)),
            "embedding_dimension": (int, type(None)),
            "file_size": (int, type(None)),
            "content_length": (int, type(None)),
            "confidence_score": (float, type(None)),
            "tags": (list, type(None)),
            "subject_areas": (list, type(None)),
            "created_at": datetime,
            "updated_at": datetime,
            "processed_at": (datetime, type(None))
        }
        
        for field, expected_type in type_rules.items():
            if field in metadata_dict and metadata_dict[field] is not None:
                value = metadata_dict[field]
                if not isinstance(value, expected_type):
                    errors.append(f"Field '{field}' has incorrect type: expected {expected_type}, got {type(value)}")
        
        # Enum validation
        if metadata.category and not isinstance(metadata.category, Weltanschauung):
            errors.append(f"Field 'category' must be a Weltanschauung enum, got {type(metadata.category)}")
        
        if metadata.source_type and not isinstance(metadata.source_type, SourceType):
            errors.append(f"Field 'source_type' must be a SourceType enum, got {type(metadata.source_type)}")
        
        return errors, warnings
    
    def _validate_field_lengths(self, metadata: DocumentMetadata) -> Tuple[List[str], List[str]]:
        """Validate field lengths are within acceptable limits"""
        errors = []
        warnings = []
        metadata_dict = metadata.dict()
        
        for field, max_length in self.validation_rules["max_lengths"].items():
            if field in metadata_dict and metadata_dict[field] is not None:
                value = str(metadata_dict[field])
                if len(value) > max_length:
                    errors.append(f"Field '{field}' exceeds maximum length {max_length}: {len(value)} characters")
                elif len(value) > max_length * 0.8:  # Warning at 80% of max
                    warnings.append(f"Field '{field}' is approaching maximum length {max_length}: {len(value)} characters")
        
        # Content length validation
        if metadata.text and len(metadata.text) < self.validation_rules["min_content_length"]:
            warnings.append(f"Text content is very short: {len(metadata.text)} characters")
        elif metadata.text and len(metadata.text) > self.validation_rules["max_content_length"]:
            errors.append(f"Text content exceeds maximum length: {len(metadata.text)} characters")
        
        return errors, warnings
    
    def _validate_field_formats(self, metadata: DocumentMetadata) -> Tuple[List[str], List[str]]:
        """Validate field formats using regex patterns"""
        errors = []
        warnings = []
        
        # Year format validation
        if metadata.year:
            if not PATTERNS["year"].match(metadata.year):
                errors.append(f"Invalid year format: '{metadata.year}' (expected YYYY)")
            else:
                year_int = int(metadata.year)
                if year_int < self.validation_rules["min_year"]:
                    warnings.append(f"Year {metadata.year} seems too old")
                elif year_int > self.validation_rules["max_year"]:
                    warnings.append(f"Year {metadata.year} is in the future")
        
        # ISBN format validation
        if metadata.isbn and not PATTERNS["isbn"].match(metadata.isbn):
            errors.append(f"Invalid ISBN format: '{metadata.isbn}'")
        
        # Page number format validation
        if metadata.page_number and not PATTERNS["page_number"].match(metadata.page_number):
            errors.append(f"Invalid page number format: '{metadata.page_number}'")
        
        # Language validation
        if metadata.language and metadata.language not in self.validation_rules["allowed_languages"]:
            warnings.append(f"Language '{metadata.language}' not in allowed languages: {self.validation_rules['allowed_languages']}")
        
        # Embedding dimension validation
        if metadata.embedding_dimension and metadata.embedding_dimension not in self.validation_rules["allowed_embedding_dimensions"]:
            errors.append(f"Invalid embedding dimension: {metadata.embedding_dimension}")
        
        # Filename extension validation
        if metadata.filename:
            file_ext = Path(metadata.filename).suffix.lower()
            if file_ext and file_ext not in self.validation_rules["valid_file_extensions"]:
                warnings.append(f"Unusual file extension: '{file_ext}'")
        
        return errors, warnings
    
    def _validate_consistency(self, metadata: DocumentMetadata) -> Tuple[List[str], List[str]]:
        """Validate cross-field consistency"""
        errors = []
        warnings = []
        
        # Chunk consistency
        if metadata.chunk_index is not None and metadata.total_chunks is not None:
            if metadata.chunk_index >= metadata.total_chunks:
                errors.append(f"chunk_index ({metadata.chunk_index}) must be less than total_chunks ({metadata.total_chunks})")
            elif metadata.chunk_index < 0:
                errors.append(f"chunk_index ({metadata.chunk_index}) cannot be negative")
        
        # Category and worldview consistency
        if metadata.category and metadata.worldview:
            if metadata.category != metadata.worldview:
                warnings.append(f"Category '{metadata.category}' and worldview '{metadata.worldview}' don't match")
        
        # Confidence score range
        if metadata.confidence_score is not None:
            if not (self.validation_rules["min_confidence_score"] <= metadata.confidence_score <= self.validation_rules["max_confidence_score"]):
                errors.append(f"confidence_score ({metadata.confidence_score}) must be between {self.validation_rules['min_confidence_score']} and {self.validation_rules['max_confidence_score']}")
        
        # Content and chunk size consistency
        if metadata.text and metadata.chunk_size:
            actual_size = len(metadata.text)
            if abs(actual_size - metadata.chunk_size) > 100:  # Allow 100 char difference
                warnings.append(f"chunk_size ({metadata.chunk_size}) doesn't match actual text length ({actual_size})")
        
        # Timestamp consistency
        if metadata.created_at and metadata.updated_at:
            if metadata.updated_at < metadata.created_at:
                errors.append("updated_at cannot be before created_at")
        
        if metadata.processed_at and metadata.created_at:
            if metadata.processed_at < metadata.created_at:
                errors.append("processed_at cannot be before created_at")
        
        return errors, warnings
    
    def _validate_business_logic(self, metadata: DocumentMetadata) -> Tuple[List[str], List[str]]:
        """Validate business-specific logic"""
        errors = []
        warnings = []
        
        # Title consistency
        if metadata.title and metadata.book_title:
            if metadata.title == metadata.book_title:
                warnings.append("title and book_title are identical - consider using chapter/section titles")
        
        # Author validation
        if metadata.author and metadata.co_authors:
            if metadata.author in metadata.co_authors:
                warnings.append(f"Primary author '{metadata.author}' is also listed in co_authors")
        
        # Chunk validation for multi-chunk documents
        if metadata.total_chunks and metadata.total_chunks > 1:
            if not metadata.chunk_index and metadata.chunk_index != 0:
                errors.append("chunk_index must be specified for multi-chunk documents")
            if not metadata.chunk_id:
                errors.append("chunk_id must be specified for multi-chunk documents")
        
        # Worldview validation for philosophical documents
        if metadata.category in [w.value for w in Weltanschauung]:
            if not metadata.author:
                warnings.append("Philosophical documents should have an author specified")
        
        # Page number validation
        if metadata.page_number and metadata.page_number == "0":
            warnings.append("Page number 0 is unusual")
        
        return errors, warnings
    
    def _validate_completeness(self, metadata: DocumentMetadata) -> Tuple[List[str], List[str]]:
        """Validate metadata completeness"""
        warnings = []
        suggestions = []
        metadata_dict = metadata.dict()
        
        # Check recommended fields
        missing_recommended = []
        for field in self.validation_rules["recommended_fields"]:
            if field not in metadata_dict or not metadata_dict[field]:
                missing_recommended.append(field)
        
        if missing_recommended:
            warnings.append(f"Missing recommended fields: {', '.join(missing_recommended)}")
            suggestions.extend([f"Consider adding '{field}' for better searchability" for field in missing_recommended])
        
        # Check searchable fields
        missing_searchable = []
        for field in self.validation_rules["searchable_fields"]:
            if field not in metadata_dict or not metadata_dict[field]:
                missing_searchable.append(field)
        
        if len(missing_searchable) > len(self.validation_rules["searchable_fields"]) // 2:
            warnings.append("Many searchable fields are missing - search functionality may be limited")
            suggestions.append("Add more searchable fields like author, title, category, or year")
        
        # Specific completeness checks
        if not metadata.year:
            suggestions.append("Adding publication year improves chronological search")
        
        if not metadata.tags:
            suggestions.append("Adding tags improves content discoverability")
        
        if not metadata.category and not metadata.worldview:
            suggestions.append("Adding category/worldview improves philosophical classification")
        
        return warnings, suggestions
    
    def _calculate_completeness_score(self, metadata: DocumentMetadata) -> float:
        """Calculate metadata completeness score (0.0 to 1.0)"""
        metadata_dict = metadata.dict()
        
        # Weight different field categories
        weights = {
            "required": 0.4,      # 40% for required fields
            "recommended": 0.3,   # 30% for recommended fields  
            "searchable": 0.2,    # 20% for searchable fields
            "optional": 0.1       # 10% for other fields
        }
        
        scores = {}
        
        # Required fields score
        required_filled = sum(1 for field in self.validation_rules["required_fields"] 
                            if field in metadata_dict and metadata_dict[field])
        scores["required"] = required_filled / len(self.validation_rules["required_fields"])
        
        # Recommended fields score
        recommended_filled = sum(1 for field in self.validation_rules["recommended_fields"] 
                               if field in metadata_dict and metadata_dict[field])
        scores["recommended"] = recommended_filled / len(self.validation_rules["recommended_fields"])
        
        # Searchable fields score
        searchable_filled = sum(1 for field in self.validation_rules["searchable_fields"] 
                              if field in metadata_dict and metadata_dict[field])
        scores["searchable"] = searchable_filled / len(self.validation_rules["searchable_fields"])
        
        # Optional fields score (all other fields)
        all_fields = set(metadata_dict.keys())
        optional_fields = all_fields - self.validation_rules["required_fields"] - self.validation_rules["recommended_fields"] - self.validation_rules["searchable_fields"]
        optional_filled = sum(1 for field in optional_fields if metadata_dict[field])
        scores["optional"] = optional_filled / max(len(optional_fields), 1)
        
        # Calculate weighted score
        total_score = sum(scores[category] * weights[category] for category in weights.keys())
        
        return round(total_score, 3)


# Standalone validation functions

def validate_document_metadata(metadata: DocumentMetadata, strict: bool = False) -> MetadataValidationResult:
    """Validate a DocumentMetadata object"""
    validator = MetadataValidator(strict_mode=strict)
    return validator.validate_metadata(metadata)


def validate_required_fields_dict(metadata_dict: Dict[str, Any]) -> List[str]:
    """Quick validation of required fields only"""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in metadata_dict or not metadata_dict[field]:
            errors.append(f"Required field '{field}' is missing or empty")
    return errors


def validate_weltanschauung(category: str) -> bool:
    """Validate if a category is a valid Weltanschauung"""
    try:
        Weltanschauung(category)
        return True
    except ValueError:
        return False


def is_metadata_complete(metadata: DocumentMetadata, threshold: float = 0.7) -> bool:
    """Check if metadata meets completeness threshold"""
    validator = MetadataValidator()
    result = validator.validate_metadata(metadata)
    return result.completeness_score >= threshold


def validate_file_metadata(filename: str, file_path: str, content_length: int) -> List[str]:
    """
    Validate file-related metadata
    
    Args:
        filename: Name of the file
        file_path: Path to the file
        content_length: Length of file content
        
    Returns:
        List of validation errors
    """
    errors = []
    
    if not filename or not filename.strip():
        errors.append("Filename cannot be empty")
    
    if len(filename) > 255:
        errors.append(f"Filename too long: {len(filename)} characters (max 255)")
    
    if not file_path or not file_path.strip():
        errors.append("File path cannot be empty")
    
    if content_length < 0:
        errors.append("Content length cannot be negative")
    
    if content_length > 1000000:  # 1MB
        errors.append(f"Content too large: {content_length} characters (max 1,000,000)")
    
    return errors


def validate_embedding_metadata(embedding_dimension: int, embedding_model: str) -> List[str]:
    """
    Validate embedding-related metadata
    
    Args:
        embedding_dimension: Dimension of embeddings
        embedding_model: Name of embedding model
        
    Returns:
        List of validation errors
    """
    errors = []
    
    allowed_dimensions = {384, 512, 768, 1024, 1536, 2048}
    if embedding_dimension not in allowed_dimensions:
        errors.append(f"Invalid embedding dimension: {embedding_dimension} (allowed: {allowed_dimensions})")
    
    if not embedding_model or not embedding_model.strip():
        errors.append("Embedding model cannot be empty")
    
    return errors


def get_metadata_quality_score(metadata: DocumentMetadata) -> Dict[str, Any]:
    """
    Get comprehensive metadata quality assessment
    
    Args:
        metadata: DocumentMetadata to assess
        
    Returns:
        Dictionary with quality metrics
    """
    validator = MetadataValidator()
    result = validator.validate_metadata(metadata)
    
    return {
        "is_valid": result.is_valid,
        "completeness_score": result.completeness_score,
        "error_count": len(result.errors),
        "warning_count": len(result.warnings),
        "suggestion_count": len(result.suggestions),
        "quality_grade": _get_quality_grade(result.completeness_score, len(result.errors), len(result.warnings)),
        "validation_summary": {
            "errors": result.errors,
            "warnings": result.warnings,
            "suggestions": result.suggestions
        }
    }


def _get_quality_grade(completeness: float, error_count: int, warning_count: int) -> str:
    """Get letter grade for metadata quality"""
    if error_count > 0:
        return "F"  # Fail if there are errors
    elif completeness >= 0.9 and warning_count == 0:
        return "A"  # Excellent
    elif completeness >= 0.8 and warning_count <= 2:
        return "B"  # Good
    elif completeness >= 0.7 and warning_count <= 5:
        return "C"  # Acceptable
    elif completeness >= 0.5:
        return "D"  # Poor but usable
    else:
        return "F"  # Fail


# Validation rule constants
VALIDATION_RULES = {
    "MIN_CONTENT_LENGTH": 10,
    "MAX_CONTENT_LENGTH": 1000000,
    "MIN_CONFIDENCE_SCORE": 0.0,
    "MAX_CONFIDENCE_SCORE": 1.0,
    "ALLOWED_LANGUAGES": {"de", "en", "fr", "es", "it", "nl", "pt", "ru"},
    "ALLOWED_EMBEDDING_DIMENSIONS": {384, 512, 768, 1024, 1536, 2048},
    "MIN_YEAR": 1000,
    "MAX_YEAR": datetime.now().year + 10,
    "MAX_FILENAME_LENGTH": 255,
    "COMPLETENESS_THRESHOLD": 0.7
} 