from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import uuid
import re
from pathlib import Path


class Weltanschauung(str, Enum):
    """The 12 philosophical worldviews according to Rudolf Steiner"""
    IDEALISMUS = "Idealismus"
    MATERIALISMUS = "Materialismus" 
    REALISMUS = "Realismus"
    SPIRITUALISMUS = "Spiritualismus"
    RATIONALISMUS = "Rationalismus"
    SENSUALISMUS = "Sensualismus"
    PHAENOMENALISMUS = "Phänomenalismus"
    PNEUMATISMUS = "Pneumatismus"
    PSYCHISMUS = "Psychismus"
    MATHEMATISMUS = "Mathematismus"
    DYNAMISMUS = "Dynamismus"
    INDIVIDUALISMUS = "Individualismus"


class SourceType(str, Enum):
    """Types of source documents"""
    BOOK = "book"
    ARTICLE = "article"
    LECTURE = "lecture"
    MANUSCRIPT = "manuscript"
    TRANSCRIPT = "transcript"
    OTHER = "other"


class DocumentMetadata(BaseModel):
    """Enhanced metadata schema for philosophical documents"""
    
    # === Core Identifiers ===
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique document identifier")
    chunk_id: Optional[str] = Field(None, description="Unique chunk identifier (document_id + chunk index)")
    
    # === Book Information ===
    title: Optional[str] = Field(None, max_length=500, description="Document/chapter title")
    book_title: Optional[str] = Field(None, max_length=500, description="Full book title (searchable)")
    author: Optional[str] = Field(None, max_length=200, description="Primary author")
    co_authors: Optional[List[str]] = Field(default_factory=list, description="Additional authors")
    translator: Optional[str] = Field(None, max_length=200, description="Translator name")
    editor: Optional[str] = Field(None, max_length=200, description="Editor name")
    
    # === Classification ===
    category: Optional[Weltanschauung] = Field(None, description="Philosophical worldview category")
    worldview: Optional[Weltanschauung] = Field(None, description="Alternative worldview field for flexible filtering")
    tags: Optional[List[str]] = Field(default_factory=list, description="Content tags")
    subject_areas: Optional[List[str]] = Field(default_factory=list, description="Broader topic areas")
    
    # === Publication Details ===
    year: Optional[str] = Field(None, pattern=r"^\d{4}$", description="Publication year (YYYY)")
    publisher: Optional[str] = Field(None, max_length=200, description="Publisher name")
    edition: Optional[str] = Field(None, max_length=50, description="Edition information")
    isbn: Optional[str] = Field(None, max_length=20, description="ISBN number")
    
    # === Content Structure ===
    chapter: Optional[str] = Field(None, max_length=200, description="Chapter name/number")
    section: Optional[str] = Field(None, max_length=200, description="Section name/number")
    page_number: Optional[str] = Field(None, description="Page number reference")
    paragraph_number: Optional[str] = Field(None, description="Paragraph number")
    
    # === Technical Metadata ===
    chunk_index: Optional[int] = Field(None, ge=0, description="Index of this chunk in the document")
    total_chunks: Optional[int] = Field(None, ge=1, description="Total number of chunks in document")
    chunk_size: Optional[int] = Field(None, ge=0, description="Size of this chunk in characters")
    embedding_model: Optional[str] = Field(None, description="Model used for embeddings")
    embedding_version: Optional[str] = Field(None, description="Version of embedding model")
    embedding_dimension: Optional[int] = Field(768, description="Dimension of embedding vector")
    
    # === File System Information ===
    filename: Optional[str] = Field(None, max_length=255, description="Original filename")
    file_path: Optional[str] = Field(None, description="Relative file path")
    file_hash: Optional[str] = Field(None, description="SHA256 hash for integrity checking")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    source_type: Optional[SourceType] = Field(SourceType.BOOK, description="Type of source document")
    
    # === Content Information ===
    language: Optional[str] = Field("de", max_length=10, description="Document language (ISO code)")
    content_length: Optional[int] = Field(None, ge=0, description="Length of document content")
    text: Optional[str] = Field(None, description="Document text content")
    
    # === Timestamps ===
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
    
    # === Quality Indicators ===
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in metadata extraction")
    validation_status: Optional[str] = Field("pending", description="Validation status")
    extraction_method: Optional[str] = Field("manual", description="How metadata was extracted")
    # === Collection weighting ===
    importance: Optional[int] = Field(5, ge=1, le=10, description="Importance weight (1–10)")
    
    @field_validator('updated_at', mode='before')
    @classmethod
    def update_timestamp(cls, v):
        """Always update the timestamp"""
        return datetime.utcnow()
    
    @field_validator('tags', mode='before')
    @classmethod
    def clean_tags(cls, v):
        """Clean and normalize tags"""
        if v:
            return [tag.strip().lower() for tag in v if tag.strip()]
        return []
    
    @field_validator('co_authors', mode='before')
    @classmethod
    def clean_co_authors(cls, v):
        """Clean co-authors list"""
        if v:
            return [author.strip() for author in v if author.strip()]
        return []
    
    @model_validator(mode='before')
    def validate_and_process(cls, values):
        """Validate metadata consistency and set derived fields"""
        if isinstance(values, dict):
            # Generate chunk_id from document_id and chunk_index if not provided
            if values.get('chunk_id') is None and values.get('document_id') and values.get('chunk_index') is not None:
                values['chunk_id'] = f"{values['document_id']}_chunk_{values['chunk_index']}"
            
            # Sync worldview with category if not explicitly set
            if values.get('worldview') is None and values.get('category'):
                values['worldview'] = values['category']
            
            # Ensure chunk_index doesn't exceed total_chunks
            chunk_index = values.get('chunk_index')
            total_chunks = values.get('total_chunks')
            if chunk_index is not None and total_chunks is not None:
                if chunk_index >= total_chunks:
                    raise ValueError(f"chunk_index ({chunk_index}) must be less than total_chunks ({total_chunks})")
            
            # Ensure embedding_dimension is valid
            embedding_dim = values.get('embedding_dimension', 768)
            if embedding_dim not in [384, 512, 768, 1024, 1536]:
                raise ValueError(f"embedding_dimension ({embedding_dim}) must be one of [384, 512, 768, 1024, 1536]")
        
        return values
    
    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB compatible metadata format"""
        # ChromaDB doesn't support nested objects, so flatten the structure
        metadata = {}
        
        for field_name, field_value in self.dict(exclude_none=True).items():
            if field_name in ['created_at', 'updated_at', 'processed_at']:
                # Convert datetime to ISO string
                if field_value:
                    metadata[field_name] = field_value.isoformat() if isinstance(field_value, datetime) else field_value
            elif field_name in ['tags', 'co_authors', 'subject_areas']:
                # Convert lists to comma-separated strings
                if field_value:
                    metadata[field_name] = ",".join(field_value) if isinstance(field_value, list) else field_value
            elif field_name in ['category', 'worldview', 'source_type']:
                # Convert enums to string values
                metadata[field_name] = field_value.value if hasattr(field_value, 'value') else field_value
            else:
                # Direct assignment for simple types
                metadata[field_name] = field_value
        
        return metadata
    
    @classmethod
    def from_chromadb_metadata(cls, metadata: Dict[str, Any]) -> 'DocumentMetadata':
        """Create DocumentMetadata from ChromaDB metadata"""
        processed_metadata = {}
        
        for key, value in metadata.items():
            if key in ['created_at', 'updated_at', 'processed_at']:
                # Convert ISO string back to datetime
                if value:
                    processed_metadata[key] = datetime.fromisoformat(value) if isinstance(value, str) else value
            elif key in ['tags', 'co_authors', 'subject_areas']:
                # Convert comma-separated strings back to lists
                if value:
                    processed_metadata[key] = value.split(",") if isinstance(value, str) else value
            elif key in ['category', 'worldview']:
                # Convert string back to enum
                if value:
                    try:
                        processed_metadata[key] = Weltanschauung(value)
                    except ValueError:
                        processed_metadata[key] = value  # Keep as string if not valid enum
            elif key == 'source_type':
                # Convert string back to enum
                if value:
                    try:
                        processed_metadata[key] = SourceType(value)
                    except ValueError:
                        processed_metadata[key] = SourceType.OTHER
            else:
                processed_metadata[key] = value
        
        return cls(**processed_metadata)
    
    def get_searchable_fields(self) -> Dict[str, Any]:
        """Get fields that should be searchable"""
        return {
            "author": self.author,
            "book_title": self.book_title,
            "title": self.title,
            "category": self.category.value if self.category else None,
            "worldview": self.worldview.value if self.worldview else None,
            "year": self.year,
            "tags": self.tags,
            "language": self.language,
            "source_type": self.source_type.value if self.source_type else None
        }
    
    def get_filter_dict(self) -> Dict[str, Any]:
        """Get metadata as filter dictionary for ChromaDB queries"""
        filter_dict = {}
        
        # Add non-None searchable fields
        searchable = self.get_searchable_fields()
        for key, value in searchable.items():
            if value is not None:
                if isinstance(value, list) and value:
                    # For list fields, we'll need special handling in queries
                    filter_dict[key] = value
                elif value:
                    filter_dict[key] = value
        
        return filter_dict


class MetadataExtractorResult(BaseModel):
    """Result of metadata extraction process"""
    metadata: DocumentMetadata
    extraction_method: str = "automatic"
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.8)
    extraction_details: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class MetadataValidationResult(BaseModel):
    """Result of metadata validation"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    completeness_score: float = Field(ge=0.0, le=1.0, description="How complete the metadata is")


# === Metadata Schema Constants ===

REQUIRED_FIELDS = {"document_id", "category", "text"}
RECOMMENDED_FIELDS = {"author", "title", "book_title", "year", "filename"}
SEARCHABLE_FIELDS = {"author", "book_title", "title", "category", "worldview", "year", "tags", "language"}
AUTOMATIC_EXTRACTION_FIELDS = {"author", "title", "year", "page_number", "worldview", "filename"}

# Validation patterns
PATTERNS = {
    "year": re.compile(r"^\d{4}$"),
    "isbn": re.compile(r"^(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]$"),
    "page_number": re.compile(r"^\d+$"),
    "filename_author_title": re.compile(r"^([^@#]+)[@#](.+)$"),
    "year_in_content": re.compile(r"\[(.*?(\d{4}).*?)\]"),
    "page_in_content": re.compile(r"\[.*?S\.\s*(\d+).*?\]"),
    "page_with_ga": re.compile(r"\[.*?(?:GA\d*[,\s]*)?S\.\s*(\d+).*?\]")
}

# Default values
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_LANGUAGE = "de"
DEFAULT_SOURCE_TYPE = SourceType.BOOK 