"""
Metadata Extraction Pipeline

This module provides a comprehensive pipeline for extracting, validating, and processing
metadata from documents. It orchestrates the entire metadata lifecycle from raw files
to validated, searchable metadata records.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from datetime import datetime
import hashlib
import json
from dataclasses import dataclass, asdict

from app.models.metadata import (
    DocumentMetadata, 
    MetadataExtractorResult,
    MetadataValidationResult,
    Weltanschauung,
    SourceType
)
from app.services.metadata_extractor import MetadataExtractor
from app.utils.metadata_validators import MetadataValidator, validate_document_metadata

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for metadata extraction pipeline"""
    # Validation settings
    strict_validation: bool = False
    min_completeness_score: float = 0.7
    require_category: bool = True
    require_author: bool = False
    
    # Extraction settings
    auto_detect_language: bool = True
    auto_detect_worldview: bool = True
    extract_from_content: bool = True
    extract_from_filename: bool = True
    extract_from_path: bool = True
    
    # Content processing
    max_content_length: int = 1000000  # 1MB
    min_content_length: int = 10
    chunk_size_threshold: int = 5000
    
    # Quality settings
    min_confidence_score: float = 0.5
    quality_grade_threshold: str = "C"  # Minimum acceptable quality grade
    
    # Error handling
    fail_on_validation_errors: bool = True
    fail_on_extraction_errors: bool = False
    log_validation_warnings: bool = True
    
    # Performance settings
    enable_caching: bool = True
    cache_extraction_results: bool = True
    parallel_processing: bool = False


@dataclass
class ExtractionResult:
    """Result of the complete extraction pipeline"""
    success: bool
    metadata: Optional[DocumentMetadata] = None
    extraction_result: Optional[MetadataExtractorResult] = None
    validation_result: Optional[MetadataValidationResult] = None
    
    # Processing info
    processing_time_ms: float = 0.0
    pipeline_version: str = "1.0"
    
    # Quality metrics
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    quality_grade: str = "F"
    
    # Status and errors
    errors: List[str] = None
    warnings: List[str] = None
    suggestions: List[str] = None
    
    # Processing details
    extraction_method: str = "pipeline"
    extraction_details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []
        if self.extraction_details is None:
            self.extraction_details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        if self.metadata:
            result["metadata"] = self.metadata.dict()
        return result


class MetadataPipeline:
    """Comprehensive metadata extraction and processing pipeline"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the metadata pipeline
        
        Args:
            config: Pipeline configuration, uses defaults if None
        """
        self.config = config or PipelineConfig()
        self.extractor = MetadataExtractor()
        self.validator = MetadataValidator(strict_mode=self.config.strict_validation)
        
        # Pipeline state
        self._cache = {} if self.config.enable_caching else None
        self._stats = {
            "total_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "validation_failures": 0,
            "average_confidence": 0.0,
            "average_completeness": 0.0
        }
        
        logger.info(f"MetadataPipeline initialized with config: {self.config}")
    
    async def process_document(
        self, 
        file_path: str, 
        content: str,
        manual_metadata: Optional[Dict[str, Any]] = None,
        override_config: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Process a single document through the complete extraction pipeline
        
        Args:
            file_path: Path to the document file
            content: Text content of the document
            manual_metadata: Manual metadata overrides
            override_config: Configuration overrides for this document
            
        Returns:
            ExtractionResult with complete processing information
        """
        start_time = datetime.now()
        
        try:
            # Apply configuration overrides
            config = self._apply_config_overrides(override_config)
            
            # Create cache key
            cache_key = self._create_cache_key(file_path, content, manual_metadata) if config.enable_caching else None
            
            # Check cache
            if cache_key and cache_key in self._cache:
                logger.debug(f"Using cached result for {file_path}")
                cached_result = self._cache[cache_key]
                cached_result.extraction_method = "cached"
                return cached_result
            
            # Stage 1: Extract metadata
            extraction_result = await self._extract_metadata(file_path, content, manual_metadata, config)
            
            if not extraction_result.metadata:
                return self._create_error_result(
                    ["Metadata extraction failed"], 
                    extraction_result,
                    start_time
                )
            
            # Stage 2: Validate metadata
            validation_result = await self._validate_metadata(extraction_result.metadata, config)
            
            # Stage 3: Quality assessment
            quality_metrics = await self._assess_quality(extraction_result, validation_result, config)
            
            # Stage 4: Create final result
            result = self._create_final_result(
                extraction_result, 
                validation_result, 
                quality_metrics, 
                start_time,
                config
            )
            
            # Stage 5: Cache result if enabled
            if cache_key and config.cache_extraction_results:
                self._cache[cache_key] = result
            
            # Update statistics
            self._update_stats(result)
            
            logger.info(f"Document processed successfully: {file_path} (confidence: {result.confidence_score:.2f}, completeness: {result.completeness_score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed for {file_path}: {str(e)}")
            return self._create_error_result([f"Pipeline error: {str(e)}"], None, start_time)
    
    async def process_batch(
        self,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]],
        batch_config: Optional[Dict[str, Any]] = None
    ) -> List[ExtractionResult]:
        """
        Process multiple documents in batch
        
        Args:
            documents: List of (file_path, content, manual_metadata) tuples
            batch_config: Configuration for batch processing
            
        Returns:
            List of ExtractionResult objects
        """
        logger.info(f"Processing batch of {len(documents)} documents")
        
        if self.config.parallel_processing:
            # Process documents in parallel
            tasks = [
                self.process_document(file_path, content, manual_metadata, batch_config)
                for file_path, content, manual_metadata in documents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_result = self._create_error_result(
                        [f"Batch processing error: {str(result)}"],
                        None,
                        datetime.now()
                    )
                    final_results.append(error_result)
                else:
                    final_results.append(result)
            
            return final_results
        else:
            # Process documents sequentially
            results = []
            for file_path, content, manual_metadata in documents:
                result = await self.process_document(file_path, content, manual_metadata, batch_config)
                results.append(result)
            
            return results
    
    async def _extract_metadata(
        self, 
        file_path: str, 
        content: str, 
        manual_metadata: Optional[Dict[str, Any]],
        config: PipelineConfig
    ) -> MetadataExtractorResult:
        """Extract metadata using the configured extractor"""
        try:
            # Validate content constraints
            if len(content) < config.min_content_length:
                return MetadataExtractorResult(
                    metadata=None,
                    extraction_method="failed",
                    confidence_score=0.0,
                    extraction_details={},
                    warnings=[f"Content too short: {len(content)} chars (min: {config.min_content_length})"],
                    errors=["Content length below minimum threshold"]
                )
            
            if len(content) > config.max_content_length:
                content = content[:config.max_content_length]
                logger.warning(f"Content truncated to {config.max_content_length} characters")
            
            # Extract metadata
            result = self.extractor.extract_metadata(file_path, content, manual_metadata)
            
            # Apply configuration-based post-processing
            if result.metadata:
                result.metadata = await self._post_process_metadata(result.metadata, config)
            
            return result
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            return MetadataExtractorResult(
                metadata=None,
                extraction_method="error",
                confidence_score=0.0,
                extraction_details={},
                warnings=[],
                errors=[f"Extraction error: {str(e)}"]
            )
    
    async def _validate_metadata(
        self, 
        metadata: DocumentMetadata, 
        config: PipelineConfig
    ) -> MetadataValidationResult:
        """Validate metadata using the configured validator"""
        try:
            # Perform validation
            result = self.validator.validate_metadata(metadata)
            
            # Apply configuration-based validation rules
            additional_errors = []
            additional_warnings = []
            
            # Check required category
            if config.require_category and not metadata.category:
                additional_errors.append("Category is required by pipeline configuration")
            
            # Check required author
            if config.require_author and not metadata.author:
                additional_errors.append("Author is required by pipeline configuration")
            
            # Check minimum completeness
            if result.completeness_score < config.min_completeness_score:
                additional_warnings.append(f"Completeness score {result.completeness_score:.2f} below threshold {config.min_completeness_score}")
            
            # Check minimum confidence
            if metadata.confidence_score and metadata.confidence_score < config.min_confidence_score:
                additional_warnings.append(f"Confidence score {metadata.confidence_score:.2f} below threshold {config.min_confidence_score}")
            
            # Add additional validation results
            result.errors.extend(additional_errors)
            result.warnings.extend(additional_warnings)
            
            # Update validation status
            if additional_errors:
                result.is_valid = False
            
            return result
            
        except Exception as e:
            logger.error(f"Metadata validation failed: {str(e)}")
            return MetadataValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                suggestions=[],
                completeness_score=0.0
            )
    
    async def _assess_quality(
        self, 
        extraction_result: MetadataExtractorResult,
        validation_result: MetadataValidationResult,
        config: PipelineConfig
    ) -> Dict[str, Any]:
        """Assess overall quality of extracted metadata"""
        try:
            from app.utils.metadata_validators import get_metadata_quality_score
            
            if not extraction_result.metadata:
                return {
                    "confidence_score": 0.0,
                    "completeness_score": 0.0,
                    "quality_grade": "F",
                    "overall_quality": 0.0
                }
            
            # Get quality assessment
            quality_score = get_metadata_quality_score(extraction_result.metadata)
            
            # Calculate overall quality (weighted combination of metrics)
            confidence_weight = 0.3
            completeness_weight = 0.4
            error_penalty_weight = 0.2
            warning_penalty_weight = 0.1
            
            confidence_score = extraction_result.confidence_score or 0.0
            completeness_score = validation_result.completeness_score
            error_penalty = min(1.0, len(validation_result.errors) * 0.2)  # Up to 1.0 penalty
            warning_penalty = min(0.5, len(validation_result.warnings) * 0.1)  # Up to 0.5 penalty
            
            overall_quality = (
                confidence_score * confidence_weight +
                completeness_score * completeness_weight -
                error_penalty * error_penalty_weight -
                warning_penalty * warning_penalty_weight
            )
            overall_quality = max(0.0, min(1.0, overall_quality))  # Clamp to [0, 1]
            
            return {
                "confidence_score": confidence_score,
                "completeness_score": completeness_score,
                "quality_grade": quality_score["quality_grade"],
                "overall_quality": overall_quality,
                "error_count": quality_score["error_count"],
                "warning_count": quality_score["warning_count"],
                "suggestion_count": quality_score["suggestion_count"]
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return {
                "confidence_score": 0.0,
                "completeness_score": 0.0,
                "quality_grade": "F",
                "overall_quality": 0.0
            }
    
    async def _post_process_metadata(
        self, 
        metadata: DocumentMetadata, 
        config: PipelineConfig
    ) -> DocumentMetadata:
        """Apply post-processing to extracted metadata"""
        try:
            # Update processing timestamps
            metadata.processed_at = datetime.utcnow()
            
            # Ensure chunk information is set for large content
            if metadata.text and len(metadata.text) > config.chunk_size_threshold:
                if not metadata.chunk_size:
                    metadata.chunk_size = len(metadata.text)
                if not metadata.total_chunks:
                    metadata.total_chunks = 1
                if metadata.chunk_index is None:
                    metadata.chunk_index = 0
            
            # Apply automatic enhancements
            if config.auto_detect_worldview and not metadata.worldview and metadata.category:
                metadata.worldview = metadata.category
            
            # Set default source type if not specified
            if not metadata.source_type:
                metadata.source_type = SourceType.BOOK
            
            # Ensure embedding dimension is set
            if not metadata.embedding_dimension:
                metadata.embedding_dimension = 768  # Default for our embedding service
            
            return metadata
            
        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            return metadata
    
    def _create_final_result(
        self,
        extraction_result: MetadataExtractorResult,
        validation_result: MetadataValidationResult,
        quality_metrics: Dict[str, Any],
        start_time: datetime,
        config: PipelineConfig
    ) -> ExtractionResult:
        """Create the final extraction result"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Determine success based on configuration
        success = True
        errors = []
        warnings = []
        suggestions = []
        
        # Collect all errors, warnings, suggestions
        if extraction_result:
            errors.extend(extraction_result.errors)
            warnings.extend(extraction_result.warnings)
        
        if validation_result:
            errors.extend(validation_result.errors)
            warnings.extend(validation_result.warnings)
            suggestions.extend(validation_result.suggestions)
        
        # Check success conditions
        if config.fail_on_extraction_errors and extraction_result and extraction_result.errors:
            success = False
        
        if config.fail_on_validation_errors and validation_result and not validation_result.is_valid:
            success = False
        
        # Check quality thresholds
        quality_grade = quality_metrics.get("quality_grade", "F")
        if self._grade_to_number(quality_grade) < self._grade_to_number(config.quality_grade_threshold):
            warnings.append(f"Quality grade {quality_grade} below threshold {config.quality_grade_threshold}")
        
        return ExtractionResult(
            success=success,
            metadata=extraction_result.metadata if extraction_result else None,
            extraction_result=extraction_result,
            validation_result=validation_result,
            processing_time_ms=processing_time,
            confidence_score=quality_metrics.get("confidence_score", 0.0),
            completeness_score=quality_metrics.get("completeness_score", 0.0),
            quality_grade=quality_grade,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            extraction_method="pipeline",
            extraction_details={
                "config": asdict(config),
                "quality_metrics": quality_metrics,
                "processing_stages": ["extraction", "validation", "quality_assessment", "post_processing"]
            }
        )
    
    def _create_error_result(
        self, 
        errors: List[str], 
        extraction_result: Optional[MetadataExtractorResult],
        start_time: datetime
    ) -> ExtractionResult:
        """Create an error result"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ExtractionResult(
            success=False,
            metadata=None,
            extraction_result=extraction_result,
            validation_result=None,
            processing_time_ms=processing_time,
            confidence_score=0.0,
            completeness_score=0.0,
            quality_grade="F",
            errors=errors,
            warnings=[],
            suggestions=[],
            extraction_method="error",
            extraction_details={}
        )
    
    def _apply_config_overrides(self, overrides: Optional[Dict[str, Any]]) -> PipelineConfig:
        """Apply configuration overrides"""
        if not overrides:
            return self.config
        
        # Create a copy of the current config
        config_dict = asdict(self.config)
        config_dict.update(overrides)
        
        return PipelineConfig(**config_dict)
    
    def _create_cache_key(
        self, 
        file_path: str, 
        content: str, 
        manual_metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Create a cache key for the extraction result"""
        cache_data = {
            "file_path": file_path,
            "content_hash": hashlib.sha256(content.encode('utf-8')).hexdigest(),
            "manual_metadata": manual_metadata or {},
            "config_hash": hashlib.sha256(str(asdict(self.config)).encode('utf-8')).hexdigest()
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode('utf-8')).hexdigest()
    
    def _grade_to_number(self, grade: str) -> int:
        """Convert letter grade to number for comparison"""
        grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        return grade_map.get(grade.upper(), 0)
    
    def _update_stats(self, result: ExtractionResult):
        """Update pipeline statistics"""
        self._stats["total_processed"] += 1
        
        if result.success:
            self._stats["successful_extractions"] += 1
        else:
            self._stats["failed_extractions"] += 1
        
        if result.validation_result and not result.validation_result.is_valid:
            self._stats["validation_failures"] += 1
        
        # Update averages
        total = self._stats["total_processed"]
        self._stats["average_confidence"] = (
            (self._stats["average_confidence"] * (total - 1) + result.confidence_score) / total
        )
        self._stats["average_completeness"] = (
            (self._stats["average_completeness"] * (total - 1) + result.completeness_score) / total
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics"""
        return {
            **self._stats,
            "success_rate": (
                self._stats["successful_extractions"] / max(1, self._stats["total_processed"])
            ),
            "validation_success_rate": (
                (self._stats["total_processed"] - self._stats["validation_failures"]) / 
                max(1, self._stats["total_processed"])
            ),
            "cache_size": len(self._cache) if self._cache else 0
        }
    
    def clear_cache(self):
        """Clear the extraction cache"""
        if self._cache:
            self._cache.clear()
            logger.info("Pipeline cache cleared")
    
    def reset_stats(self):
        """Reset pipeline statistics"""
        self._stats = {
            "total_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "validation_failures": 0,
            "average_confidence": 0.0,
            "average_completeness": 0.0
        }
        logger.info("Pipeline statistics reset")


# Convenience functions for easy pipeline usage

async def extract_document_metadata(
    file_path: str,
    content: str,
    manual_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[PipelineConfig] = None
) -> ExtractionResult:
    """
    Extract metadata from a single document using the pipeline
    
    Args:
        file_path: Path to the document
        content: Document content
        manual_metadata: Manual metadata overrides
        config: Pipeline configuration
        
    Returns:
        ExtractionResult with complete processing information
    """
    pipeline = MetadataPipeline(config)
    return await pipeline.process_document(file_path, content, manual_metadata)


async def extract_batch_metadata(
    documents: List[Tuple[str, str, Optional[Dict[str, Any]]]],
    config: Optional[PipelineConfig] = None
) -> List[ExtractionResult]:
    """
    Extract metadata from multiple documents using the pipeline
    
    Args:
        documents: List of (file_path, content, manual_metadata) tuples
        config: Pipeline configuration
        
    Returns:
        List of ExtractionResult objects
    """
    pipeline = MetadataPipeline(config)
    return await pipeline.process_batch(documents)


def create_production_config() -> PipelineConfig:
    """Create a production-ready pipeline configuration"""
    return PipelineConfig(
        strict_validation=False,
        min_completeness_score=0.6,
        require_category=True,
        require_author=False,
        fail_on_validation_errors=False,
        fail_on_extraction_errors=False,
        log_validation_warnings=True,
        quality_grade_threshold="C",
        enable_caching=True,
        cache_extraction_results=True,
        parallel_processing=True
    )


def create_development_config() -> PipelineConfig:
    """Create a development-friendly pipeline configuration"""
    return PipelineConfig(
        strict_validation=True,
        min_completeness_score=0.8,
        require_category=True,
        require_author=True,
        fail_on_validation_errors=True,
        fail_on_extraction_errors=False,
        log_validation_warnings=True,
        quality_grade_threshold="B",
        enable_caching=False,
        cache_extraction_results=False,
        parallel_processing=False
    ) 