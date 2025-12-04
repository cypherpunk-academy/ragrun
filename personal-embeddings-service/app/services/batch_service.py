import asyncio
from typing import List, Dict, Any, Generator
import numpy as np
import logging
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class BatchEmbeddingService:
    """Service for processing large batches of texts efficiently."""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
    
    def _chunk_texts(self, texts: List[str], chunk_size: int) -> Generator[List[str], None, None]:
        """Split texts into chunks of specified size."""
        for i in range(0, len(texts), chunk_size):
            yield texts[i:i + chunk_size]
    
    async def process_batch(
        self, 
        texts: List[str], 
        chunk_size: int = None
    ) -> List[List[float]]:
        """
        Process a large batch of texts by splitting into smaller chunks.
        
        Args:
            texts: List of texts to process
            chunk_size: Size of each processing chunk (defaults to self.batch_size)
            
        Returns:
            List of embeddings as lists
        """
        if chunk_size is None:
            chunk_size = self.batch_size
        
        if len(texts) == 0:
            return []
        
        # If batch is small enough, process directly
        if len(texts) <= chunk_size:
            embeddings = await embedding_service.encode_texts(texts)
            return embeddings.tolist()
        
        # Process in chunks
        all_embeddings = []
        chunks = list(self._chunk_texts(texts, chunk_size))
        
        logger.info(f"Processing {len(texts)} texts in {len(chunks)} chunks of size {chunk_size}")
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_embeddings = await embedding_service.encode_texts(chunk)
                all_embeddings.extend(chunk_embeddings.tolist())
                
                if (i + 1) % 10 == 0:  # Log progress every 10 chunks
                    logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
                    
            except Exception as e:
                logger.error(f"Failed to process chunk {i + 1}: {str(e)}")
                raise
        
        logger.info(f"Completed processing {len(texts)} texts")
        return all_embeddings
    
    async def process_documents_with_metadata(
        self, 
        documents: List[Dict[str, Any]], 
        text_field: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Process documents with metadata, adding embeddings to each document.
        
        Args:
            documents: List of documents with metadata
            text_field: Field name containing the text to embed
            
        Returns:
            List of documents with added embedding field
        """
        if not documents:
            return []
        
        # Extract texts for embedding
        texts = []
        for doc in documents:
            if text_field not in doc:
                raise ValueError(f"Document missing required field: {text_field}")
            texts.append(doc[text_field])
        
        # Get embeddings
        embeddings = await self.process_batch(texts)
        
        # Add embeddings to documents
        result_documents = []
        for i, doc in enumerate(documents):
            result_doc = doc.copy()
            result_doc["embedding"] = embeddings[i]
            result_documents.append(result_doc)
        
        return result_documents
    
    async def similarity_search_batch(
        self,
        queries: List[str],
        documents: List[str],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform similarity search for multiple queries.
        
        Args:
            queries: List of search queries
            documents: List of documents to search
            top_k: Number of top results per query
            
        Returns:
            List of search results for each query
        """
        all_results = []
        
        for query in queries:
            results = await embedding_service.similarity_search(
                query, documents, top_k
            )
            all_results.append(results)
        
        return all_results

# Create instance
batch_service = BatchEmbeddingService() 