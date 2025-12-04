import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Union, Dict, Any
import logging
import time
from app.models.embedding_model import EmbeddingModel
from app.config import settings

logger = logging.getLogger(__name__)

class LocalEmbeddingService:
    """Asynchronous embedding service using the configured model."""
    
    def __init__(self):
        self.model = EmbeddingModel()
        self.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
        self.ready = False
        
    async def load_model(self):
        """Load the model asynchronously."""
        def _load_model():
            return self.model.load_model()
        
        logger.info(f"Loading {settings.model_name} model asynchronously")
        success = await asyncio.get_event_loop().run_in_executor(
            self.executor, _load_model
        )
        
        if success:
            self.ready = True
            logger.info("Embedding service ready")
        else:
            raise RuntimeError("Failed to load embedding model")
    
    async def encode_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode texts to embeddings asynchronously.
        
        Args:
            texts: Single text or list of texts to encode
            
        Returns:
            Numpy array of embeddings
        """
        if not self.ready:
            raise RuntimeError("Service not ready. Call load_model() first.")
        
        def _encode(texts_input):
            return self.model.encode(texts_input)
        
        # Run encoding in thread pool to avoid blocking
        embeddings = await asyncio.get_event_loop().run_in_executor(
            self.executor, _encode, texts
        )
        
        return embeddings
    
    async def similarity_search(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using local embeddings.
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of top results to return
            
        Returns:
            List of search results with scores
        """
        try:
            # Get embeddings for query and documents
            query_embedding = await self.encode_texts(query)
            doc_embeddings = await self.encode_texts(documents)
            
            # Handle single query embedding
            if query_embedding.ndim == 2:
                query_embedding = query_embedding[0]
            
            # Calculate cosine similarity
            similarities = np.dot(doc_embeddings, query_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top_k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Format results
            results = []
            for idx in top_indices:
                results.append({
                    "index": int(idx),
                    "document": documents[idx],
                    "score": float(similarities[idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {str(e)}")
            raise
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            "ready": self.ready,
            "model_info": self.model.get_model_info() if self.model else None,
            "max_workers": settings.max_workers
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check with a test embedding."""
        try:
            if not self.ready:
                return {"status": "unhealthy", "error": "Service not ready"}
            
            # Quick test embedding
            start_time = time.time()
            test_embedding = await self.encode_texts("health check test")
            processing_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "model": self.model.model_name,
                "device": self.model.device,
                "embedding_dimension": test_embedding.shape[1],
                "test_processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

# Singleton instance
embedding_service = LocalEmbeddingService() 