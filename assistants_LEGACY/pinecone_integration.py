#!/usr/bin/env python3
"""
Pinecone Integration for Philosophical Assistants

This script provides functions to integrate philosophical assistants with Pinecone's
vector database for knowledge retrieval. It allows assistants to query the vector store
based on their specific philosophical worldview.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
import httpx
import asyncio
from dotenv import load_dotenv

# Import local modules
from .common_instructions import compose_instructions, update_assistant_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST = os.environ.get("PINECONE_HOST")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# Embedding service configuration
EMBEDDING_SERVICE_URL = os.environ.get("EMBEDDING_SERVICE_URL", "http://localhost:8001")

# Worldviews
WORLDVIEWS = ["Idealismus", "Materialismus", "Realismus", "Spiritualismus"]

class EmbeddingClient:
    """Client for the personal-embeddings-service."""
    
    def __init__(self, base_url: str = EMBEDDING_SERVICE_URL):
        """Initialize the embedding client.
        
        Args:
            base_url: URL of the embedding service
        """
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/v1/embeddings"
        
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.endpoint,
                    json={"texts": text},
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()["embeddings"][0]
            except Exception as e:
                logger.error(f"Error embedding text: {e}")
                raise
    
    async def embed_batch(self, texts: List[str], chunk_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            chunk_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings
        """
        if len(texts) <= chunk_size:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.endpoint,
                    json={"texts": texts},
                    timeout=60.0
                )
                return response.json()["embeddings"]
        else:
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), chunk_size):
                batch = texts[i:i+chunk_size]
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.endpoint,
                        json={"texts": batch},
                        timeout=60.0
                    )
                    all_embeddings.extend(response.json()["embeddings"])
            return all_embeddings

class PineconeClient:
    """Client for Pinecone vector database."""
    
    def __init__(
        self, 
        api_key: str = PINECONE_API_KEY,
        host: str = PINECONE_HOST,
        index_name: str = PINECONE_INDEX_NAME
    ):
        """Initialize the Pinecone client.
        
        Args:
            api_key: Pinecone API key
            host: Pinecone host URL
            index_name: Name of the Pinecone index
        """
        self.api_key = api_key
        self.host = host
        self.index_name = index_name
        self.base_url = f"{host}/indexes/{index_name}"
        self.embedding_client = EmbeddingClient()
        
        # Validate configuration
        if not all([api_key, host, index_name]):
            raise ValueError(
                "Missing Pinecone configuration. "
                "Please set PINECONE_API_KEY, PINECONE_HOST, and PINECONE_INDEX_NAME."
            )
    
    async def query_by_worldview(
        self, 
        query_text: str, 
        worldview: str,
        top_k: int = 10,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Query the vector store filtered by worldview.
        
        Args:
            query_text: Text to search for
            worldview: Philosophical worldview to filter by
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            Query results
        """
        if worldview not in WORLDVIEWS:
            raise ValueError(f"Invalid worldview: {worldview}. Must be one of {WORLDVIEWS}")
        
        # Generate embedding for query
        query_embedding = await self.embedding_client.embed_text(query_text)
        
        # Prepare the query request
        headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "vector": query_embedding,
            "filter": {"worldview": worldview},
            "topK": top_k,
            "includeMetadata": include_metadata
        }
        
        # Execute the query
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/query",
                    headers=headers,
                    json=data,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"Error querying Pinecone: {e}")
                raise

class AssistantManager:
    """Manager for philosophical assistants."""
    
    def __init__(self, config_dir: str = "assistants/config"):
        """Initialize the assistant manager.
        
        Args:
            config_dir: Directory containing assistant configuration files
        """
        self.config_dir = config_dir
        self.pinecone_client = PineconeClient()
        self.assistants = {}
        self._load_assistants()
    
    def _load_assistants(self) -> None:
        """Load assistant configurations from JSON files."""
        try:
            for worldview in WORLDVIEWS:
                config_path = os.path.join(self.config_dir, f"{worldview.lower()}.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        
                        # Apply common instructions pattern
                        config = update_assistant_config(config)
                        
                        self.assistants[worldview] = config
                        logger.info(f"Loaded configuration for {worldview} assistant")
                else:
                    logger.warning(f"Configuration file not found for {worldview}")
        except Exception as e:
            logger.error(f"Error loading assistant configurations: {e}")
            raise
    
    def get_assistant(self, worldview: str) -> Optional[Dict[str, Any]]:
        """Get assistant configuration by worldview.
        
        Args:
            worldview: Philosophical worldview
            
        Returns:
            Assistant configuration or None if not found
        """
        return self.assistants.get(worldview)
    
    def list_assistants(self) -> List[Dict[str, Any]]:
        """List all available assistants.
        
        Returns:
            List of assistant configurations
        """
        return list(self.assistants.values())
    
    async def query_knowledge_base(
        self, 
        worldview: str, 
        query_text: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Query the knowledge base for a specific worldview.
        
        Args:
            worldview: Philosophical worldview
            query_text: Text to search for
            top_k: Number of results to return
            
        Returns:
            Query results
        """
        if worldview not in WORLDVIEWS:
            raise ValueError(f"Invalid worldview: {worldview}. Must be one of {WORLDVIEWS}")
        
        results = await self.pinecone_client.query_by_worldview(
            query_text=query_text,
            worldview=worldview,
            top_k=top_k
        )
        
        return results

    def save_assistant_config(self, assistant_id: str = None, worldview: str = None) -> None:
        """Save an assistant configuration to file.
        
        Args:
            assistant_id: ID of the assistant to save (optional)
            worldview: Worldview of the assistant to save (optional)
            
        Note:
            Either assistant_id or worldview must be provided
        """
        if assistant_id is None and worldview is None:
            raise ValueError("Either assistant_id or worldview must be provided")
        
        # Find the assistant config
        assistant = None
        if assistant_id is not None:
            for worldview_name, config in self.assistants.items():
                if config.get("id") == assistant_id:
                    assistant = config
                    worldview = worldview_name
                    break
        else:
            assistant = self.assistants.get(worldview)
        
        if assistant is None:
            raise ValueError(f"Assistant not found with ID {assistant_id} or worldview {worldview}")
        
        # Save the assistant config
        config_path = os.path.join(self.config_dir, f"{worldview.lower()}.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(assistant, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Saved configuration for {worldview} assistant")

async def main():
    """Main function for testing."""
    try:
        # Initialize assistant manager
        manager = AssistantManager()
        
        # List available assistants
        assistants = manager.list_assistants()
        print(f"Found {len(assistants)} assistants:")
        for assistant in assistants:
            print(f"- {assistant['name']} ({assistant['weltanschauung']})")
        
        # Test query for each worldview
        for worldview in WORLDVIEWS:
            if worldview in manager.assistants:
                print(f"\nTesting query for {worldview}:")
                results = await manager.query_knowledge_base(
                    worldview=worldview,
                    query_text="Was ist die Beziehung zwischen Geist und Materie?",
                    top_k=3
                )
                
                print(f"Found {len(results.get('matches', []))} matches:")
                for i, match in enumerate(results.get('matches', [])):
                    print(f"{i+1}. Score: {match.get('score', 0):.4f}")
                    print(f"   Text: {match.get('metadata', {}).get('text', '')[:100]}...")
            else:
                print(f"\nSkipping {worldview} - assistant not configured")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 