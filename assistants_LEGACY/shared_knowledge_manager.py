#!/usr/bin/env python3
"""
Shared Knowledge Base Manager for Philosophical Assistants

This module manages the existing shared knowledge base that persists independently of individual assistants,
allowing temporary assistants to access philosophical texts without losing documents when cleaned up.

Works with existing index: german-philosophic-index-12-worldviews
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from pinecone import Pinecone

logger = logging.getLogger(__name__)

class SharedKnowledgeManager:
    """Manages the existing shared knowledge base for philosophical assistants."""
    
    def __init__(self, api_key: str = None, index_name: str = "german-philosophic-index-12-worldviews"):
        """Initialize the shared knowledge manager.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the existing shared knowledge index
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY must be provided or set as environment variable")
        
        self.pc = Pinecone(api_key=self.api_key)
        self.index_name = index_name
        self.index = None
        
        # Mapping of German worldview names to categories
        self.worldview_categories = {
            "Idealismus": "Idealismus",
            "Materialismus": "Materialismus", 
            "Realismus": "Realismus",
            "Spiritualismus": "Spiritualismus"
        }
        
    def connect_to_shared_index(self):
        """Connect to the existing shared knowledge index."""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                raise ValueError(f"Shared knowledge index '{self.index_name}' not found. Available indexes: {index_names}")
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to existing shared knowledge index: {self.index_name}")
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Index contains {stats.total_vector_count} documents across {len(stats.namespaces)} namespaces")
            
        except Exception as e:
            logger.error(f"Error connecting to shared index: {e}")
            raise
    
    def query_knowledge_base(
        self, 
        query_embedding: List[float], 
        worldview: str = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Query the existing shared knowledge base.
        
        Args:
            query_embedding: Query vector embedding
            worldview: Optional worldview filter (Idealismus, Materialismus, etc.)
            top_k: Number of results to return
            
        Returns:
            Query results
        """
        if not self.index:
            self.connect_to_shared_index()
        
        try:
            # Build filter for worldview category if specified
            filter_dict = {}
            if worldview and worldview in self.worldview_categories:
                filter_dict['category'] = self.worldview_categories[worldview]
            
            # Query the index
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            logger.info(f"Knowledge base query returned {len(results.matches)} results for {worldview or 'all worldviews'}")
            
            return {
                'matches': results.matches,
                'worldview_filter': worldview,
                'total_results': len(results.matches),
                'index_name': self.index_name
            }
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            raise
    
    def list_worldview_documents(self, worldview: str) -> Dict[str, Any]:
        """List documents for a specific worldview category.
        
        Args:
            worldview: Philosophical worldview (Idealismus, etc.)
            
        Returns:
            Document statistics
        """
        if not self.index:
            self.connect_to_shared_index()
        
        try:
            if worldview not in self.worldview_categories:
                raise ValueError(f"Unknown worldview: {worldview}. Available: {list(self.worldview_categories.keys())}")
            
            # Query with worldview filter to get document count
            stats = self.index.describe_index_stats(filter={'category': self.worldview_categories[worldview]})
            
            return {
                'worldview': worldview,
                'category': self.worldview_categories[worldview],
                'document_count': stats.total_vector_count,
                'index_name': self.index_name
            }
            
        except Exception as e:
            logger.error(f"Error listing documents for {worldview}: {e}")
            raise
    
    def get_available_worldviews(self) -> List[str]:
        """Get list of available worldviews in the knowledge base.
        
        Returns:
            List of worldview names
        """
        return list(self.worldview_categories.keys())
    
    def create_assistant_with_shared_knowledge(
        self, 
        assistant_name: str,
        worldview: str,
        instructions: str,
        model: str = "deepseek-reasoner"
    ) -> str:
        """Create instructions for an assistant that uses the shared knowledge base.
        
        Args:
            assistant_name: Name of the assistant
            worldview: Philosophical worldview
            instructions: Base instructions
            model: LLM model to use
            
        Returns:
            Enhanced instructions with shared knowledge access
        """
        if worldview not in self.worldview_categories:
            raise ValueError(f"Unknown worldview: {worldview}. Available: {list(self.worldview_categories.keys())}")
        
        enhanced_instructions = f"""
{instructions}

WICHTIG - WISSENSQUELLE:
Du hast Zugang zu einer umfangreichen philosophischen Wissensbasis mit Texten zum {worldview}.
Diese Texte sind in einem separaten, persistenten Vektor-Index gespeichert und bleiben auch dann erhalten,
wenn temporäre Assistenten gelöscht werden.

Deine Wissensbasis:
- Index: {self.index_name}
- Kategorie: {self.worldview_categories[worldview]}
- Weltanschauung: {worldview}
- Assistent: {assistant_name}

Wenn du Fragen beantwortest:
1. Nutze primär dein philosophisches Verständnis des {worldview}
2. Falls verfügbar, ergänze deine Antworten mit spezifischen Textverweisen aus der Wissensbasis
3. Die Wissensbasis enthält relevante philosophische Texte für den {worldview}
4. Du kannst die Dateisuche nutzen, um spezifische Informationen zu finden

Die Dokumente in deiner Wissensbasis sind dauerhaft verfügbar und gehen nicht verloren,
auch wenn dieser temporäre Assistent später gelöscht wird.
"""
        return enhanced_instructions

# Integration with existing assistant manager
class CostOptimizedAssistantManager:
    """Cost-optimized assistant manager using the existing shared knowledge base."""
    
    def __init__(self):
        self.knowledge_manager = SharedKnowledgeManager()
        # Import here to avoid circular imports
        from assistants.deepseek_assistant_manager import DeepSeekAssistantManager as PineconeAssistantManager
        self.assistant_manager = PineconeAssistantManager()
    
    def create_temporary_assistant_with_knowledge(
        self,
        worldview: str,
        session_id: str = None,
        auto_cleanup_minutes: int = 60
    ):
        """Create a temporary assistant with access to the existing shared knowledge base.
        
        Args:
            worldview: Philosophical worldview (Idealismus, Materialismus, Realismus, Spiritualismus)
            session_id: Unique session identifier (auto-generated if not provided)
            auto_cleanup_minutes: Minutes after which to cleanup
            
        Returns:
            Assistant instance
        """
        # Generate session ID if not provided
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())[:8]
        
        # Create temporary assistant name
        assistant_name = f"temp-{worldview.lower()}-{session_id}"
        
        # Get enhanced instructions with shared knowledge access
        base_instructions = self._get_base_instructions(worldview)
        enhanced_instructions = self.knowledge_manager.create_assistant_with_shared_knowledge(
            assistant_name, worldview, base_instructions
        )
        
        # Create temporary assistant (without uploading documents - they're already in shared index)
        assistant = self.assistant_manager.get_or_create_temporary_assistant(
            name=assistant_name,
            worldview=worldview,
            instructions=enhanced_instructions,
            auto_cleanup_hours=auto_cleanup_minutes // 60
        )
        
        logger.info(f"Created temporary assistant {assistant_name} with access to shared knowledge base")
        logger.info(f"Documents for {worldview} are safely stored in: {self.knowledge_manager.index_name}")
        
        return assistant
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the shared knowledge base.
        
        Returns:
            Knowledge base statistics
        """
        stats = {}
        for worldview in self.knowledge_manager.get_available_worldviews():
            try:
                worldview_stats = self.knowledge_manager.list_worldview_documents(worldview)
                stats[worldview] = worldview_stats
            except Exception as e:
                stats[worldview] = {"error": str(e)}
        
        return {
            "index_name": self.knowledge_manager.index_name,
            "worldviews": stats,
            "total_worldviews": len(stats)
        }
    
    def _get_base_instructions(self, worldview: str) -> str:
        """Get base instructions for a worldview."""
        instructions_map = {
            "Idealismus": "Du bist Aurelian I. Schelling, ein philosophischer Berater des Idealismus...",
            "Materialismus": "Du bist Aloys I. Freud, ein philosophischer Berater des Materialismus...", 
            "Realismus": "Du bist Arvid I. Steiner, ein philosophischer Berater des Realismus...",
            "Spiritualismus": "Du bist Amara I. Steiner, ein philosophische Beraterin des Spiritualismus..."
        }
        return instructions_map.get(worldview, "Du bist ein philosophischer Assistent.") 