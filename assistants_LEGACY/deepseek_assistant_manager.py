#!/usr/bin/env python3
"""
DeepSeek + Pinecone Hybrid Assistant Manager

Hybrid approach combining the best of both:
- Pinecone for RAG search (excellent vector database)
- DeepSeek API for language model (95% cost savings vs Pinecone Assistants)

Cost comparison:
- Pinecone Assistants: 12 assistants × $1.20/day = $5,184/year
- Hybrid approach: ~$50-200/year DeepSeek + minimal Pinecone search costs
- Savings: $5,000+ annually with SAME search quality + SAME model quality!

Now uses code-only assistant definitions for better development workflow.
"""

import os
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import openai
from pinecone import Pinecone

logger = logging.getLogger(__name__)

class DeepSeekAssistantManager:
    """Hybrid assistant manager using Pinecone for RAG search and DeepSeek for LLM."""
    
    def __init__(self, api_key: str = None, pinecone_api_key: str = None, development_mode: bool = False):
        """Initialize the hybrid assistant manager.
        
        Args:
            api_key: DeepSeek API key
            pinecone_api_key: Pinecone API key (for RAG search only)
            development_mode: Enable development features (hot reload, debug logging)
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY must be provided or set as environment variable")
        
        self.pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY must be provided or set as environment variable")
        
        # Configure OpenAI client for DeepSeek
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
        
        # Initialize Pinecone for direct index access (no Assistant API)
        self.pinecone_api_key = pinecone_api_key
        self._pinecone_index = None  # Will be initialized lazily
        
        # Assistant configurations - mimics Pinecone structure
        self.assistant_configs = {}
        
        # Development mode
        self.development_mode = development_mode
        
        # Usage tracking
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "daily_cost": 0.0,
            "last_reset": datetime.now().date()
        }
        
        # Load philosophical assistants from code definitions
        self._load_assistants_from_definitions()
        
        logger.info("Initialized hybrid DeepSeek + Pinecone assistant manager")
    
    def _load_assistants_from_definitions(self):
        """Load assistants from the code-only assistant definitions file."""
        try:
            # Import the assistant definitions
            from .assistant_definitions import PHILOSOPHICAL_ASSISTANTS
            
            # Convert to internal format
            for assistant_id, definition in PHILOSOPHICAL_ASSISTANTS.items():
                self.assistant_configs[assistant_id] = {
                    "name": definition.name,
                    "worldview": definition.worldview.value,
                    "instructions": definition.instructions,
                    "model": definition.model,
                    "temperature": definition.temperature,
                    "max_tokens": definition.max_tokens,
                    "created_on": datetime.now().isoformat(),
                    "status": "Ready",
                    "total_queries": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    # Development features
                    "development_mode": definition.development_mode,
                    "debug_logging": definition.debug_logging,
                    "version": definition.version,
                    "author": definition.author,
                    "description": definition.description
                }
            
            logger.info(f"Loaded {len(PHILOSOPHICAL_ASSISTANTS)} assistants from code definitions")
            
        except ImportError as e:
            logger.error(f"Could not import assistant definitions: {e}")
            logger.info("Falling back to minimal default assistants")
            self._create_minimal_fallback_assistants()
        except Exception as e:
            logger.error(f"Error loading assistant definitions: {e}")
            self._create_minimal_fallback_assistants()
    
    def _create_minimal_fallback_assistants(self):
        """Create minimal fallback assistants if code definitions can't be loaded."""
        fallback_assistants = {
            "aurelian-i--schelling": {
                "name": "Aurelian I. Schelling",
                "worldview": "Idealismus",
                "instructions": "Du bist Aurelian I. Schelling, ein philosophischer Berater des Idealismus."
            },
            "aloys-i--freud": {
                "name": "Aloys I. Freud", 
                "worldview": "Materialismus",
                "instructions": "Du bist Aloys I. Freud, ein philosophischer Berater des Materialismus."
            }
        }
        
        for assistant_id, config in fallback_assistants.items():
            self.assistant_configs[assistant_id] = {
                "name": config["name"],
                "worldview": config["worldview"],
                "instructions": config["instructions"],
                "model": "deepseek-reasoner",
                "temperature": 0.7,
                "max_tokens": 2000,
                "created_on": datetime.now().isoformat(),
                "status": "Ready",
                "total_queries": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "development_mode": False,
                "debug_logging": False,
                "version": "1.0.0",
                "author": "Fallback",
                "description": "Minimal fallback assistant"
            }
        
        logger.warning(f"Created {len(fallback_assistants)} fallback assistants")
    
    def reload_assistant_definitions(self):
        """Reload assistant definitions from code (useful for development)."""
        if not self.development_mode:
            logger.warning("reload_assistant_definitions called but development_mode is False")
            return
        
        try:
            # Clear existing configs
            self.assistant_configs.clear()
            
            # Reimport the definitions module to get latest changes
            import importlib
            from . import assistant_definitions
            importlib.reload(assistant_definitions)
            
            # Reload assistants
            self._load_assistants_from_definitions()
            
            logger.info("Successfully reloaded assistant definitions from code")
            
        except Exception as e:
            logger.error(f"Error reloading assistant definitions: {e}")
            # Restore minimal fallback
            self._create_minimal_fallback_assistants()
    
    def _get_pinecone_index(self):
        """Get the Pinecone index for direct querying (no Assistant API)."""
        if not hasattr(self, '_pinecone_index') or self._pinecone_index is None:
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.pinecone_api_key)
                self._pinecone_index = pc.Index("german-philosophic-index-12-worldviews")
                logger.info("Connected to Pinecone index directly")
            except Exception as e:
                logger.error(f"Error connecting to Pinecone index: {e}")
                self._pinecone_index = None
        
        return self._pinecone_index
    

    
    def _get_assistant_documents(self, assistant_id: str) -> List[Dict[str, Any]]:
        """Get the actual documents available to an assistant via direct Pinecone index query."""
        try:
            # Get the assistant's worldview to filter documents
            config = self.assistant_configs.get(assistant_id, {})
            worldview = config.get("worldview", "Unknown")
            
            # Get Pinecone index directly (no Assistant API)
            index = self._get_pinecone_index()
            if index:
                # First, try to query without filter to see what dimensions work and what metadata is available
                working_dimension = None
                sample_metadata = {}
                
                # Try different vector dimensions to find the one that works
                expected_dimension = int(os.environ.get("EMBEDDINGS_DIMENSION", "768"))
                dimensions_to_try = [expected_dimension, 384, 512, 768, 1024, 1536]
                # Remove duplicates while preserving order
                dimensions_to_try = list(dict.fromkeys(dimensions_to_try))
                
                for dimension in dimensions_to_try:
                    try:
                        logger.debug(f"Trying {dimension} dimensions...")
                        query_response = index.query(
                            vector=[0.0] * dimension,
                            top_k=10,  # Just get a few samples first
                            include_metadata=True
                        )
                        
                        if query_response.matches:
                            working_dimension = dimension
                            # Sample the metadata to understand the structure
                            for match in query_response.matches[:3]:
                                if match.metadata:
                                    sample_metadata = match.metadata
                                    logger.debug(f"Sample metadata: {sample_metadata}")
                                    break
                            break
                            
                    except Exception as dim_e:
                        logger.debug(f"Failed with {dimension} dimensions: {dim_e}")
                        continue
                
                if not working_dimension:
                    logger.warning("Could not find working dimension for Pinecone index")
                    return []
                
                logger.info(f"Using {working_dimension} dimensions for Pinecone queries")
                
                # Now try to find the right metadata filter for worldview
                worldview_filter = None
                
                # Check common metadata field names for worldview
                possible_worldview_fields = ["worldview", "category", "weltanschauung", "topic", "subject"]
                for field in possible_worldview_fields:
                    if field in sample_metadata:
                        logger.debug(f"Found potential worldview field: {field} = {sample_metadata[field]}")
                        # Try filtering with this field
                        try:
                            test_response = index.query(
                                vector=[0.0] * working_dimension,
                                top_k=10,
                                include_metadata=True,
                                filter={field: worldview}
                            )
                            if test_response.matches:
                                worldview_filter = {field: worldview}
                                logger.info(f"Using filter: {worldview_filter}")
                                break
                        except Exception as filter_e:
                            logger.debug(f"Filter {field}={worldview} failed: {filter_e}")
                
                # If no exact worldview match, try case variations
                if not worldview_filter and worldview != "Unknown":
                    worldview_variations = [
                        worldview.lower(),
                        worldview.upper(), 
                        worldview.capitalize()
                    ]
                    
                    for field in possible_worldview_fields:
                        for variation in worldview_variations:
                            try:
                                test_response = index.query(
                                    vector=[0.0] * working_dimension,
                                    top_k=10,
                                    include_metadata=True,
                                    filter={field: variation}
                                )
                                if test_response.matches:
                                    worldview_filter = {field: variation}
                                    logger.info(f"Using filter with variation: {worldview_filter}")
                                    break
                            except:
                                continue
                        if worldview_filter:
                            break
                
                # Final query with or without filter
                try:
                    query_response = index.query(
                        vector=[0.0] * working_dimension,
                        top_k=10000,  # Get many matches
                        include_metadata=True,
                        filter=worldview_filter
                    )
                    
                    # Extract unique documents
                    unique_docs = {}
                    for match in query_response.matches:
                        if match.metadata:
                            source = match.metadata.get("source", match.metadata.get("title", f"doc_{match.id}"))
                            if source not in unique_docs:
                                unique_docs[source] = match.metadata
                    
                    documents = list(unique_docs.values())
                    documents.sort(key=lambda x: x.get("title", x.get("source", "")))
                    
                    filter_desc = f"with filter {worldview_filter}" if worldview_filter else "without filter"
                    logger.info(f"Found {len(documents)} unique documents for {worldview} {filter_desc}")
                    
                    # If no documents found with filter, try without filter to see what's available
                    if not documents and worldview_filter:
                        logger.info("No documents found with filter, trying without filter to see available data...")
                        query_response = index.query(
                            vector=[0.0] * working_dimension,
                            top_k=20,  # Just get a sample
                            include_metadata=True
                        )
                        
                        logger.info("Available metadata fields in index:")
                        for i, match in enumerate(query_response.matches[:5]):
                            if match.metadata:
                                logger.info(f"Document {i+1}: {list(match.metadata.keys())}")
                                logger.info(f"  Sample values: {match.metadata}")
                    
                    return documents
                    
                except Exception as e:
                    logger.error(f"Final query failed: {e}")
                    return []
            
            return []
                
        except Exception as e:
            logger.warning(f"Could not get documents for assistant {assistant_id}: {e}")
            return []
    
    # ===== PINECONE ASSISTANT MANAGER COMPATIBLE INTERFACE =====
    
    def list_assistants(self) -> List[Dict[str, Any]]:
        """List all assistants (compatible with PineconeAssistantManager interface)."""
        assistants = []
        for assistant_id, config in self.assistant_configs.items():
            assistants.append({
                "id": assistant_id,
                "name": config["name"],
                "created_on": config["created_on"],
                "status": config["status"],
                "model": config["model"],
                "worldview": config["worldview"],
                "total_queries": config["total_queries"],
                "total_cost": config["total_cost"]
            })
        return assistants
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return [
            "deepseek-reasoner",
            "deepseek-chat", 
            "deepseek-coder"
        ]
    
    def delete_assistant(self, assistant_name: str) -> bool:
        """Delete an assistant."""
        if assistant_name in self.assistant_configs:
            del self.assistant_configs[assistant_name]
            logger.info(f"Deleted assistant: {assistant_name}")
            return True
        return False
    
    def create_assistant(self, assistant_name: str, instructions: str, **kwargs) -> Any:
        """Create a new assistant (compatible interface)."""
        worldview = kwargs.get("worldview", "Unknown")
        model = kwargs.get("model", "deepseek-reasoner")
        
        config = {
            "name": assistant_name,
            "worldview": worldview,
            "instructions": instructions,
            "model": model,
            "created_on": datetime.now().isoformat(),
            "status": "Ready",
            "total_queries": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }
        
        self.assistant_configs[assistant_name] = config
        logger.info(f"Created assistant: {assistant_name}")
        
        return MockAssistant(assistant_name, self)
    
    def chat_with_assistant(
        self, 
        assistant: Any, 
        message: str, 
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Chat with an assistant (compatible with PineconeAssistantManager interface)."""
        assistant_name = assistant.name if hasattr(assistant, 'name') else str(assistant)
        
        return self.query_assistant(
            assistant_id=assistant_name,
            user_message=message,
            chat_history=chat_history
        )
    
    # ===== HYBRID IMPLEMENTATION: PINECONE SEARCH + DEEPSEEK LLM =====
    
    def query_assistant(
        self,
        assistant_id: str,
        user_message: str,
        chat_history: List[Dict[str, str]] = None,
        use_knowledge_base: bool = True,
        temperature: float = None,
        debug_mode: bool = False,
        model_override: str = None
    ) -> Dict[str, Any]:
        """Query assistant using hybrid approach: Pinecone search + DeepSeek LLM."""
        start_time = time.time()
        
        # Check daily cost reset
        self._check_daily_reset()
        
        try:
            if assistant_id not in self.assistant_configs:
                raise ValueError(f"Assistant {assistant_id} not found")
            
            config = self.assistant_configs[assistant_id]
            worldview = config["worldview"]
            
            # Use assistant's temperature if not overridden
            actual_temperature = temperature if temperature is not None else config.get("temperature", 0.7)
            actual_max_tokens = config.get("max_tokens", 2000)
            actual_model = model_override if model_override is not None else config.get("model", "deepseek-reasoner")
            
            # Debug logging if enabled
            debug_enabled = debug_mode or config.get("debug_logging", False) or self.development_mode
            if debug_enabled:
                logger.info(f"[DEBUG] Assistant: {assistant_id} ({config.get('name', 'Unknown')})")
                logger.info(f"[DEBUG] Worldview: {worldview}")
                logger.info(f"[DEBUG] Model: {actual_model} {'(overridden)' if model_override else '(default)'}")
                logger.info(f"[DEBUG] Temperature: {actual_temperature}")
                logger.info(f"[DEBUG] Max tokens: {actual_max_tokens}")
                logger.info(f"[DEBUG] User message: {user_message[:100]}...")
                logger.info(f"[DEBUG] Development mode: {config.get('development_mode', False)}")
                logger.info(f"[DEBUG] Version: {config.get('version', 'Unknown')}")
            
            # Prepare system message
            system_message = config["instructions"]
            
            # Get knowledge base context using Pinecone RAG search
            if use_knowledge_base:
                knowledge_context = self._get_pinecone_knowledge_context(
                    assistant_id, user_message, worldview
                )
                if knowledge_context:
                    system_message += f"\n\nRelevante Textstellen aus der Wissensbasis:\n{knowledge_context}"
                    if debug_enabled:
                        logger.info(f"[DEBUG] Knowledge context: {len(knowledge_context)} characters")
            
            # Prepare messages for DeepSeek
            messages = [{"role": "system", "content": system_message}]
            
            # Add chat history if provided
            if chat_history:
                for msg in chat_history:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
                if debug_enabled:
                    logger.info(f"[DEBUG] Chat history: {len(chat_history)} messages")
            
            # Add current message
            messages.append({"role": "user", "content": user_message})
            
            if debug_enabled:
                total_prompt_length = sum(len(msg["content"]) for msg in messages)
                logger.info(f"[DEBUG] Total prompt length: {total_prompt_length} characters")
            
            # Call DeepSeek API
            response = self.client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=actual_temperature,
                max_tokens=actual_max_tokens
            )
            
            # Extract response and usage
            assistant_response = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost (DeepSeek pricing)
            input_cost = usage.prompt_tokens * 0.00000014  # $0.14 per 1M tokens
            output_cost = usage.completion_tokens * 0.00000028  # $0.28 per 1M tokens
            total_cost = input_cost + output_cost
            
            # Update statistics
            config["total_queries"] += 1
            config["total_tokens"] += usage.total_tokens
            config["total_cost"] += total_cost
            
            self.usage_stats["total_requests"] += 1
            self.usage_stats["total_tokens"] += usage.total_tokens
            self.usage_stats["total_cost"] += total_cost
            self.usage_stats["daily_cost"] += total_cost
            
            processing_time = time.time() - start_time
            
            if debug_enabled:
                logger.info(f"[DEBUG] Response length: {len(assistant_response)} characters")
                logger.info(f"[DEBUG] Tokens used: {usage.total_tokens}")
                logger.info(f"[DEBUG] Cost: ${total_cost:.6f}")
                logger.info(f"[DEBUG] Processing time: {processing_time:.2f}s")
            
            # Build response
            response_data = {
                "message": assistant_response,
                "citations": [],  # Will be populated below
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "cost": total_cost
                },
                "model": actual_model,
                "processing_time": processing_time,
                "assistant_id": assistant_id,
                "worldview": worldview
            }
            
            # Get citations from knowledge base context
            if use_knowledge_base:
                rag_citations = self._get_pinecone_citations(
                    assistant_id, user_message, worldview
                )
                response_data["citations"] = rag_citations
                if debug_enabled:
                    logger.info(f"[DEBUG] RAG citations: {len(rag_citations)} results")

            # Add development info if in development mode
            if self.development_mode or debug_enabled:
                response_data["development_info"] = {
                    "temperature_used": actual_temperature,
                    "max_tokens_used": actual_max_tokens,
                    "debug_mode": debug_enabled,
                    "assistant_config": {
                        "name": config.get("name", "Unknown"),
                        "version": config.get("version", "Unknown"),
                        "author": config.get("author", "Unknown"),
                        "development_mode": config.get("development_mode", False)
                    },
                    "system_message_length": len(system_message),
                    "knowledge_base_used": use_knowledge_base
                }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error querying assistant {assistant_id}: {e}")
            raise
    
    def _get_pinecone_knowledge_context(
        self, 
        assistant_id: str, 
        query: str, 
        worldview: str, 
        top_k: int = 10
    ) -> str:
        """Get relevant context using direct Pinecone index search (no Assistant API)."""
        try:
            # Get Pinecone index directly
            index = self._get_pinecone_index()
            if not index:
                logger.warning("No Pinecone index available")
                return ""
            
            # Generate embedding for the query using the same model as the knowledge base
            try:
                # Use the same embedding model as specified in the environment
                model_name = os.environ.get("EMBEDDINGS_MODEL", "T-Systems-onsite/cross-en-de-roberta-sentence-transformer")
                
                # Use sentence-transformers library to generate embeddings
                from sentence_transformers import SentenceTransformer
                
                # Initialize the model (cache it for efficiency)
                if not hasattr(self, '_embedding_model') or self._embedding_model is None:
                    logger.info(f"Loading embedding model: {model_name}")
                    self._embedding_model = SentenceTransformer(model_name)
                
                # Generate embedding
                query_vector = self._embedding_model.encode(query).tolist()
                expected_dimensions = int(os.environ.get("EMBEDDINGS_DIMENSION", "768"))
                logger.debug(f"Generated embedding with {len(query_vector)} dimensions using {model_name} (expected: {expected_dimensions})")
                
            except Exception as e:
                logger.error(f"Error generating embedding for query with model {model_name}: {e}")
                logger.error(f"Query was: {query[:100]}...")
                return ""
            
            # Determine the correct metadata filter field
            metadata_filter = None
            if worldview != "Unknown":
                # Try common metadata field names
                possible_filters = [
                    {"worldview": worldview},
                    {"category": worldview},
                    {"weltanschauung": worldview},
                    {"worldview": worldview.lower()},
                    {"category": worldview.lower()}
                ]
                
                # Use the first filter that works, or none if none work
                for filter_option in possible_filters:
                    try:
                        # Test the filter with a small query
                        test_response = index.query(
                            vector=query_vector,
                            top_k=1,
                            include_metadata=True,
                            filter=filter_option
                        )
                        if test_response.matches:
                            metadata_filter = filter_option
                            logger.debug(f"Using metadata filter: {metadata_filter}")
                            break
                    except Exception as filter_e:
                        logger.debug(f"Filter {filter_option} failed: {filter_e}")
                        continue
            
            # Query Pinecone index directly
            query_response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=metadata_filter
            )
            
            # Format context from search results
            context_parts = []
            for i, match in enumerate(query_response.matches[:top_k], 1):
                content = match.metadata.get("text", "")[:500]  # Limit content length
                source = match.metadata.get("source", match.metadata.get("title", "Unknown"))
                score = match.score
                
                if content:
                    context_parts.append(f"[{i}] {source} (Score: {score:.3f}): {content}...")
            
            context = "\n".join(context_parts) if context_parts else ""
            return context
            
        except Exception as e:
            logger.error(f"Error getting Pinecone knowledge context: {e}")
            return ""
    
    def _get_pinecone_citations(
        self, 
        assistant_id: str, 
        query: str, 
        worldview: str, 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get citations from Pinecone search results."""
        try:
            # Get Pinecone index directly
            index = self._get_pinecone_index()
            if not index:
                logger.warning("No Pinecone index available for citations")
                return []
            
            # Generate embedding for the query
            try:
                model_name = os.environ.get("EMBEDDINGS_MODEL", "T-Systems-onsite/cross-en-de-roberta-sentence-transformer")
                
                from sentence_transformers import SentenceTransformer
                
                if not hasattr(self, '_embedding_model') or self._embedding_model is None:
                    logger.info(f"Loading embedding model for citations: {model_name}")
                    self._embedding_model = SentenceTransformer(model_name)
                
                query_vector = self._embedding_model.encode(query).tolist()
                
            except Exception as e:
                logger.error(f"Error generating embedding for citations: {e}")
                return []
            
            # Determine metadata filter
            metadata_filter = None
            if worldview != "Unknown":
                possible_filters = [
                    {"worldview": worldview},
                    {"category": worldview},
                    {"weltanschauung": worldview},
                    {"worldview": worldview.lower()},
                    {"category": worldview.lower()}
                ]
                
                for filter_option in possible_filters:
                    try:
                        test_response = index.query(
                            vector=query_vector,
                            top_k=1,
                            include_metadata=True,
                            filter=filter_option
                        )
                        if test_response.matches:
                            metadata_filter = filter_option
                            break
                    except:
                        continue
            
            # Query Pinecone for citations
            query_response = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=metadata_filter
            )
            
            # Format citations
            citations = []
            for match in query_response.matches:
                citation = {
                    "score": float(match.score),
                    "text": match.metadata.get("text", "")[:500],
                    "source": match.metadata.get("source", match.metadata.get("title", "Unknown")),
                    "id": match.id
                }
                
                # Add additional metadata if available
                if "author" in match.metadata:
                    citation["author"] = match.metadata["author"]
                if "year" in match.metadata:
                    citation["year"] = match.metadata["year"]
                if "page" in match.metadata:
                    citation["page"] = match.metadata["page"]
                
                citations.append(citation)
            
            logger.debug(f"Generated {len(citations)} citations from Pinecone search")
            return citations
            
        except Exception as e:
            logger.error(f"Error getting Pinecone citations: {e}")
            logger.exception("Full traceback:")
            return []
    
    def _check_daily_reset(self):
        """Reset daily cost tracking if it's a new day."""
        today = datetime.now().date()
        if today != self.usage_stats["last_reset"]:
            logger.info(f"Daily cost reset: ${self.usage_stats['daily_cost']:.4f} -> $0.0000")
            self.usage_stats["daily_cost"] = 0.0
            self.usage_stats["last_reset"] = today
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get detailed cost analysis compared to Pinecone Assistants."""
        assistant_count = len(self.assistant_configs)
        
        # Current usage statistics
        avg_cost_per_query = (self.usage_stats["total_cost"] / 
                             max(self.usage_stats["total_requests"], 1))
        
        # Projected costs for different usage levels
        queries_per_day_low = 100
        queries_per_day_high = 1000
        
        # DeepSeek costs + minimal Pinecone search costs
        deepseek_daily_low = queries_per_day_low * avg_cost_per_query
        deepseek_daily_high = queries_per_day_high * avg_cost_per_query
        
        # Add estimated Pinecone search costs (much cheaper than Assistant API)
        pinecone_search_daily = 0.10  # Estimated $0.10/day for search only
        
        hybrid_daily_low = deepseek_daily_low + pinecone_search_daily
        hybrid_daily_high = deepseek_daily_high + pinecone_search_daily
        
        # Pinecone Assistant API costs
        pinecone_assistant_daily = assistant_count * 1.20
        
        return {
            "backend": "Hybrid (DeepSeek LLM + Pinecone Search)",
            "assistant_count": assistant_count,
            "current_usage": self.usage_stats,
            "avg_cost_per_query": avg_cost_per_query,
            "projections": {
                "low_usage": {
                    "queries_per_day": queries_per_day_low,
                    "hybrid_daily": hybrid_daily_low,
                    "hybrid_yearly": hybrid_daily_low * 365,
                    "pinecone_assistant_daily": pinecone_assistant_daily,
                    "pinecone_assistant_yearly": pinecone_assistant_daily * 365,
                    "savings_yearly": (pinecone_assistant_daily * 365) - (hybrid_daily_low * 365)
                },
                "high_usage": {
                    "queries_per_day": queries_per_day_high,
                    "hybrid_daily": hybrid_daily_high,
                    "hybrid_yearly": hybrid_daily_high * 365,
                    "pinecone_assistant_daily": pinecone_assistant_daily,
                    "pinecone_assistant_yearly": pinecone_assistant_daily * 365,
                    "savings_yearly": (pinecone_assistant_daily * 365) - (hybrid_daily_high * 365)
                }
            }
        }

class MockAssistant:
    """Mock assistant object to maintain compatibility with existing code."""
    
    def __init__(self, name: str, manager: DeepSeekAssistantManager):
        self.name = name
        self.manager = manager
    
    def __str__(self):
        return self.name

# Alias for backward compatibility
PineconeAssistantManager = DeepSeekAssistantManager

# Template processor for DeepSeek assistants
class DeepSeekTemplateProcessor:
    """Process templates for DeepSeek assistants."""
    
    def __init__(self, assistant_manager: DeepSeekAssistantManager):
        self.assistant_manager = assistant_manager
    
    def process_resolve_request(
        self, 
        assistant_id: str, 
        gedanke_in_weltanschauung: str,
        aspekte: str = None
    ) -> Dict[str, Any]:
        """Process a resolve request using DeepSeek assistant."""
        
        # Build the template prompt (same as your current Pinecone template)
        prompt = f"""Korrigiere folgenden kulturgewordenen Gedankenfehler:

** {gedanke_in_weltanschauung} **

{f"Berücksichtige dabei: {aspekte}" if aspekte else ""}

Bitte antworte mit einem **gültigen und kommentarlosen JSON-Objekt** im folgenden Format:

{{
    "gedanke": "300-Wort-Korrektur aus deiner philosophischen Perspektive",
    "gedanke_zusammenfassung": "Kurze Zusammenfassung in 30-35 Worten",
    "gedanke_kind": "Kinderfreundliche Erklärung für 10-Jährige"
}}

Stelle sicher, dass wirklich nur der Text des JSON-Objekts zurückgegeben wird."""
        
        # Query the DeepSeek assistant
        response = self.assistant_manager.query_assistant_with_knowledge(
            assistant_id=assistant_id,
            user_message=prompt,
            use_knowledge_base=True
        )
        
        # Parse JSON response
        try:
            import re
            json_match = re.search(r'\{.*\}', response["message"], re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    **parsed,
                    "assistant_id": assistant_id,
                    "weltanschauung": response["worldview"],
                    "processing_time": response["processing_time"],
                    "cost": response["usage"]["cost"]
                }
        except Exception as e:
            logger.error(f"Error parsing JSON response: {e}")
        
        # Fallback if JSON parsing fails
        return {
            "gedanke": response["message"],
            "gedanke_zusammenfassung": "Philosophische Korrektur bereitgestellt",
            "gedanke_kind": "Eine neue Art, über das Thema nachzudenken",
            "assistant_id": assistant_id,
            "weltanschauung": response["worldview"],
            "processing_time": response["processing_time"],
            "cost": response["usage"]["cost"]
        }

# Example usage
async def main():
    """Example usage of DeepSeek assistant manager."""
    try:
        # Initialize DeepSeek manager
        manager = DeepSeekAssistantManager()
        
        # Create all philosophical assistants
        assistants = manager.create_all_philosophical_assistants()
        
        print("Created DeepSeek assistants:")
        for worldview, config in assistants.items():
            if "error" not in config:
                print(f"✅ {worldview}: {config['id']}")
            else:
                print(f"❌ {worldview}: {config['error']}")
        
        # Show cost analysis
        costs = manager.get_cost_analysis()
        print(f"\nCost analysis for {costs['assistant_count']} assistants:")
        print(f"Pinecone yearly (low usage): ${costs['projections']['low_usage']['pinecone_yearly']:,.2f}")
        print(f"DeepSeek yearly (low usage): ${costs['projections']['low_usage']['hybrid_yearly']:,.2f}")
        print(f"Annual savings (low usage): ${costs['projections']['low_usage']['savings_yearly']:,.2f}")
        
        print(f"\nPinecone yearly (high usage): ${costs['projections']['high_usage']['pinecone_yearly']:,.2f}")
        print(f"DeepSeek yearly (high usage): ${costs['projections']['high_usage']['hybrid_yearly']:,.2f}")
        print(f"Annual savings (high usage): ${costs['projections']['high_usage']['savings_yearly']:,.2f}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 