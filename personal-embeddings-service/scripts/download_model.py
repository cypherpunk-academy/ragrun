#!/usr/bin/env python3
"""
Script to pre-download the multilingual-e5-large model.
This can be run to cache the model before starting the service.
"""

import os
import sys
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name: str = "intfloat/multilingual-e5-large", cache_dir: str = None):
    """
    Download and cache the embedding model.
    
    Args:
        model_name: Name of the model to download
        cache_dir: Directory to cache the model
    """
    try:
        logger.info(f"Downloading model: {model_name}")
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
        else:
            model = SentenceTransformer(model_name)
        
        logger.info(f"Model downloaded successfully!")
        logger.info(f"Model info:")
        logger.info(f"  - Max sequence length: {model.max_seq_length}")
        logger.info(f"  - Embedding dimension: {model.get_sentence_embedding_dimension()}")
        
        # Test the model with a simple sentence
        test_text = "This is a test sentence."
        logger.info("Testing model with sample text...")
        embedding = model.encode(test_text)
        logger.info(f"Test embedding shape: {embedding.shape}")
        logger.info("Model is ready to use!")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download multilingual-e5-large model")
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default="/app/models",
        help="Directory to cache the model (default: /app/models)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="Model name to download (default: intfloat/multilingual-e5-large)"
    )
    
    args = parser.parse_args()
    
    success = download_model(args.model_name, args.cache_dir)
    sys.exit(0 if success else 1) 