#!/usr/bin/env python3
"""
Test script for the Personal Embeddings Service.
Tests various endpoints and functionality.
"""

import asyncio
import httpx
import numpy as np
import time
import sys
import json
from typing import List, Dict, Any

async def test_embedding_service(base_url: str = "http://localhost:8001"):
    """
    Comprehensive test of the embedding service.
    
    Args:
        base_url: Base URL of the embedding service
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("🚀 Testing Personal Embeddings Service")
        print(f"📍 Base URL: {base_url}")
        print("=" * 50)
        
        # Test 1: Root endpoint
        print("\n1️⃣ Testing root endpoint...")
        try:
            response = await client.get(f"{base_url}/")
            print(f"✅ Root endpoint: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"❌ Root endpoint failed: {e}")
            return False
        
        # Test 2: Health check
        print("\n2️⃣ Testing health check...")
        try:
            response = await client.get(f"{base_url}/api/v1/health")
            health_data = response.json()
            print(f"✅ Health check: {response.status_code}")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Model: {health_data.get('model')}")
            print(f"   Device: {health_data.get('device')}")
            print(f"   Dimensions: {health_data.get('embedding_dimension')}")
            
            if health_data.get('status') != 'healthy':
                print("❌ Service not healthy!")
                return False
                
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
        
        # Test 3: Single text embedding
        print("\n3️⃣ Testing single text embedding...")
        try:
            start_time = time.time()
            response = await client.post(
                f"{base_url}/api/v1/embeddings",
                json={"texts": "This is a test sentence for embedding."}
            )
            processing_time = time.time() - start_time
            
            result = response.json()
            print(f"✅ Single embedding: {response.status_code}")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            print(f"   Total time: {processing_time:.3f}s")
            print(f"   Count: {result['count']}")
            
            # Validate embedding
            embedding = result['embeddings'][0]
            if len(embedding) != 1024:
                print(f"❌ Wrong embedding dimension: {len(embedding)}")
                return False
                
        except Exception as e:
            print(f"❌ Single embedding failed: {e}")
            return False
        
        # Test 4: Batch embeddings
        print("\n4️⃣ Testing batch embeddings...")
        try:
            test_texts = [
                "Machine learning algorithms are powerful tools.",
                "Natural language processing enables text understanding.",
                "Computer vision helps machines see and interpret images.",
                "Deep learning uses neural networks for complex tasks.",
                "Artificial intelligence transforms various industries."
            ]
            
            start_time = time.time()
            response = await client.post(
                f"{base_url}/api/v1/embeddings",
                json={"texts": test_texts}
            )
            processing_time = time.time() - start_time
            
            result = response.json()
            print(f"✅ Batch embeddings: {response.status_code}")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            print(f"   Total time: {processing_time:.3f}s")
            print(f"   Count: {result['count']}")
            print(f"   Texts processed: {len(test_texts)}")
            
            # Validate batch embeddings
            embeddings = result['embeddings']
            if len(embeddings) != len(test_texts):
                print(f"❌ Wrong number of embeddings: {len(embeddings)}")
                return False
                
        except Exception as e:
            print(f"❌ Batch embeddings failed: {e}")
            return False
        
        # Test 5: Large batch processing
        print("\n5️⃣ Testing large batch processing...")
        try:
            large_texts = [f"Test sentence number {i} for large batch processing." for i in range(100)]
            
            start_time = time.time()
            response = await client.post(
                f"{base_url}/api/v1/embeddings/batch",
                json={"texts": large_texts, "chunk_size": 32}
            )
            processing_time = time.time() - start_time
            
            result = response.json()
            print(f"✅ Large batch: {response.status_code}")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            print(f"   Total time: {processing_time:.3f}s")
            print(f"   Count: {result['count']}")
            print(f"   Texts processed: {len(large_texts)}")
            print(f"   Throughput: {len(large_texts)/processing_time:.1f} texts/sec")
            
        except Exception as e:
            print(f"❌ Large batch failed: {e}")
            return False
        
        # Test 6: Similarity search
        print("\n6️⃣ Testing similarity search...")
        try:
            query = "machine learning algorithms for text processing"
            documents = [
                "Deep learning models excel at natural language understanding tasks.",
                "Traditional machine learning uses statistical methods for classification.",
                "Computer vision algorithms process and analyze digital images.",
                "Text processing involves tokenization, parsing, and semantic analysis.",
                "Database systems efficiently store and retrieve structured data.",
                "Web development frameworks simplify building online applications.",
                "Neural networks learn complex patterns from training data."
            ]
            
            start_time = time.time()
            response = await client.post(
                f"{base_url}/api/v1/search",
                json={
                    "query": query,
                    "documents": documents,
                    "top_k": 3
                }
            )
            processing_time = time.time() - start_time
            
            result = response.json()
            print(f"✅ Similarity search: {response.status_code}")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            print(f"   Total time: {processing_time:.3f}s")
            print(f"   Query: {result['query']}")
            print(f"   Results:")
            
            for i, res in enumerate(result['results'], 1):
                print(f"     {i}. Score: {res['score']:.3f} - {res['document'][:60]}...")
            
        except Exception as e:
            print(f"❌ Similarity search failed: {e}")
            return False
        
        # Test 7: Service info
        print("\n7️⃣ Testing service info...")
        try:
            response = await client.get(f"{base_url}/api/v1/info")
            result = response.json()
            print(f"✅ Service info: {response.status_code}")
            print(f"   Ready: {result.get('ready')}")
            print(f"   Max workers: {result.get('max_workers')}")
            
        except Exception as e:
            print(f"❌ Service info failed: {e}")
            return False
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("✨ Personal Embeddings Service is working correctly!")
        return True

async def benchmark_performance(base_url: str = "http://localhost:8001", num_texts: int = 50):
    """
    Benchmark the performance of the embedding service.
    
    Args:
        base_url: Base URL of the embedding service
        num_texts: Number of texts to benchmark with
    """
    print(f"\n🏃 Performance Benchmark ({num_texts} texts)")
    print("=" * 40)
    
    # Generate test texts
    test_texts = [f"Performance test sentence number {i} with some additional content to make it longer and more realistic for benchmarking purposes." for i in range(num_texts)]
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Warm up
        await client.post(
            f"{base_url}/api/v1/embeddings",
            json={"texts": "Warmup text"}
        )
        
        # Benchmark single requests
        print("\n📊 Single request performance:")
        times = []
        for i in range(5):
            start_time = time.time()
            response = await client.post(
                f"{base_url}/api/v1/embeddings",
                json={"texts": test_texts[i]}
            )
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_single = np.mean(times)
        print(f"   Average single embedding time: {avg_single:.3f}s")
        
        # Benchmark batch request
        print(f"\n📊 Batch processing performance ({num_texts} texts):")
        start_time = time.time()
        response = await client.post(
            f"{base_url}/api/v1/embeddings/batch",
            json={"texts": test_texts, "chunk_size": 32}
        )
        end_time = time.time()
        
        batch_time = end_time - start_time
        throughput = num_texts / batch_time
        
        print(f"   Batch processing time: {batch_time:.3f}s")
        print(f"   Throughput: {throughput:.1f} texts/sec")
        print(f"   Time per text: {batch_time/num_texts:.3f}s")

def main():
    """Main function to run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Personal Embeddings Service")
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://localhost:8001",
        help="Base URL of the embedding service"
    )
    parser.add_argument(
        "--benchmark", 
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--benchmark-size",
        type=int,
        default=50,
        help="Number of texts for benchmark"
    )
    
    args = parser.parse_args()
    
    async def run_tests():
        # Run basic functionality tests
        success = await test_embedding_service(args.url)
        
        if success and args.benchmark:
            await benchmark_performance(args.url, args.benchmark_size)
        
        return success
    
    try:
        success = asyncio.run(run_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 