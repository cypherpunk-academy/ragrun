#!/usr/bin/env python3
"""
Benchmark script for the Personal Embeddings Service.
Measures performance across different scenarios.
"""

import asyncio
import httpx
import numpy as np
import time
import sys
import json
import statistics
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

async def benchmark_single_embeddings(base_url: str, num_iterations: int = 20) -> Dict[str, Any]:
    """
    Benchmark single text embedding performance.
    
    Args:
        base_url: Base URL of the embedding service
        num_iterations: Number of iterations to run
        
    Returns:
        Performance statistics
    """
    print(f"🔍 Benchmarking single embeddings ({num_iterations} iterations)")
    
    test_text = "This is a test sentence for performance benchmarking of single text embedding generation."
    times = []
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Warmup
        await client.post(f"{base_url}/api/v1/embeddings", json={"texts": test_text})
        
        for i in range(num_iterations):
            start_time = time.time()
            response = await client.post(
                f"{base_url}/api/v1/embeddings",
                json={"texts": test_text}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
            else:
                print(f"❌ Request {i+1} failed with status {response.status_code}")
    
    if times:
        return {
            "type": "single_embedding",
            "iterations": len(times),
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "min_time": min(times),
            "max_time": max(times),
            "times": times
        }
    else:
        return {"type": "single_embedding", "error": "No successful requests"}

async def benchmark_batch_sizes(base_url: str, batch_sizes: List[int] = None) -> List[Dict[str, Any]]:
    """
    Benchmark different batch sizes.
    
    Args:
        base_url: Base URL of the embedding service
        batch_sizes: List of batch sizes to test
        
    Returns:
        List of performance results for each batch size
    """
    if batch_sizes is None:
        batch_sizes = [1, 5, 10, 20, 50, 100]
    
    print(f"📊 Benchmarking batch sizes: {batch_sizes}")
    
    results = []
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for batch_size in batch_sizes:
            print(f"   Testing batch size: {batch_size}")
            
            # Generate test texts
            test_texts = [f"Test sentence number {i} for batch size {batch_size} benchmarking." for i in range(batch_size)]
            
            # Run multiple iterations
            times = []
            for _ in range(3):  # 3 iterations per batch size
                start_time = time.time()
                response = await client.post(
                    f"{base_url}/api/v1/embeddings",
                    json={"texts": test_texts}
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
            
            if times:
                avg_time = statistics.mean(times)
                throughput = batch_size / avg_time
                
                results.append({
                    "batch_size": batch_size,
                    "avg_time": avg_time,
                    "throughput": throughput,
                    "time_per_text": avg_time / batch_size,
                    "times": times
                })
    
    return results

async def benchmark_large_batch_processing(base_url: str, total_texts: int = 1000) -> Dict[str, Any]:
    """
    Benchmark large batch processing with chunking.
    
    Args:
        base_url: Base URL of the embedding service
        total_texts: Total number of texts to process
        
    Returns:
        Performance statistics
    """
    print(f"🚀 Benchmarking large batch processing ({total_texts} texts)")
    
    # Generate test texts
    test_texts = [f"Large batch test sentence {i} with additional content for realistic benchmarking." for i in range(total_texts)]
    
    chunk_sizes = [16, 32, 64, 128]
    results = []
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        for chunk_size in chunk_sizes:
            print(f"   Testing chunk size: {chunk_size}")
            
            start_time = time.time()
            response = await client.post(
                f"{base_url}/api/v1/embeddings/batch",
                json={"texts": test_texts, "chunk_size": chunk_size}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                total_time = end_time - start_time
                result_data = response.json()
                
                results.append({
                    "chunk_size": chunk_size,
                    "total_time": total_time,
                    "processing_time": result_data.get("processing_time", 0),
                    "throughput": total_texts / total_time,
                    "time_per_text": total_time / total_texts
                })
    
    return {
        "type": "large_batch",
        "total_texts": total_texts,
        "results": results
    }

async def benchmark_similarity_search(base_url: str) -> Dict[str, Any]:
    """
    Benchmark similarity search performance.
    
    Args:
        base_url: Base URL of the embedding service
        
    Returns:
        Performance statistics
    """
    print("🔍 Benchmarking similarity search")
    
    query = "machine learning algorithms for natural language processing"
    
    # Test different document set sizes
    base_docs = [
        "Deep learning models excel at natural language understanding tasks.",
        "Traditional machine learning uses statistical methods for classification.",
        "Computer vision algorithms process and analyze digital images.",
        "Text processing involves tokenization, parsing, and semantic analysis.",
        "Database systems efficiently store and retrieve structured data.",
        "Web development frameworks simplify building online applications.",
        "Neural networks learn complex patterns from training data.",
        "Artificial intelligence transforms various industries and applications.",
        "Data science combines statistics, programming, and domain expertise.",
        "Cloud computing provides scalable infrastructure for modern applications."
    ]
    
    document_counts = [10, 50, 100, 200]
    results = []
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for doc_count in document_counts:
            # Create document set by repeating and modifying base docs
            documents = []
            for i in range(doc_count):
                base_doc = base_docs[i % len(base_docs)]
                documents.append(f"{base_doc} (Document {i+1})")
            
            print(f"   Testing with {doc_count} documents")
            
            # Run multiple iterations
            times = []
            for _ in range(3):
                start_time = time.time()
                response = await client.post(
                    f"{base_url}/api/v1/search",
                    json={
                        "query": query,
                        "documents": documents,
                        "top_k": 5
                    }
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
            
            if times:
                results.append({
                    "document_count": doc_count,
                    "avg_time": statistics.mean(times),
                    "times": times
                })
    
    return {
        "type": "similarity_search",
        "results": results
    }

def generate_report(benchmark_results: Dict[str, Any]) -> str:
    """
    Generate a formatted benchmark report.
    
    Args:
        benchmark_results: Dictionary containing all benchmark results
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("Personal Embeddings Service - Performance Benchmark Report")
    report.append("=" * 60)
    
    # Single embedding results
    if "single_embedding" in benchmark_results:
        single_results = benchmark_results["single_embedding"]
        report.append("\n📝 Single Text Embedding Performance:")
        report.append(f"   Iterations: {single_results.get('iterations', 0)}")
        report.append(f"   Mean time: {single_results.get('mean_time', 0):.3f}s")
        report.append(f"   Median time: {single_results.get('median_time', 0):.3f}s")
        report.append(f"   Std deviation: {single_results.get('std_time', 0):.3f}s")
        report.append(f"   Min time: {single_results.get('min_time', 0):.3f}s")
        report.append(f"   Max time: {single_results.get('max_time', 0):.3f}s")
    
    # Batch size results
    if "batch_sizes" in benchmark_results:
        batch_results = benchmark_results["batch_sizes"]
        report.append("\n📊 Batch Size Performance:")
        report.append(f"{'Batch Size':<12} {'Avg Time':<10} {'Throughput':<12} {'Time/Text':<10}")
        report.append("-" * 50)
        for result in batch_results:
            report.append(f"{result['batch_size']:<12} {result['avg_time']:<10.3f} {result['throughput']:<12.1f} {result['time_per_text']:<10.4f}")
    
    # Large batch results
    if "large_batch" in benchmark_results:
        large_results = benchmark_results["large_batch"]
        report.append(f"\n🚀 Large Batch Processing ({large_results['total_texts']} texts):")
        report.append(f"{'Chunk Size':<12} {'Total Time':<12} {'Throughput':<12} {'Time/Text':<10}")
        report.append("-" * 50)
        for result in large_results["results"]:
            report.append(f"{result['chunk_size']:<12} {result['total_time']:<12.3f} {result['throughput']:<12.1f} {result['time_per_text']:<10.4f}")
    
    # Similarity search results
    if "similarity_search" in benchmark_results:
        search_results = benchmark_results["similarity_search"]
        report.append("\n🔍 Similarity Search Performance:")
        report.append(f"{'Doc Count':<12} {'Avg Time':<10}")
        report.append("-" * 25)
        for result in search_results["results"]:
            report.append(f"{result['document_count']:<12} {result['avg_time']:<10.3f}")
    
    report.append("\n" + "=" * 60)
    return "\n".join(report)

async def run_comprehensive_benchmark(base_url: str = "http://localhost:8001") -> Dict[str, Any]:
    """
    Run comprehensive benchmark suite.
    
    Args:
        base_url: Base URL of the embedding service
        
    Returns:
        Complete benchmark results
    """
    print("🏁 Starting Comprehensive Benchmark Suite")
    print(f"🌐 Service URL: {base_url}")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Single embedding performance
    try:
        results["single_embedding"] = await benchmark_single_embeddings(base_url)
    except Exception as e:
        print(f"❌ Single embedding benchmark failed: {e}")
    
    # Test 2: Batch size performance
    try:
        results["batch_sizes"] = await benchmark_batch_sizes(base_url)
    except Exception as e:
        print(f"❌ Batch size benchmark failed: {e}")
    
    # Test 3: Large batch processing
    try:
        results["large_batch"] = await benchmark_large_batch_processing(base_url)
    except Exception as e:
        print(f"❌ Large batch benchmark failed: {e}")
    
    # Test 4: Similarity search
    try:
        results["similarity_search"] = await benchmark_similarity_search(base_url)
    except Exception as e:
        print(f"❌ Similarity search benchmark failed: {e}")
    
    return results

def main():
    """Main function to run benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Personal Embeddings Service")
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://localhost:8001",
        help="Base URL of the embedding service"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for benchmark results (JSON format)"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Output file for benchmark report (text format)"
    )
    
    args = parser.parse_args()
    
    async def run_benchmarks():
        try:
            # Run all benchmarks
            results = await run_comprehensive_benchmark(args.url)
            
            # Generate and display report
            report = generate_report(results)
            print(report)
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n💾 Results saved to: {args.output}")
            
            if args.report:
                with open(args.report, 'w') as f:
                    f.write(report)
                print(f"📄 Report saved to: {args.report}")
            
            return True
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return False
    
    try:
        success = asyncio.run(run_benchmarks())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 