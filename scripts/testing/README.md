# Testing Scripts

This directory contains various test scripts for validating the RAG system's functionality, performance, and correctness.

## Contents

-   `run_test.py` - Basic test for the RAG system
-   `run_test_deutsch.py` - Test for German language support
-   `run_test_mit_log.py` - Test with detailed logging
-   `run_test_mit_metadata.py` - Test for metadata handling
-   `run_test_without_filter.py` - Test search without category filters
-   `run_test_without_filter_modified.py` - Modified version of the unfiltered search test
-   `run_realismus_tests.py` - Tests for the "Realismus" category
-   `test_embedding_performance.py` - Performance tests for embedding generation
-   `test_new_model.py` - Tests for new embedding models
-   `test_query_variations.py` - Tests for different query formulations
-   `test_rag_pipeline.py` - End-to-end tests for the RAG pipeline
-   `integration_tests.py` - Integration tests for DeepSeek API connectivity and model selection

## Usage

These scripts can be run directly from the command line:

```bash
python -m scripts.testing.run_test_without_filter
```

Some scripts accept command line arguments:

```bash
python -m scripts.testing.test_query_variations --query "your query here" --top-k 10
```

The integration tests can be run with various options:

```bash
python -m scripts.testing.integration_tests --verbose
python -m scripts.testing.integration_tests --query "What is the meaning of life?"
python -m scripts.testing.integration_tests --test-philosophical "Does free will exist?"
```
