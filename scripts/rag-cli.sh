#!/bin/bash
# Simple script to run the RAG CLI from the new location

# Determine the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Run the RAG CLI with the provided arguments
PYTHONPATH="$PROJECT_ROOT" python "$SCRIPT_DIR/rag-cli/main.py" "$@" 