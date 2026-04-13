#!/usr/bin/env python3
"""
Main entry point for the RAG CLI tool.
"""

import os
import sys

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the CLI directly
from rag_cli import cli

if __name__ == '__main__':
    cli() 