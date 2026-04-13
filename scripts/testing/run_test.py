#!/usr/bin/env python3
"""
Simple wrapper to run tests with the correct Python path
"""
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath("."))

# Import and run the test
from tests.test_realismus_query import run_standalone_test

if __name__ == "__main__":
    success = run_standalone_test()
    exit(0 if success else 1) 