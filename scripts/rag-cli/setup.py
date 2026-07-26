#!/usr/bin/env python3
"""
Setup script for the RAG CLI tool.
"""

from setuptools import setup, find_packages

setup(
    name="rag-cli",
    version="0.1.0",
    description="Command Line Interface for RAG (Retrieval-Augmented Generation) Management",
    author="Your Name",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "click",
        "tqdm",
        "pinecone-client",
        "sentence-transformers",
    ],
    entry_points="""
        [console_scripts]
        rag-cli=scripts.rag_cli.main:cli
    """,
) 