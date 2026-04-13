#!/usr/bin/env python3
"""
Evaluate the RAG server with philosophical questions using DeepSeek.
Saves results to a YAML file in the results folder.
"""

import os
import sys
import yaml
import json
import time
import requests
from datetime import datetime
from pathlib import Path

# Configuration
RAG_SERVER_URL = "http://localhost:8000/api/v1/rag/query"
RESULTS_DIR = Path("results")

# Philosophical questions to evaluate
QUESTIONS = [
    "Fasse die Kernpunkte der sozialen Frage in 200 Worten zusammen",
    "Wo liegt der Ausgangspunkt der Wissenschaft, der tiefste Punkt, bei Rudolf Steiner und im Materialismus?",
    "Was charakterisiert Rudolf Steiner als das Besondere an Friedrich Nietzsche?",
    "Was ist laut Steiner bei Kindern in der 5. Schulklasse besonders zu beachten? Ist das nach dem Rubikon?",
    "Warum eignet sich die Geistesart Goethes als Grundlage für eine Wissenschaft der geistigen Welt?"
]

def ensure_results_dir():
    """Ensure the results directory exists."""
    RESULTS_DIR.mkdir(exist_ok=True)

def generate_output_filename():
    """Generate the output filename based on the current date."""
    today = datetime.now().strftime("%Y-%m-%d")
    return RESULTS_DIR / f"{today}-eval-philosophy-001.yaml"

def query_rag_server(question):
    """
    Query the RAG server with a question.
    
    Args:
        question: The question to ask
        
    Returns:
        dict: The response from the RAG server
    """
    print(f"Querying RAG server: {question}")
    
    # Prepare the request payload
    payload = {
        "messages": [
            {"role": "user", "content": question}
        ],
        "system_prompt": "Du bist ein philosophischer Assistent mit Expertise in anthroposophischer Philosophie und Rudolf Steiner. Beantworte die Frage basierend auf den abgerufenen Dokumenten.",
        "top_k": 5
    }
    
    try:
        response = requests.post(
            RAG_SERVER_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return {
                "error": f"Status code: {response.status_code}",
                "response": response.text
            }
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return {"error": str(e)}

def evaluate_questions():
    """
    Evaluate all philosophical questions and save results to a YAML file.
    
    Returns:
        Path: The path to the output file
    """
    results = []
    
    for i, question in enumerate(QUESTIONS):
        print(f"\nProcessing question {i+1}/{len(QUESTIONS)}")
        
        # Query the RAG server
        start_time = time.time()
        response = query_rag_server(question)
        end_time = time.time()
        
        # Extract the results
        result = {
            "question": question,
            "response": response.get("content", "No content returned"),
            "model": response.get("model", "Unknown"),
            "retrieved_documents": response.get("retrieved_documents", []),
            "processing_time": round(end_time - start_time, 2)
        }
        
        results.append(result)
        
        # Add a small delay between requests to avoid rate limiting
        if i < len(QUESTIONS) - 1:
            time.sleep(1)
    
    # Create the final results structure
    evaluation_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "questions_count": len(QUESTIONS),
            "rag_server": RAG_SERVER_URL
        },
        "results": results
    }
    
    # Save to YAML file
    output_file = generate_output_filename()
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(evaluation_results, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"\nResults saved to {output_file}")
    return output_file

def main():
    """Main function."""
    print("RAG Server Philosophical Evaluation")
    print("==================================")
    
    # Ensure the results directory exists
    ensure_results_dir()
    
    # Check if the RAG server is running
    try:
        response = requests.get(RAG_SERVER_URL.replace("/rag/query", "/health"))
        if response.status_code != 200:
            print(f"Warning: RAG server health check failed with status code {response.status_code}")
            proceed = input("Do you want to proceed anyway? (y/n): ")
            if proceed.lower() != 'y':
                print("Exiting...")
                sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to RAG server at {RAG_SERVER_URL}")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
    
    # Evaluate the questions
    output_file = evaluate_questions()
    
    print("\nEvaluation completed successfully!")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main() 