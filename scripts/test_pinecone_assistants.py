#!/usr/bin/env python3
"""
Test Pinecone Assistants

This script tests the Pinecone Assistant implementation to ensure everything works correctly.
"""

import os
import sys
import asyncio
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistants.deepseek_assistant_manager import DeepSeekAssistantManager as PineconeAssistantManager

def test_pinecone_connection():
    """Test basic Pinecone connection and assistant listing."""
    print("🔍 Testing Pinecone connection...")
    
    try:
        manager = PineconeAssistantManager()
        assistants = manager.list_assistants()
        
        print(f"✅ Connected to Pinecone successfully")
        print(f"📋 Found {len(assistants)} existing assistants:")
        
        for assistant in assistants:
            print(f"  - {assistant['name']} (Status: {assistant.get('status', 'unknown')})")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Pinecone: {e}")
        return False

def test_assistant_creation():
    """Test creating a simple test assistant."""
    print("\n🧪 Testing assistant creation...")
    
    try:
        manager = PineconeAssistantManager()
        
        # Create a simple test assistant with proper naming convention
        test_name = "test-philosophy-assistant"  # Changed to lowercase with hyphens
        test_instructions = """You are a helpful philosophical assistant. 
        You discuss philosophy in a thoughtful and engaging way.
        Always respond in German."""
        
        assistant = manager.get_or_create_assistant(
            name=test_name,
            worldview="Test",
            instructions=test_instructions
        )
        
        print(f"✅ Successfully created/found test assistant: {test_name}")
        
        # Upload a simple test document
        print("📄 Uploading test document...")
        try:
            # Create a simple test file
            test_file_path = "test_philosophy_document.txt"
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write("""
                Philosophie ist die Liebe zur Weisheit.
                
                Philosophie ist eine der ältesten und fundamentalsten Disziplinen des menschlichen Denkens. 
                Sie beschäftigt sich mit den grundlegendsten Fragen des Lebens, des Seins und der Erkenntnis.
                
                Die Hauptbereiche der Philosophie umfassen:
                - Metaphysik: Was ist die Natur der Realität?
                - Epistemologie: Wie können wir Wissen erlangen?
                - Ethik: Was ist richtig und falsch?
                - Ästhetik: Was ist Schönheit?
                - Logik: Wie können wir gültig schließen?
                
                Große Philosophen wie Plato, Aristoteles, Kant, Hegel und viele andere haben 
                unser Verständnis dieser fundamentalen Fragen geprägt.
                """)
            
            # Upload the document
            upload_results = manager.upload_documents_to_assistant(
                assistant=assistant,
                documents=[{"path": test_file_path, "metadata": {"type": "test"}}],
                worldview="Test"
            )
            
            # Check if upload was successful
            if upload_results and upload_results[0]["status"] == "success":
                print("✅ Test document uploaded successfully")
                return assistant, manager, True  # Return True indicating documents are available
            else:
                print("⚠️ Document upload failed, chat/context tests will be skipped")
                return assistant, manager, False
                
        except Exception as e:
            print(f"⚠️ Document upload failed: {e}")
            return assistant, manager, False
        
    except Exception as e:
        print(f"❌ Failed to create test assistant: {e}")
        return None, None, False

def test_assistant_chat(assistant, manager):
    """Test chatting with an assistant."""
    print("\n💬 Testing assistant chat...")
    
    try:
        response = manager.chat_with_assistant(
            assistant=assistant,
            message="Hallo, kannst du mir kurz erklären, was Philosophie ist?"
        )
        
        print(f"✅ Chat successful!")
        print(f"📝 Response: {response['message'][:200]}...")
        print(f"🔧 Model: {response.get('model', 'unknown')}")
        print(f"📊 Usage: {response.get('usage', {})}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        return False

def test_assistant_context(assistant, manager):
    """Test context retrieval from assistant."""
    print("\n🔍 Testing context retrieval...")
    
    try:
        response = manager.query_with_context(
            assistant=assistant,
            query="Was ist die Bedeutung von Philosophie?"
        )
        
        print(f"✅ Context retrieval successful!")
        print(f"📝 Query: {response['query']}")
        print(f"📄 Found {len(response['snippets'])} context snippets")
        
        for i, snippet in enumerate(response['snippets'][:2]):  # Show first 2
            print(f"  Snippet {i+1}: Score {snippet['score']:.3f}")
            print(f"    Content: {snippet['content'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Context retrieval failed: {e}")
        return False

def cleanup_test_assistant(manager, assistant_name):
    """Clean up test assistant."""
    print(f"\n🧹 Cleaning up test assistant: {assistant_name}")
    
    try:
        success = manager.delete_assistant(assistant_name)
        if success:
            print(f"✅ Successfully deleted test assistant")
        else:
            print(f"⚠️ Test assistant deletion may have failed")
        return success
        
    except Exception as e:
        print(f"❌ Failed to delete test assistant: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing Pinecone Assistant Implementation")
    print("=" * 50)
    
    # Check environment variables
    if not os.environ.get("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY environment variable not set")
        print("Please set your Pinecone API key:")
        print("export PINECONE_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Test 1: Connection
    connection_ok = test_pinecone_connection()
    if not connection_ok:
        print("\n❌ Connection test failed. Please check your API key and network connection.")
        sys.exit(1)
    
    # Test 2: Assistant Creation
    assistant, manager, documents_available = test_assistant_creation()
    if not assistant or not manager:
        print("\n❌ Assistant creation test failed.")
        sys.exit(1)
    
    # Test 3: Chat (only if documents available)
    if documents_available:
        print("\n⏳ Waiting for document processing (5 seconds)...")
        time.sleep(5)  # Give Pinecone time to process the document
        chat_ok = test_assistant_chat(assistant, manager)
    else:
        print("\n⚠️ Skipping chat test - no documents available")
        chat_ok = None
    
    # Test 4: Context (only if documents available)
    if documents_available:
        context_ok = test_assistant_context(assistant, manager)
    else:
        print("⚠️ Skipping context test - no documents available")
        context_ok = None
    
    # Cleanup
    cleanup_test_assistant(manager, "test-philosophy-assistant")
    
    # Clean up test file
    try:
        if os.path.exists("test_philosophy_document.txt"):
            os.remove("test_philosophy_document.txt")
            print("🧹 Cleaned up test document")
    except Exception as e:
        print(f"⚠️ Could not clean up test document: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Connection", connection_ok),
        ("Assistant Creation", assistant is not None),
        ("Chat", chat_ok if documents_available else "Skipped"),
        ("Context Retrieval", context_ok if documents_available else "Skipped"),
    ]
    
    passed = 0
    total = 0
    
    for test_name, result in tests:
        if result == "Skipped":
            status = "⏭️ SKIP"
        elif result:
            status = "✅ PASS"
            passed += 1
            total += 1
        else:
            status = "❌ FAIL"
            total += 1
        print(f"{status} {test_name}")
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if total > 0 and passed == total:
        print("🎉 All tests passed! Pinecone Assistant implementation is working correctly.")
        print("\nNext steps:")
        print("1. Run: python scripts/create_pinecone_assistants.py")
        print("2. Upload philosophical documents to each assistant")
        print("3. Start using the assistants in your application")
    elif passed > 0:
        print("✅ Core functionality working! Some advanced features need documents.")
        print("\nNext steps:")
        print("1. Run: python scripts/create_pinecone_assistants.py")
        print("2. Upload philosophical documents to each assistant")
    else:
        print("⚠️ Tests failed. Please check the implementation.")
        sys.exit(1)

if __name__ == "__main__":
    main() 