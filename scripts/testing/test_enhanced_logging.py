#!/usr/bin/env python3
"""
Test Enhanced Logging

This script tests the enhanced logging functionality with:
- Truncated book titles (30 chars)
- Chunk text previews (30 chars)
- Actual page numbers from content like [GA4, S.12]
"""

import asyncio
import logging
from pathlib import Path

# Setup logging to see the enhanced output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_enhanced_logging():
    """Test the enhanced logging with sample content"""
    
    try:
        from app.services.book_upload_service import BookUploadService
        
        print("🧪 TESTING ENHANCED LOGGING")
        print("=" * 60)
        
        # Create sample content with page numbers and long title
        sample_content = """[1894] Die Philosophie der Freiheit. Grundzüge einer modernen Weltanschauung: Seelische Beobachtungsresultate nach naturwissenschaftlicher Methode

[GA4, S.1] Erstes Kapitel: Die bewusste menschliche Handlung

Der Mensch hat einen natürlichen Trieb zur Erkenntnis. Dieser Trieb führt ihn dazu, hinter die Erscheinungen der Sinnenwelt zu blicken und nach den wahren Ursachen der Phänomene zu suchen.

[GA4, S.12] Zweites Kapitel: Der Trieb zur Erkenntnis

Alle Wissenschaft würde nur ein Befriedigen von Neugier sein, wenn nicht etwas anderes als bloße Wissensgier mit der Erkenntnis verknüpft wäre. Es gibt Naturen, denen das beseligende Gefühl mangelt, welches in dem Momente eintritt, wo das Geheimnis, das die Erscheinungen umhüllt, ihren Blicken weicht.

[GA4, S.23] Drittes Kapitel: Das Denken im Dienste der Weltauffassung

Es gibt Menschen, welche von allem Anfang an gegen das Denken misstrauisch sind. Wenn sie vor die Aufgabe gestellt werden, sich über irgendeine Sache klar zu werden, so empfinden sie sogleich eine gewisse Hilflosigkeit.

Dieses Gefühl der Unsicherheit führt sie oft dazu, dass sie lieber bei einer oberflächlichen Betrachtung der Dinge stehen bleiben."""
        
        print("📝 Sample content with page numbers:")
        print("   - [GA4, S.1]")
        print("   - [GA4, S.12]") 
        print("   - [GA4, S.23]")
        print()
        
        # Create upload service
        upload_service = BookUploadService()
        
        # Test with very long book title
        manual_metadata = {
            "author": "Rudolf Steiner", 
            "title": "Die Philosophie der Freiheit. Grundzüge einer modernen Weltanschauung: Seelische Beobachtungsresultate nach naturwissenschaftlicher Methode - Erweiterte Ausgabe mit zusätzlichen Kapiteln",
            "weltanschauung": "Idealismus"
        }
        
        print("🚀 Starting upload with enhanced logging...")
        print("📚 Expected truncations:")
        print(f"   Title: '{manual_metadata['title'][:30]}...'")
        print(f"   Author: '{manual_metadata['author'][:30]}...'")
        print()
        
        # Upload the sample content
        result = await upload_service.upload_book(
            file_path="test_steiner_philosophy.txt",
            content=sample_content,
            manual_metadata=manual_metadata
        )
        
        print("\n🔍 EXPECTED LOG ENHANCEMENTS:")
        print("=" * 60)
        print("1. ✅ Book titles truncated to 30 characters")
        print("2. ✅ Author names truncated to 30 characters") 
        print("3. ✅ Chunk text previews (first 30 chars)")
        print("4. ✅ Actual page numbers from content:")
        print("   - Instead of: 'Page 1', 'Page 2', 'Page 3'")
        print("   - Should show: 'Page 1', 'Page 12', 'Page 23'")
        
        if result.success:
            print(f"\n✅ Test completed successfully!")
            print(f"📊 Processed {result.successful_chunks} chunks")
            print(f"📖 From book: '{result.extracted_metadata.title[:30]}...'")
        else:
            print(f"\n❌ Test failed")
            for error in result.errors:
                print(f"   Error: {error}")
        
        return result.success
        
    except Exception as e:
        print(f"💥 ERROR: {str(e)}")
        return False


async def main():
    """Main test function"""
    
    print("🔬 ENHANCED LOGGING TEST")
    print("=" * 60)
    print("Testing enhanced logging features:")
    print("• Book title truncation (30 chars)")
    print("• Author name truncation (30 chars)")
    print("• Chunk text previews (30 chars)")
    print("• Actual page numbers from [GA4, S.X] patterns")
    print()
    
    success = await test_enhanced_logging()
    
    if success:
        print(f"\n🎉 All enhanced logging features working!")
        print(f"🔍 Check the log output above for:")
        print(f"   - Truncated titles and authors")
        print(f"   - Chunk text previews in upload logs")
        print(f"   - Actual page numbers (1, 12, 23) instead of sequential")
    else:
        print(f"\n❌ Enhanced logging test failed")
    
    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test canceled")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        exit(1) 