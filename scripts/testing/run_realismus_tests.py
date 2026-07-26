#!/usr/bin/env python3
"""
Convenient test runner for the Realismus category query tests.
Runs both service-level and API-level tests for "Was ist Moralische Fantasie?" with category "Realismus".
"""
import sys
import logging
from pathlib import Path

# Add the tests directory to the path
sys.path.append(str(Path(__file__).parent / "tests"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main test runner function."""
    logger.info("🧪 Starting Realismus Query Tests")
    logger.info("=" * 60)
    
    all_tests_passed = True
    
    try:
        # Run service-level tests
        logger.info("\n📚 Running Service-Level Tests...")
        logger.info("-" * 40)
        
        from test_realismus_query import run_standalone_test as run_service_test
        service_success = run_service_test()
        
        if service_success:
            logger.info("✅ Service-level tests PASSED")
        else:
            logger.error("❌ Service-level tests FAILED")
            all_tests_passed = False
        
        # Run API-level tests
        logger.info("\n🌐 Running API-Level Tests...")
        logger.info("-" * 40)
        
        from test_realismus_api import run_standalone_api_test as run_api_test
        api_success = run_api_test()
        
        if api_success:
            logger.info("✅ API-level tests PASSED")
        else:
            logger.error("❌ API-level tests FAILED")
            all_tests_passed = False
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure you're running this from the project root directory")
        all_tests_passed = False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        all_tests_passed = False
    
    # Final summary
    logger.info("\n" + "=" * 60)
    if all_tests_passed:
        logger.info("🎉 ALL TESTS PASSED! Your Realismus query is working correctly.")
        logger.info("The RAG system successfully filters by category 'Realismus' and")
        logger.info("responds to the query 'Was ist Moralische Fantasie?'")
    else:
        logger.error("💥 SOME TESTS FAILED! Check the logs above for details.")
    
    logger.info("=" * 60)
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 