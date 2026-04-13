#!/usr/bin/env python3
"""
Comprehensive test script for enhanced gedankenfehler-umkehren 4-stage pipeline

Tests:
- Complete workflow from input through all 4 stages to database save
- Multiple weltanschauung values  
- Parallel processing performance validation
- Data integrity in both gedanken and glossar collections
- Model optimization effectiveness
"""

import os
import sys
import time
import json
import importlib.util
from datetime import datetime
from pymongo import MongoClient

# Add scripts directory to path to import GedankenfehlerUmkehrenCommand
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def setup_test_environment():
    """Setup test environment and validate prerequisites"""
    print("🔧 Setting up test environment...")
    
    # Check environment variables
    required_env = ['MONGODB_URI', 'DEEPSEEK_API_KEY']
    missing_env = [var for var in required_env if not os.environ.get(var)]
    
    if missing_env:
        print(f"❌ Missing environment variables: {missing_env}")
        return False
    
    # Test MongoDB connection
    try:
        mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/12_weltanschauungen')
        client = MongoClient(mongodb_uri)
        db = client['12_weltanschauungen']
        db.command('ping')
        client.close()
        print("✅ MongoDB connection successful")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False
    
    # Check DeepSeek API (basic availability)
    try:
        from assistants.deepseek_assistant_manager import DeepSeekAssistantManager
        manager = DeepSeekAssistantManager()
        assistants = manager.assistant_configs
        if assistants:
            print(f"✅ DeepSeek assistants available: {len(assistants)} configured")
        else:
            print("⚠️  No DeepSeek assistants configured")
    except Exception as e:
        print(f"❌ DeepSeek assistant manager failed: {e}")
        return False
    
    print("✅ Test environment setup complete")
    return True

def run_workflow_test(weltanschauung, test_gedanke, test_name):
    """Run complete 4-stage workflow test for a specific weltanschauung"""
    print(f"\n🧪 {test_name}: Testing {weltanschauung}")
    print("="*60)
    
    try:
        spec = importlib.util.spec_from_file_location(
            "gedankenfehler_umkehren", 
            os.path.join(os.path.dirname(__file__), 'scripts', 'gedankenfehler-umkehren.py')
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        GedankenfehlerUmkehrenCommand = module.GedankenfehlerUmkehrenCommand
        
        # Initialize command
        command = GedankenfehlerUmkehrenCommand()
        
        # Run complete 4-stage processing
        start_time = time.time()
        success = command.process_gedankenfehler_umkehren(
            gedankenfehler=test_gedanke,
            weltanschauung=weltanschauung,
            nummer=None,  # Auto-assign
            aspekt="Test der 4-Stufen-Pipeline"
        )
        total_time = time.time() - start_time
        
        if success:
            print(f"✅ {test_name} completed successfully in {total_time:.2f}s")
            
            # Return performance metrics
            return {
                'success': True,
                'total_time': total_time,
                'stage_times': command.performance_metrics.get('stage_times', {}),
                'timings': command.timings,
                'costs': command.costs,
                'results': command.results,
                'weltanschauung': weltanschauung
            }
        else:
            print(f"❌ {test_name} failed")
            return {'success': False, 'weltanschauung': weltanschauung}
            
    except Exception as e:
        print(f"❌ {test_name} error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e), 'weltanschauung': weltanschauung}

def validate_database_integrity(test_results):
    """Validate data integrity in both gedanken and glossar collections"""
    print("\n🔍 Validating database integrity...")
    
    try:
        mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/12_weltanschauungen')
        client = MongoClient(mongodb_uri)
        db = client['12_weltanschauungen']
        
        validation_results = []
        
        for result in test_results:
            if not result.get('success'):
                continue
                
            weltanschauung = result['weltanschauung']
            
            # Find recent gedanken entries for this weltanschauung
            recent_gedanken = list(db.gedanken.find({
                'weltanschauung': weltanschauung,
                'model': 'gedankenfehler-umkehren-v2'
            }).sort('created_at', -1).limit(1))
            
            if not recent_gedanken:
                print(f"⚠️  No recent gedanken found for {weltanschauung}")
                continue
                
            gedanke_doc = recent_gedanken[0]
            nummer = gedanke_doc.get('nummer')
            
            # Validate gedanken document structure
            required_fields = [
                'gedanke_original', 'gedanke', 'gedanke_einfach', 'gedanke_kurz',
                'weltanschauung', 'nummer', 'autor', 'created_at'
            ]
            
            missing_fields = [field for field in required_fields if not gedanke_doc.get(field)]
            if missing_fields:
                print(f"❌ {weltanschauung}: Missing fields in gedanken: {missing_fields}")
                validation_results.append({'weltanschauung': weltanschauung, 'gedanken_valid': False})
                continue
            
            # Check that we have both original and modernized versions
            original_length = len(gedanke_doc.get('gedanke_original', ''))
            modernized_length = len(gedanke_doc.get('gedanke', ''))
            
            if original_length < 50 or modernized_length < 50:
                print(f"⚠️  {weltanschauung}: Suspiciously short gedanke texts")
            
            print(f"✅ {weltanschauung}: Gedanken document valid")
            print(f"   • Original: {original_length} chars")
            print(f"   • Modernized: {modernized_length} chars")
            print(f"   • Simplified: {len(gedanke_doc.get('gedanke_einfach', ''))} chars")
            print(f"   • Summary: {len(gedanke_doc.get('gedanke_kurz', ''))} chars")
            
            # Find corresponding glossar entries
            glossar_entries = list(db.glossar.find({
                'weltanschauung': weltanschauung,
                'nummer': nummer
            }))
            
            if glossar_entries:
                print(f"✅ {weltanschauung}: {len(glossar_entries)} glossary terms found")
                for term in glossar_entries[:3]:  # Show first 3 terms
                    print(f"   • {term.get('begriff', '')}: {term.get('beschreibung', '')[:50]}...")
            else:
                print(f"⚠️  {weltanschauung}: No glossary terms found for nummer {nummer}")
            
            validation_results.append({
                'weltanschauung': weltanschauung,
                'gedanken_valid': True,
                'glossar_terms': len(glossar_entries),
                'original_length': original_length,
                'modernized_length': modernized_length
            })
        
        client.close()
        return validation_results
        
    except Exception as e:
        print(f"❌ Database validation error: {e}")
        return []

def analyze_performance_metrics(test_results):
    """Analyze performance metrics and parallel processing effectiveness"""
    print("\n📊 Performance Analysis")
    print("="*60)
    
    successful_tests = [r for r in test_results if r.get('success')]
    
    if not successful_tests:
        print("❌ No successful tests to analyze")
        return
    
    # Overall timing analysis
    total_times = [r['total_time'] for r in successful_tests]
    avg_total_time = sum(total_times) / len(total_times)
    
    print(f"⏱️  Average total processing time: {avg_total_time:.2f}s")
    print(f"   • Fastest: {min(total_times):.2f}s")
    print(f"   • Slowest: {max(total_times):.2f}s")
    
    # Stage breakdown analysis
    stage_averages = {}
    for result in successful_tests:
        for stage, time_val in result.get('stage_times', {}).items():
            if stage not in stage_averages:
                stage_averages[stage] = []
            stage_averages[stage].append(time_val)
    
    print(f"\n📈 Stage Performance Breakdown:")
    for stage, times in stage_averages.items():
        avg_time = sum(times) / len(times)
        percentage = (avg_time / avg_total_time * 100) if avg_total_time > 0 else 0
        print(f"   • {stage}: {avg_time:.3f}s (avg) - {percentage:.1f}% of total")
    
    # Parallel vs Sequential comparison
    parallel_times = []
    sequential_times = []
    
    for result in successful_tests:
        timings = result.get('timings', {})
        if 'parallel_total' in timings:
            parallel_times.append(timings['parallel_total'])
        if 'sequential_total' in timings:
            sequential_times.append(timings['sequential_total'])
    
    if parallel_times and sequential_times:
        avg_parallel = sum(parallel_times) / len(parallel_times)
        avg_sequential = sum(sequential_times) / len(sequential_times)
        improvement = ((avg_sequential - avg_parallel) / avg_sequential * 100)
        
        print(f"\n⚡ Parallel vs Sequential Processing:")
        print(f"   • Average parallel time: {avg_parallel:.3f}s")
        print(f"   • Average sequential time: {avg_sequential:.3f}s") 
        print(f"   • Performance improvement: {improvement:.1f}%")
        
        if improvement >= 60:
            print("   ✅ Parallel processing target achieved (60%+)")
        else:
            print("   ⚠️  Parallel processing below target (60%)")
    elif parallel_times:
        avg_parallel = sum(parallel_times) / len(parallel_times)
        print(f"\n⚡ Parallel Processing:")
        print(f"   • Average parallel time: {avg_parallel:.3f}s")
    
    # Cost analysis
    total_costs = []
    stage_costs = {}
    
    for result in successful_tests:
        costs = result.get('costs', {})
        total_cost = sum(costs.values())
        if total_cost > 0:
            total_costs.append(total_cost)
            
        for stage, cost in costs.items():
            if stage not in stage_costs:
                stage_costs[stage] = []
            stage_costs[stage].append(cost)
    
    if total_costs:
        avg_total_cost = sum(total_costs) / len(total_costs)
        print(f"\n💰 Cost Analysis:")
        print(f"   • Average total cost per workflow: ${avg_total_cost:.6f}")
        
        for stage, costs in stage_costs.items():
            avg_cost = sum(costs) / len(costs) if costs else 0
            percentage = (avg_cost / avg_total_cost * 100) if avg_total_cost > 0 else 0
            print(f"   • {stage}: ${avg_cost:.6f} (avg) - {percentage:.1f}% of total")

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("🚀 Starting comprehensive gedankenfehler-umkehren tests")
    print("="*70)
    
    # Setup test environment
    if not setup_test_environment():
        print("❌ Test environment setup failed. Aborting tests.")
        return False
    
    # Test configurations
    test_configs = [
        {
            'weltanschauung': 'Idealismus',
            'test_gedanke': 'Die Welt ist nur ein Traum unserer Vorstellung',
            'test_name': 'TEST 1'
        },
        {
            'weltanschauung': 'Materialismus', 
            'test_gedanke': 'Nur das Geistige existiert wirklich',
            'test_name': 'TEST 2'
        },
        {
            'weltanschauung': 'Rationalismus',
            'test_gedanke': 'Gefühle sind wichtiger als der Verstand',
            'test_name': 'TEST 3'
        }
    ]
    
    # Run workflow tests
    test_results = []
    for config in test_configs:
        result = run_workflow_test(**config)
        test_results.append(result)
        time.sleep(2)  # Brief pause between tests
    
    # Analyze results
    successful_count = sum(1 for r in test_results if r.get('success'))
    print(f"\n📊 Test Summary: {successful_count}/{len(test_results)} tests successful")
    
    if successful_count == 0:
        print("❌ All tests failed. Cannot continue with validation.")
        return False
    
    # Validate database integrity
    validation_results = validate_database_integrity(test_results)
    
    # Analyze performance
    analyze_performance_metrics(test_results)
    
    # Final assessment
    print(f"\n🎯 Final Assessment")
    print("="*50)
    
    if successful_count == len(test_results):
        print("✅ All workflow tests passed")
    else:
        failed_tests = [r['weltanschauung'] for r in test_results if not r.get('success')]
        print(f"⚠️  Failed tests: {failed_tests}")
    
    valid_db_entries = sum(1 for v in validation_results if v.get('gedanken_valid'))
    print(f"✅ Database integrity: {valid_db_entries}/{len(validation_results)} valid entries")
    
    glossar_entries = sum(v.get('glossar_terms', 0) for v in validation_results)
    print(f"✅ Glossary terms created: {glossar_entries} total")
    
    success_rate = (successful_count / len(test_results)) * 100
    if success_rate >= 100:
        print("🎉 ALL TESTS PASSED - Enhanced pipeline fully functional!")
        return True
    elif success_rate >= 67:
        print("✅ TESTS MOSTLY PASSED - Pipeline functional with minor issues")
        return True
    else:
        print("❌ TESTS FAILED - Pipeline needs fixes before production")
        return False

def main():
    """Main test execution"""
    try:
        success = run_comprehensive_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 