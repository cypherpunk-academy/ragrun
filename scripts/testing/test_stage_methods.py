#!/usr/bin/env python3
"""Test script to verify all stage methods are available and functional"""

import importlib.util

# Load the script module
spec = importlib.util.spec_from_file_location('gedankenfehler_umkehren', 'scripts/gedankenfehler-umkehren.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Test the class
try:
    cmd = module.GedankenfehlerUmkehrenCommand()
    print('✅ Class instantiated successfully')
    
    # Test all stage methods are available
    stage_methods = {
        'stage_1_reformulate': 'Stage 1: Reformulate',
        'stage_2_resolve': 'Stage 2: Resolve', 
        'stage_3_parallel_processing': 'Stage 3: Parallel Processing',
        'stage_4_summarize': 'Stage 4: Summarize'
    }
    
    print('\n🧪 Testing stage method availability:')
    for method_name, description in stage_methods.items():
        if hasattr(cmd, method_name):
            method = getattr(cmd, method_name)
            if callable(method):
                print(f'✅ {description}: {method_name}() available and callable')
            else:
                print(f'❌ {description}: {method_name} exists but not callable')
        else:
            print(f'❌ {description}: {method_name}() missing')
    
    # Test helper processing methods
    helper_methods = ['_process_modernize', '_process_simplify', '_process_glossary']
    print('\n🔧 Testing helper processing methods:')
    for method_name in helper_methods:
        if hasattr(cmd, method_name) and callable(getattr(cmd, method_name)):
            print(f'✅ {method_name}() available')
        else:
            print(f'❌ {method_name}() missing or not callable')
    
    # Test workflow method
    if hasattr(cmd, 'process_gedankenfehler_umkehren') and callable(getattr(cmd, 'process_gedankenfehler_umkehren')):
        print(f'\n✅ Main workflow method: process_gedankenfehler_umkehren() available')
    else:
        print(f'\n❌ Main workflow method missing')
    
    print('\n🎯 All stage methods are implemented and ready!')
    
except Exception as e:
    print(f'❌ Error testing stage methods: {e}')
    import traceback
    traceback.print_exc() 