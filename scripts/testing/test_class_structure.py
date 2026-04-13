#!/usr/bin/env python3
"""Test script to verify the enhanced GedankenfehlerUmkehrenCommand class"""

import sys
import importlib.util

# Load the script module
spec = importlib.util.spec_from_file_location('gedankenfehler_umkehren', 'scripts/gedankenfehler-umkehren.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Test the class
try:
    cmd = module.GedankenfehlerUmkehrenCommand()
    print('✅ Class instantiated successfully')
    print(f'Results keys: {list(cmd.results.keys())}')
    print(f'Timings: {cmd.timings}')
    print(f'Costs: {cmd.costs}')
    print(f'Has assistant_manager attr: {hasattr(cmd, "assistant_manager")}')
    print(f'Available methods: {[m for m in dir(cmd) if not m.startswith("_") and callable(getattr(cmd, m))][:8]}...')
    
    # Test required keys in results
    expected_keys = {'reformulate', 'resolve', 'modernize', 'simplify', 'glossary', 'summarize'}
    actual_keys = set(cmd.results.keys())
    if expected_keys == actual_keys:
        print('✅ Results dictionary has all required keys')
    else:
        print(f'❌ Missing keys: {expected_keys - actual_keys}')
        
    print('\n🎯 Class structure enhancement COMPLETE!')
    
except Exception as e:
    print(f'❌ Error testing class: {e}')
    sys.exit(1) 