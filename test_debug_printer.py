#!/usr/bin/env python3
"""
Test script to demonstrate the debug printer functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tools.debug_utils import debug_printer

def test_debug_printer():
    """Test the debug printer with different configurations."""
    
    print("=== Testing Debug Printer ===\n")
    
    # Test 1: Print only after epoch 2, last 3 samples
    print("1. Configure: start_epoch=2, last_n_samples=3, enabled=True")
    debug_printer.configure(start_epoch=2, last_n_samples=3, enabled=True)
    
    # Simulate training for 4 epochs with 5 samples each
    for epoch in range(4):
        debug_printer.set_epoch(epoch)
        debug_printer.set_total_samples(5)
        print(f"\n--- Epoch {epoch} ---")
        
        for sample in range(5):
            debug_printer.set_sample(sample)
            debug_printer.print(f"Processing sample data - some stats here")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Disable debug printing
    print("2. Configure: enabled=False")
    debug_printer.configure(enabled=False)
    
    debug_printer.set_epoch(5)
    debug_printer.set_total_samples(3)
    print(f"\n--- Epoch 5 (Debug Disabled) ---")
    
    for sample in range(3):
        debug_printer.set_sample(sample)
        debug_printer.print("This should not print")
    
    print("No debug output should appear above this line.")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Print from beginning, last 2 samples only
    print("3. Configure: start_epoch=0, last_n_samples=2, enabled=True")
    debug_printer.configure(start_epoch=0, last_n_samples=2, enabled=True)
    
    debug_printer.set_epoch(0)
    debug_printer.set_total_samples(4)
    print(f"\n--- Epoch 0 (Last 2 samples only) ---")
    
    for sample in range(4):
        debug_printer.set_sample(sample)
        debug_printer.print(f"Sample {sample} debug info")

if __name__ == "__main__":
    test_debug_printer()
