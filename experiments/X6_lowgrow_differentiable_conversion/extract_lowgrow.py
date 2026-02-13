#!/usr/bin/env python3
"""
X6: LowGrow Model Extraction and Analysis
==========================================

This script uses the StellaParser from lib/stella_parser.py to:
1. Extract the complete LowGrow model structure from the .stmx file
2. Generate an inventory of all variables (stocks, flows, auxiliaries)
3. Identify key policy levers and state variables
4. Create a mapping for conversion to differentiable PyTorch implementation
5. Suggest variable β annealing schedule for eliminating oscillations

The LowGrow model from LGSSE (LowGrow-SFC for Simulation Exercises) is a
system dynamics model of macroeconomics with environmental accounting. It exhibits
the oscillatory behavior documented in the research notes—binary policy triggers
causing chattering. This extraction is the first step in converting it to a
differentiable, optimizable form using the variable β framework from X5.
"""

import sys
import os
from pathlib import Path

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))

from stella_parser import StellaParser


def extract_lowgrow_structure():
    """
    Extract and analyze the LowGrow Stella model.

    Steps:
    1. Parse the .stmx file (8.7 MB XML with full model structure)
    2. Load parameter data from the Excel export
    3. Generate structured inventory
    4. Identify conversion strategy
    """

    # Paths
    stmx_path = "_private/LowGrow/LGSSE 29 JAN 2026.stmx"

    # Check if file exists
    if not os.path.exists(stmx_path):
        print(f"ERROR: Could not find LowGrow model at {stmx_path}")
        print("Make sure _private/ symlink points to your OneDrive folder.")
        return None

    print("=" * 80)
    print("X6: LOWGROW MODEL EXTRACTION AND ANALYSIS")
    print("=" * 80)
    print(f"\nExtracting model from: {stmx_path}")
    print(f"File size: {os.path.getsize(stmx_path) / 1024 / 1024:.1f} MB")
    print()

    # Initialize parser
    parser = StellaParser(stmx_path, csv_dir="_private/LowGrow")

    # Parse the model
    try:
        parser.parse_stmx()
    except Exception as e:
        print(f"ERROR parsing model: {e}")
        return None

    # Load CSV data if available
    parser.load_csv_data()

    # Generate detailed inventory
    print("\n" + "=" * 80)
    inventory = parser.generate_inventory()

    # Save inventory to file
    inventory_file = "experiments/X6_lowgrow_differentiable_conversion/LOWGROW_INVENTORY.txt"
    os.makedirs(os.path.dirname(inventory_file), exist_ok=True)
    parser.generate_inventory(output_file=inventory_file)

    # Analyze the structure
    print("\n" + "=" * 80)
    print("CONVERSION STRATEGY ANALYSIS")
    print("=" * 80)

    stocks = {k: v for k, v in parser.variables.items() if v.var_type == 'stock'}
    flows = {k: v for k, v in parser.variables.items() if v.var_type == 'flow'}
    auxes = {k: v for k, v in parser.variables.items() if v.var_type == 'aux'}

    print(f"\nModel Structure:")
    print(f"  Stocks (state variables):     {len(stocks)}")
    print(f"  Flows (rates of change):      {len(flows)}")
    print(f"  Auxiliaries (calculations):   {len(auxes)}")
    print(f"  Total variables:              {len(parser.variables)}")

    # Identify key policy triggers (look for IF/THEN in equations)
    print(f"\nIdentifying Policy Triggers (binary IF/THEN logic):")
    trigger_count = 0
    for name, var in parser.variables.items():
        if 'IF' in var.equation.upper():
            trigger_count += 1
            print(f"  ✓ {var.original_name}")
            print(f"    Equation: {var.equation[:100]}{'...' if len(var.equation) > 100 else ''}")

    if trigger_count == 0:
        print("  (None found with explicit IF statements in equation)")

    # Identify lookup tables (graphical functions)
    print(f"\nIdentifying Lookup Tables (graphical functions):")
    lookup_count = sum(1 for v in parser.variables.values() if v.graphical_function)
    if lookup_count > 0:
        for name, var in parser.variables.items():
            if var.graphical_function:
                gf = var.graphical_function
                print(f"  ✓ {var.original_name}: {gf['n_points']} points")
                print(f"    X range: [{gf['xmin']:.2f}, {gf['xmax']:.2f}]")
    else:
        print("  (None found)")

    # Generate PyTorch stub
    print("\n" + "=" * 80)
    print("GENERATING PYTORCH STUB")
    print("=" * 80)

    stub_file = "experiments/X6_lowgrow_differentiable_conversion/lowgrow_pytorch_skeleton.py"
    stub = parser.get_pytorch_stub(output_file=stub_file)
    print(f"\nPyTorch skeleton saved to: {stub_file}")

    # Summary report
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"""
1. INVENTORY REVIEW
   - Open {inventory_file}
   - Identify key state variables (unemployment, wealth, emissions, debt)
   - Identify policy levers (tax rates, green investment, govt spending)
   - Note any variables with graphical functions (non-linear relationships)

2. EQUATION MAPPING
   - Review each flow equation to understand model dynamics
   - Identify IF/THEN structures causing oscillations (chattering)
   - These are candidates for sigmoid softening with variable β

3. PYTORCH IMPLEMENTATION
   - Use {stub_file} as skeleton
   - Implement equations manually using torch operations
   - Replace hard IF/THEN logic with sigmoid(β * (condition))
   - Apply variable β annealing schedule (see X5 for pattern)

4. VARIABLE β ANNEALING STRATEGY
   - Phase 1: β = 0.5 (very fuzzy, enable broad optimization)
   - Phase 2: β = 5.0 (moderate sigmoid sharpness)
   - Phase 3: β = 20.0 (approaching hard logic while staying smooth)
   - This matches X5 approach and eliminates documented oscillations

5. VALIDATION
   - Train on historical LowGrow baseline runs
   - Compare differentiable version against original Stella output
   - Verify oscillation elimination under annealing schedule
   - Benchmark gradient-based optimization vs manual tuning

Key Insight:
The root cause of LowGrow's oscillatory behavior is binary policy triggers
(unemployment > threshold → activate spending). Variable β provides the
mathematical bridge: soft triggers for optimization, hard triggers for
accounting accuracy. This experiment shows how to apply that bridge to a
real, complex economic model.
""")

    return parser


if __name__ == "__main__":
    parser = extract_lowgrow_structure()

    if parser:
        print("\n✓ Extraction complete. Model is ready for conversion to PyTorch.")
    else:
        print("\n✗ Extraction failed. Check file paths and try again.")
        sys.exit(1)
