#!/usr/bin/env python3
"""
Pipeline: Analyze Worker Quality

Analyzes worker data quality based on behavioral metrics and generates
quality reports.

Usage:
    python src/pipelines/analyze_quality.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.constants import DATA_DIR, OUTPUT_DIR
from src.quality.worker_analyzer import WorkerAnalyzer


def main():
    """Run the quality analysis pipeline"""
    print("=" * 60)
    print("PIPELINE: Analyze Worker Quality")
    print("=" * 60)
    print()
    
    print(f"Data directory:   {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Run analysis
    analyzer = WorkerAnalyzer(DATA_DIR, OUTPUT_DIR)
    analyzer.analyze_all_workers()
    
    print("\n✓ Pipeline completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())
