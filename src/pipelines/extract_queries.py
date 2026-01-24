#!/usr/bin/env python3
"""
Pipeline: Extract Query Data

Extracts query data from the master LLM logs CSV and creates a structured
JSON file for easier data processing and analysis.

Usage:
    python src/pipelines/extract_queries.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.constants import QUERY_LOGS_FILE, OUTPUT_DIR
from src.processing.query_extractor import QueryExtractor


def main():
    """Run the query extraction pipeline"""
    print("=" * 60)
    print("PIPELINE: Extract Query Data")
    print("=" * 60)
    print()
    
    # Configure paths
    output_json = OUTPUT_DIR / "query_data.json"
    
    print(f"Input:  {QUERY_LOGS_FILE}")
    print(f"Output: {output_json}")
    print()
    
    # Run extraction
    extractor = QueryExtractor(QUERY_LOGS_FILE, output_json)
    success = extractor.extract()
    
    if success:
        print("\n✓ Pipeline completed successfully!")
        print(f"\nNext step: python src/pipelines/match_gaze.py")
    else:
        print("\n✗ Pipeline failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
