#!/usr/bin/env python3
"""
Pipeline: Match Gaze to Queries

Annotates behavioral data (gaze and mouse tracking) with query IDs by
matching the text users were looking at with query responses.

Usage:
    python src/pipelines/match_gaze.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.constants import DATA_DIR, OUTPUT_DIR
from src.processing.gaze_matcher import GazeMatcher


def main():
    """Run the gaze matching pipeline"""
    print("=" * 60)
    print("PIPELINE: Match Gaze to Queries")
    print("=" * 60)
    print()
    
    # Configure paths
    query_json = OUTPUT_DIR / "query_data.json"
    
    print(f"Data directory: {DATA_DIR}")
    print(f"Query data:     {query_json}")
    print()
    
    # Check if query data exists
    if not query_json.exists():
        print("✗ Error: query_data.json not found!")
        print("  Run: python src/pipelines/extract_queries.py first")
        return 1
    
    # Run matching
    matcher = GazeMatcher(DATA_DIR, query_json)
    matcher.match_all()
    
    print("\n✓ Pipeline completed successfully!")
    print(f"\nNext step: python src/pipelines/extract_features.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
