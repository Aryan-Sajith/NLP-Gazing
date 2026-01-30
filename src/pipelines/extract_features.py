#!/usr/bin/env python3
"""
Pipeline: Extract Features

Extracts behavioral features from annotated gaze and mouse data for
pairwise and pointwise comparisons.

Usage:
    source venv/bin/activate  # Activate virtual environment first
    python src/pipelines/extract_features.py
    
Or run directly:
    ./venv/bin/python src/pipelines/extract_features.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.constants import DATA_DIR, QUERY_LOGS_FILE, OUTPUT_DIR
from src.processing.feature_extractor import FeatureExtractor


def main():
    """Run the feature extraction pipeline"""
    print("=" * 60)
    print("PIPELINE: Extract Features")
    print("=" * 60)
    print()
    
    # Configure paths
    output_csv = OUTPUT_DIR / "extracted_features.csv"
    
    print(f"Data directory: {DATA_DIR}")
    print(f"Query logs:     {QUERY_LOGS_FILE}")
    print(f"Output:         {output_csv}")
    print()
    
    # Run extraction
    extractor = FeatureExtractor(DATA_DIR, QUERY_LOGS_FILE, output_csv)
    successful = extractor.extract_all_features()
    
    if successful > 0:
        print(f"\n✓ Pipeline completed successfully!")
        print(f"  Extracted features for {successful} comparisons")
    else:
        print("\n✗ Pipeline failed: No features extracted")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
