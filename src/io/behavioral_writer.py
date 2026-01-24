"""Write behavioral data to CSV files"""

import csv
from pathlib import Path
from typing import List


class BehavioralWriter:
    """Writes annotated behavioral data to CSV"""
    
    def write_annotated_csv(self, output_path: Path, rows: List[list]):
        """
        Write annotated behavioral data with query IDs.
        
        Args:
            output_path: Path for output CSV
            rows: List of rows (first row is header)
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(rows)
                
        except Exception as e:
            print(f"Error writing {output_path}: {e}")
