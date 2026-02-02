"""Read behavioral data from CSV files"""

import csv
from pathlib import Path
from typing import List
from src.models.behavioral_data import BehavioralDataPoint


class BehavioralReader:
    """Reads gaze and mouse tracking CSV files"""
    
    def read_raw_csv(self, file_path: Path) -> List[list]:
        """
        Read raw CSV rows without parsing.
        
        Returns:
            List of raw rows as lists
        """
        try:
            with open(file_path, encoding="utf-8") as fh:
                return list(csv.reader(fh))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
    
    def read_annotated_csv(self, file_path: Path, query_id: int) -> List[BehavioralDataPoint]:
        """
        Read annotated CSV and filter by query_id.
        
        Args:
            file_path: Path to annotated CSV file
            query_id: Filter for this query ID
        
        Returns:
            List of behavioral data points for the query
        """
        data = []
        
        try:
            if not file_path.exists():
                return data
            
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('query_id') == str(query_id):
                        is_exp_text = row.get('is_experimental_text', '').lower() == 'true'
                        is_not_looking = row.get('is_not_looking', '').lower() == 'true'
                        
                        point = BehavioralDataPoint.from_csv_row(row, query_id, is_exp_text, is_not_looking)
                        if point:
                            data.append(point)
                            
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
        
        return data
    
    def read_full_csv(self, file_path: Path) -> List[BehavioralDataPoint]:
        """
        Read annotated CSV without filtering by query_id (for new timeline-based approach).
        
        Args:
            file_path: Path to annotated CSV file
        
        Returns:
            List of all behavioral data points in file
        """
        data = []
        
        try:
            if not file_path.exists():
                return data
            
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    query_id = int(row.get('query_id', -1))
                    is_exp_text = row.get('is_experimental_text', '').lower() == 'true'
                    is_not_looking = row.get('is_not_looking', '').lower() == 'true'
                    
                    point = BehavioralDataPoint.from_csv_row(row, query_id, is_exp_text, is_not_looking)
                    if point:
                        data.append(point)
                        
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
        
        return data
