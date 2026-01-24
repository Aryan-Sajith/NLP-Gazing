"""Extract and organize query data from logs"""

from pathlib import Path
from typing import Dict, List
from src.models.query import Query
from src.io.query_reader import QueryReader
from src.io.query_writer import QueryWriter


class QueryExtractor:
    """Extracts query data from LLM logs and creates structured JSON"""
    
    def __init__(self, csv_path: Path, output_path: Path):
        self.reader = QueryReader(csv_path)
        self.writer = QueryWriter(output_path)
    
    def extract(self) -> bool:
        """
        Extract queries from CSV and write to JSON.
        
        Returns:
            True if successful, False otherwise
        """
        print("Extracting query data from logs...")
        
        # Read all queries
        user_data = self.reader.read_all_queries()
        
        if not user_data:
            print("No query data found")
            return False
        
        # Write to JSON
        success = self.writer.write_queries(user_data)
        
        if success:
            print("\nExtraction completed successfully!")
        else:
            print("\nExtraction failed!")
        
        return success
