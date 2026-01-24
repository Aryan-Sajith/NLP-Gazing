"""Write query data to JSON files"""

import json
from pathlib import Path
from typing import Dict, List
from src.models.query import Query


class QueryWriter:
    """Writes query data to JSON format"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
    
    def write_queries(self, user_data: Dict[str, Dict[str, List[Query]]]) -> bool:
        """
        Write organized query data to JSON file.
        
        Args:
            user_data: {user_id: {task_id: [Query, ...]}}
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert Query objects to dictionaries
            json_data = {}
            for user_id, tasks in user_data.items():
                json_data[user_id] = {}
                for task_id, queries in tasks.items():
                    json_data[user_id][task_id] = [
                        {
                            'query_id': q.query_id,
                            'user_query': q.user_query,
                            'llm_response_1': q.llm_response_1,
                            'llm_response_2': q.llm_response_2,
                            'unix_timestamp': q.unix_timestamp
                        }
                        for q in queries
                    ]
            
            # Ensure output directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"Query data successfully extracted to {self.output_path}")
            self._print_summary(user_data)
            return True
            
        except Exception as e:
            print(f"Error saving JSON file: {e}")
            return False
    
    def _print_summary(self, user_data: Dict[str, Dict[str, List[Query]]]):
        """Print extraction summary"""
        print("\nExtraction Summary:")
        total_entries = 0
        
        for user_id, tasks in user_data.items():
            user_entries = sum(len(queries) for queries in tasks.values())
            total_entries += user_entries
            task_count = len(tasks)
            print(f"  User {user_id}: {task_count} tasks, {user_entries} total entries")
        
        print(f"\nTotal entries processed: {total_entries}")
