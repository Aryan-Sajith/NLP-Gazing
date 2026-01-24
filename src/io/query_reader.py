"""Read query logs from CSV files"""

import csv
from pathlib import Path
from typing import Dict, List
from src.models.query import Query
from src.utils.timestamp import parse_utc_timestamp


class QueryReader:
    """Reads LLM query logs from CSV"""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
    
    def read_all_queries(self) -> Dict[str, Dict[str, List[Query]]]:
        """
        Read all queries organized by user_id and task_id.
        
        Returns:
            {user_id: {task_id: [Query, ...]}}
        """
        user_data = {}
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    query = self._parse_row(row)
                    if query is None:
                        continue
                    
                    user_id = row['user_id']
                    task_id = row['task_id']
                    
                    # Initialize nested structure
                    if user_id not in user_data:
                        user_data[user_id] = {}
                    if task_id not in user_data[user_id]:
                        user_data[user_id][task_id] = []
                    
                    user_data[user_id][task_id].append(query)
            
            # Sort by timestamp within each user-task combination
            for user_id in user_data:
                for task_id in user_data[user_id]:
                    user_data[user_id][task_id].sort(key=lambda q: q.unix_timestamp)
            
            print(f"Loaded data for {len(user_data)} users")
            return user_data
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return {}
    
    def _parse_row(self, row: dict) -> Query:
        """Parse a single CSV row into a Query object"""
        try:
            query_id = int(row['query_ID'])
            user_query = row['user_query']
            llm_response_1 = row['llm_response_1']
            llm_response_2 = row['llm_response_2']
            
            # Handle NULL responses
            if llm_response_2 == 'NULL':
                llm_response_2 = None
            
            # Parse timestamp
            unix_timestamp = parse_utc_timestamp(row['query_timestamp'])
            if unix_timestamp is None:
                return None
            
            return Query(
                query_id=query_id,
                user_query=user_query,
                llm_response_1=llm_response_1,
                llm_response_2=llm_response_2,
                unix_timestamp=unix_timestamp
            )
        except Exception as e:
            print(f"Error parsing row: {e}")
            return None
    
    def read_query_cache(self) -> Dict[int, dict]:
        """Read queries as a flat cache by query_id for feature extraction"""
        cache = {}
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    query_id = int(row['query_ID'])
                    cache[query_id] = row
            
            print(f"Loaded {len(cache)} queries from logs")
            return cache
            
        except Exception as e:
            print(f"Error loading query logs: {e}")
            return {}
