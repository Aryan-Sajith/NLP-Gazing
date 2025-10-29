"""
Step 1: Extract Query Data
Extracts query data from the LLM logs(query logs table) and creates a structured JSON file for easier data processing and analysis.
"""

import csv
import json
from datetime import datetime, timezone


def parse_timestamp(timestamp_str):
    """
    Parse UTC timestamp from format '2025-04-23 13:31:31' to Unix timestamp.
    
    NOTE: The query_timestamp field in full_query_logs_table.csv is in UTC.
    We explicitly mark it as UTC to ensure correct conversion regardless of
    the system timezone where this script runs.
    
    Args:
        timestamp_str: Timestamp string in format 'YYYY-MM-DD HH:MM:SS'
    
    Returns:
        Unix timestamp in milliseconds, or None if parsing fails
    """
    try:
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        dt_utc = dt.replace(tzinfo=timezone.utc)  # Explicitly mark as UTC
        return int(dt_utc.timestamp() * 1000)
    except Exception as e:
        print(f"Error parsing timestamp {timestamp_str}: {e}")
        return None

def extract_query_data(csv_file_path, output_json_path):
    """
    Extract query data from CSV and create structured JSON file.
    Storage format: {
        User ID: {
            Task ID: [
                {
                    'query_id': value,
                    'user_query': value,
                    'llm_response_1': value,
                    'llm_response_2': value,
                    'unix_timestamp': value
                },
                ...
            ]
        }
    }
    
    Args:
        csv_file_path (str): Path to the LLM query logs CSV
        output_json_path (str): Path for output JSON file
    """
    
    user_data = {}
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                user_id = row['user_id']
                task_id = row['task_id']  # Keep as string to match directory names
                query_id = int(row['query_ID'])
                user_query = row['user_query']
                llm_response_1 = row['llm_response_1']
                llm_response_2 = row['llm_response_2']
                if llm_response_2 == 'NULL': # Handles cases where only llm is used for a task
                    llm_response_2 = None
                timestamp_str = row['query_timestamp']
                
                # Parse timestamp
                unix_timestamp = parse_timestamp(timestamp_str)
                if unix_timestamp is None:
                    continue
                
                # Initialize user if not exists
                if user_id not in user_data:
                    user_data[user_id] = {}
                
                # Initialize task if not exists
                if task_id not in user_data[user_id]:
                    user_data[user_id][task_id] = []

                # Create entry as dictionary with labeled keys
                entry = {
                    'query_id': query_id,
                    'user_query': user_query,
                    'llm_response_1': llm_response_1,
                    'llm_response_2': llm_response_2,
                    'unix_timestamp': unix_timestamp
                }
                
                user_data[user_id][task_id].append(entry)
        
        print(f"Loaded data for {len(user_data)} users")
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

    # Sort entries by unix_timestamp for each user-task combination
    for user_id in user_data:
        for task_id in user_data[user_id]:
            user_data[user_id][task_id].sort(key=lambda x: x['unix_timestamp'])
    
    # Save to JSON file
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)
        print(f"Query data successfully extracted to {output_json_path}")
        
        # Print summary
        print("\nExtraction Summary:")
        total_entries = 0
        for user_id in user_data:
            user_entries = sum(len(tasks) for tasks in user_data[user_id].values())
            total_entries += user_entries
            task_count = len(user_data[user_id])
            print(f"  User {user_id}: {task_count} tasks, {user_entries} total entries")
        
        print(f"\nTotal entries processed: {total_entries}")
        return True
        
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False


def main():
    """Main function to run the query extraction."""
    csv_file = 'full_query_logs_table.csv'
    json_file = 'query_data.json'
    
    print("Step 1: Extracting Query Data")
    print("=" * 40)
    
    success = extract_query_data(csv_file, json_file)
    
    if success:
        print("\nStep 1 completed successfully!")
        print(f"Query data saved to: {json_file}")
        print("\nData structure: {User ID: {Task ID: [{'query_id': value, 'user_query': value, 'llm_response_1': value, 'llm_response_2': value, 'unix_timestamp': value}, ...]}}")
        print("Entries are sorted by unix_timestamp within each user-task combination")
    else:
        print("\nStep 1 failed!")


if __name__ == "__main__":
    main()