"""
Step 1: Extract Query Data
Extracts query data from the LLM logs and creates a structured JSON file.
"""

import pandas as pd
import json
from datetime import datetime


def parse_timestamp(timestamp_str):
    """Parse timestamp from format '4/23/25 13:31' to Unix timestamp."""
    try:
        # Parse the date string
        dt = datetime.strptime(timestamp_str, '%m/%d/%y %H:%M')
        # Convert to Unix timestamp (in milliseconds to match gaze data)
        return int(dt.timestamp() * 1000)
    except Exception as e:
        print(f"Error parsing timestamp {timestamp_str}: {e}")
        return None


def extract_query_data(csv_file_path, output_json_path):
    """
    Extract query data from CSV and create structured JSON file.
    
    Args:
        csv_file_path (str): Path to the LLM query logs CSV
        output_json_path (str): Path for output JSON file
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} total query records")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False
    
    # Group data by user and task
    user_task_data = {}
    
    for _, row in df.iterrows():
        user_id = row['user_id']
        task_id = row['task_id']
        query_id = row['query_ID']
        user_query = row['user_query']
        llm_response = row['llm_response_1']
        timestamp_str = row['query_timestamp']
        
        # Parse timestamp
        unix_timestamp = parse_timestamp(timestamp_str)
        if unix_timestamp is None:
            continue
        
        # Create user key
        user_key = user_id
        
        # Initialize user data if not exists
        if user_key not in user_task_data:
            user_task_data[user_key] = {}
        
        # Initialize task data if not exists
        if task_id not in user_task_data[user_key]:
            user_task_data[user_key][task_id] = {
                'queries': [],
                'timestamps': []
            }
        
        # Add query data
        query_data = {
            'query_id': query_id,
            'timestamp': unix_timestamp,
            'user_query': user_query,
            'llm_response': llm_response,
            'original_timestamp': timestamp_str
        }
        
        user_task_data[user_key][task_id]['queries'].append(query_data)
        user_task_data[user_key][task_id]['timestamps'].append(unix_timestamp)
    
    # Sort queries by timestamp for each user-task combination
    for user_id in user_task_data:
        for task_id in user_task_data[user_id]:
            # Sort queries by timestamp
            user_task_data[user_id][task_id]['queries'].sort(key=lambda x: x['timestamp'])
            # Sort timestamps
            user_task_data[user_id][task_id]['timestamps'].sort()
    
    # Save to JSON file
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(user_task_data, f, indent=2, ensure_ascii=False)
        print(f"Query data successfully extracted to {output_json_path}")
        
        # Print summary
        print("\\nExtraction Summary:")
        for user_id in user_task_data:
            for task_id in user_task_data[user_id]:
                query_count = len(user_task_data[user_id][task_id]['queries'])
                print(f"  User {user_id}, Task {task_id}: {query_count} queries")
                
                # Print specific details for our target user
                if user_id == 'AGD0PFON3GYZT' and task_id == 28:
                    print(f"    Target user-task details:")
                    for query in user_task_data[user_id][task_id]['queries']:
                        print(f"      Query {query['query_id']}: {query['original_timestamp']} -> {query['timestamp']}")
                        print(f"        Question: {query['user_query']}")
                        print(f"        Response preview: {query['llm_response'][:100]}...")
                        print()
        
        return True
        
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False


def main():
    """Main function to run the query extraction."""
    csv_file = 'one_llm_query_logs_table.csv'
    json_file = 'query_data.json'
    
    print("Step 1: Extracting Query Data")
    print("=" * 40)
    
    success = extract_query_data(csv_file, json_file)
    
    if success:
        print("\\nStep 1 completed successfully!")
        print(f"Query data saved to: {json_file}")
    else:
        print("\\nStep 1 failed!")


if __name__ == "__main__":
    main()
