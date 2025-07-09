#!/usr/bin/env python3
"""
Step 2: Match Gaze Data with Query IDs
Processes gaze data and assigns query IDs based on timestamps and text matching.
"""

import json
import csv
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any


def find_user_task_files(base_dir: str, file_patterns: List[str]) -> List[Dict[str, str]]:
    """
    Recursively find files matching patterns within user/task directory structure.
    Expected structure: base_dir/user_id/task_id/files
    
    Args:
        base_dir (str): Base directory containing user directories
        file_patterns (List[str]): List of file patterns to match (e.g., ['*.csv', 'rel_*.csv'])
    
    Returns:
        List[Dict]: List of dictionaries containing file info with keys:
                   'file_path', 'user_id', 'task_id', 'output_path'
    """
    matching_files = []
    base_path = Path(base_dir)
    
    # Iterate through user directories (immediate children of base_dir)
    for user_dir in base_path.iterdir():
        if not user_dir.is_dir():
            continue
            
        user_id = user_dir.name
        
        # Iterate through task directories (children of user directories)
        for task_dir in user_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            try:
                task_id = int(task_dir.name)
            except ValueError:
                # Skip directories that aren't numeric task IDs
                continue
            
            # Search for matching files in this task directory
            for pattern in file_patterns:
                matches = list(task_dir.glob(pattern))
                for match in matches:
                    if match.is_file():
                        output_path = generate_output_filename(str(match))
                        matching_files.append({
                            'file_path': str(match),
                            'user_id': user_id,
                            'task_id': task_id,
                            'output_path': output_path
                        })
    
    return sorted(matching_files, key=lambda x: (x['user_id'], x['task_id'], x['file_path']))


def generate_output_filename(input_file: str) -> str:
    """
    Generate output filename by adding 'with-query-id' before the file extension.
    
    Args:
        input_file (str): Input file path
    
    Returns:
        str: Output file path
    """
    path = Path(input_file)
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    
    output_filename = f"{stem}_with_query_ids{suffix}"
    return str(parent / output_filename)


def load_query_data(json_file_path: str) -> Dict[str, Any]:
    """Load query data from JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading query data: {e}")
        return {}


def text_matches_query(gaze_text: str, query_data: Dict[str, Any], verbose: bool = False) -> bool:
    """
    Check if gaze text matches content in query or response.
    Requires exact substring match - the gaze text must be present as-is in the query content.
    
    Args:
        gaze_text (str): Text from gaze data
        query_data (dict): Query data containing user_query and llm_response
        verbose (bool): Whether to print debug information
    
    Returns:
        bool: True if text matches exactly
    """
    if not gaze_text or gaze_text.strip() == "":
        return False
    
    # Clean and normalize text for comparison
    gaze_text_clean = gaze_text.lower().strip()
    
    # Must be at least 4 characters to avoid false matches
    if len(gaze_text_clean) < 4:
        return False
    
    # Check if the exact gaze text appears as a substring in user query
    user_query = query_data.get('user_query', '').lower()
    if gaze_text_clean in user_query:
        if verbose:
            print(f"        Exact match found in user query: '{gaze_text_clean}' in '{user_query}'")
        return True
    
    # Check if the exact gaze text appears as a substring in LLM response
    llm_response = query_data.get('llm_response', '').lower()
    if gaze_text_clean in llm_response:
        if verbose:
            print(f"        Exact match found in LLM response: '{gaze_text_clean}'")
        return True
    
    return False


def check_transition_confirmation(gaze_rows: List[List[str]], start_idx: int, next_query: Dict[str, Any], 
                                confirmation_count: int = 3) -> bool:
    """
    Check subsequent non-placeholder entries to confirm the transition is correct.
    Uses a while loop to continue until we find the required number of non-placeholder entries
    or reach the end of the file.
    
    Args:
        gaze_rows (List[List[str]]): All gaze data rows
        start_idx (int): Starting index for confirmation check
        next_query (dict): The query we're considering transitioning to
        confirmation_count (int): Number of subsequent entries to check
    
    Returns:
        bool: True if transition is confirmed by subsequent entries
    """
    confirmed_matches = 0
    checked_entries = 0
    current_idx = start_idx + 1
    
    # Continue looking ahead until we find enough non-placeholder entries or reach end of file
    while checked_entries < confirmation_count and current_idx < len(gaze_rows):
        row = gaze_rows[current_idx]
        
        # Skip rows with insufficient columns
        if len(row) < 6:
            current_idx += 1
            continue
            
        # Skip placeholder entries (-1, -1) - these don't count toward our confirmation
        try:
            pixel_x = float(row[0]) if row[0] != '-1' else -1
            pixel_y = float(row[1]) if row[1] != '-1' else -1
            
            if pixel_x == -1 and pixel_y == -1:
                current_idx += 1
                continue
        except (ValueError, IndexError):
            current_idx += 1
            continue
        
        # This is a valid non-placeholder entry
        text_content = row[2].strip('"') if len(row[2]) > 0 else ""
        
        if text_content and text_content.strip() != "":
            checked_entries += 1
            if text_matches_query(text_content, next_query, verbose=False):
                confirmed_matches += 1
        
        current_idx += 1
    
    # Calculate confirmation ratio
    if checked_entries == 0:
        print(f"        Confirmation check: No valid entries found for confirmation")
        return False
    
    confirmation_ratio = confirmed_matches / checked_entries
    print(f"        Confirmation check: {confirmed_matches}/{checked_entries} subsequent entries match ({confirmation_ratio:.2f})")
    
    # If we reached end of file before finding enough entries, decide based on what we have
    if checked_entries < confirmation_count:
        print(f"        Note: Only found {checked_entries} entries before end of file (needed {confirmation_count})")
    
    # Require ALL subsequent entries to match for confirmation (100% match rate)
    return confirmed_matches == checked_entries


def process_gaze_data(gaze_file_path: str, query_data: Dict[str, Any], 
                     user_id: str, task_id: int, output_file_path: str, 
                     confirmation_count: int = 3) -> bool:
    """
    Process gaze data and assign query IDs.
    
    Args:
        gaze_file_path (str): Path to gaze CSV file
        query_data (dict): Query data from JSON
        user_id (str): Target user ID
        task_id (int): Target task ID
        output_file_path (str): Output file path
        confirmation_count (int): Number of subsequent non-placeholder entries to check for transition confirmation
    
    Returns:
        bool: Success status
    """
    
    print(f"\nProcessing file: {gaze_file_path}")
    print(f"Output file: {output_file_path}")
    
    # Get query data for specific user and task
    # Convert task_id to string since JSON keys are strings
    task_id_str = str(task_id)
    if user_id not in query_data or task_id_str not in query_data[user_id]:
        print(f"No query data found for user {user_id}, task {task_id}")
        print(f"Available users: {list(query_data.keys())}")
        if user_id in query_data:
            print(f"Available tasks for user {user_id}: {list(query_data[user_id].keys())}")
        return False
    
    queries = query_data[user_id][task_id_str]['queries']
    print(f"Processing {len(queries)} queries for user {user_id}, task {task_id}")
    
    # Print query information
    for query in queries:
        print(f"  Query {query['query_id']}: {query['timestamp']} - {query['user_query']}")
    
    # Read gaze data
    try:
        gaze_rows = []
        with open(gaze_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:  # Ensure we have all required columns
                    gaze_rows.append(row)
        
        print(f"Loaded {len(gaze_rows)} gaze data rows")
        
    except Exception as e:
        print(f"Error reading gaze file: {e}")
        return False
    
    # Process each gaze entry
    processed_rows = []
    current_query_idx = 0
    current_query_id = queries[0]['query_id']
    
    transition_lines = []  # Track where query transitions happen
    
    for row_idx, row in enumerate(gaze_rows):
        try:
            pixel_x = float(row[0]) if row[0] != '-1' else -1
            pixel_y = float(row[1]) if row[1] != '-1' else -1
            text_content = row[2].strip('"') if len(row[2]) > 0 else ""
            unix_timestamp = int(row[5])
            
            # Skip placeholder entries (-1, -1)
            if pixel_x == -1 and pixel_y == -1:
                query_id = current_query_id
            else:
                # Check if we should transition to next query
                # Only transition if BOTH conditions are met:
                # 1. Timestamp is after next query timestamp
                # 2. Text content matches the next query
                if current_query_idx < len(queries) - 1:  # Not the last query
                    next_query = queries[current_query_idx + 1]
                    
                    # Only check for text match if timestamp condition is met
                    # AND there is actual text content to match against
                    if (unix_timestamp >= next_query['timestamp'] and 
                        text_content and text_content.strip() != ""):
                        
                        # First check if current text matches the next query content
                        if text_matches_query(text_content, next_query, verbose=True):
                            print(f"      Potential transition detected at line {row_idx + 1}")
                            print(f"      Text: '{text_content}' matches next query")
                            
                            # Confirm transition by checking subsequent entries
                            if check_transition_confirmation(gaze_rows, row_idx, next_query, confirmation_count):
                                # Transition to next query
                                current_query_idx += 1
                                current_query_id = next_query['query_id']
                                transition_lines.append({
                                    'line': row_idx + 1,
                                    'from_query': queries[current_query_idx-1]['query_id'],
                                    'to_query': current_query_id,
                                    'text': text_content,
                                    'timestamp': unix_timestamp
                                })
                                print(f"    ✅ CONFIRMED Transition at line {row_idx + 1}: Query {queries[current_query_idx-1]['query_id']} -> {current_query_id}")
                                print(f"      Text: '{text_content}'")
                                print(f"      Timestamp: {unix_timestamp}")
                            else:
                                print(f"    ❌ REJECTED Transition at line {row_idx + 1}: Insufficient confirmation from subsequent entries")
                
                query_id = current_query_id
            
            # Create new row with query_id
            new_row = row + [query_id]
            processed_rows.append(new_row)
            
        except Exception as e:
            print(f"Error processing row {row_idx + 1}: {e}")
            # Keep original row and add empty query_id
            processed_rows.append(row + [''])
    
    # Write processed data
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['pixel_x', 'pixel_y', 'text_content', 'char_count', 'timestamp', 'unix_timestamp', 'query_id']
            writer.writerow(header)
            
            # Write data
            for row in processed_rows:
                writer.writerow(row)
        
        print(f"\nProcessed gaze data saved to: {output_file_path}")
        print(f"Total rows processed: {len(processed_rows)}")
        
        # Print transition summary
        if transition_lines:
            print(f"\nQuery transitions detected:")
            for transition in transition_lines:
                print(f"  Line {transition['line']}: Query {transition['from_query']} -> {transition['to_query']}")
        
        return True
        
    except Exception as e:
        print(f"Error saving processed file: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Match gaze/mouse tracking data with query IDs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all CSV files in user/task structure
  python step-2-match-gaze-queries.py --base-dir . --file-patterns "*.csv" --query-json query_data.json

  # Process specific patterns with custom confirmation count
  python step-2-match-gaze-queries.py --base-dir /data --file-patterns "rel_*.csv" "gaze_*.csv" --query-json queries.json --confirmation-count 10
        """
    )
    
    parser.add_argument('--base-dir', '-d', required=True,
                       help='Base directory to search for files')
    
    parser.add_argument('--file-patterns', '-f', nargs='+', required=True,
                       help='File patterns to match (e.g., *.csv rel_*.csv)')
    
    parser.add_argument('--query-json', '-q', required=True,
                       help='Path to JSON file containing query data')
    
    parser.add_argument('--confirmation-count', '-c', type=int, default=3,
                       help='Number of subsequent entries to confirm transition (default: 3)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def main():
    """Main function to run gaze data processing."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("Step 2: Matching Gaze Data with Query IDs")
    print("=" * 45)
    print(f"Base directory: {args.base_dir}")
    print(f"File patterns: {args.file_patterns}")
    print(f"Query JSON: {args.query_json}")
    print(f"Confirmation count: {args.confirmation_count}")
    
    # Validate base directory
    if not os.path.isdir(args.base_dir):
        print(f"Error: Base directory '{args.base_dir}' does not exist!")
        return 1
    
    # Validate query JSON file
    if not os.path.isfile(args.query_json):
        print(f"Error: Query JSON file '{args.query_json}' does not exist!")
        return 1
    
    # Load query data
    print("\nLoading query data...")
    query_data = load_query_data(args.query_json)
    
    if not query_data:
        print("Failed to load query data!")
        return 1
    
    # Find matching files in user/task structure
    print(f"\nSearching for files matching patterns: {args.file_patterns}")
    print("Expected directory structure: base_dir/user_id/task_id/files")
    file_info_list = find_user_task_files(args.base_dir, args.file_patterns)
    
    if not file_info_list:
        print("No matching files found in user/task directory structure!")
        return 1
    
    print(f"Found {len(file_info_list)} matching files:")
    
    # Group files by user/task for display
    current_user = None
    current_task = None
    for file_info in file_info_list:
        if file_info['user_id'] != current_user:
            current_user = file_info['user_id']
            print(f"\n📁 User: {current_user}")
        if file_info['task_id'] != current_task:
            current_task = file_info['task_id']
            print(f"  📁 Task: {current_task}")
        print(f"    📄 {os.path.basename(file_info['file_path'])}")
    
    # Process each file
    successful_files = 0
    failed_files = 0
    
    for file_info in file_info_list:
        print(f"\n{'-' * 60}")
        print(f"Processing User: {file_info['user_id']}, Task: {file_info['task_id']}")
        
        try:
            success = process_gaze_data(
                file_info['file_path'],
                query_data,
                file_info['user_id'],
                file_info['task_id'],
                file_info['output_path'],
                args.confirmation_count
            )
            
            if success:
                successful_files += 1
                if args.verbose:
                    print(f"✅ Successfully processed: {file_info['file_path']}")
            else:
                failed_files += 1
                print(f"❌ Failed to process: {file_info['file_path']}")
                
        except Exception as e:
            failed_files += 1
            print(f"❌ Error processing {file_info['file_path']}: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files found: {len(file_info_list)}")
    print(f"Successfully processed: {successful_files}")
    print(f"Failed: {failed_files}")
    
    # Show unique users and tasks processed
    unique_users = set(f['user_id'] for f in file_info_list)
    unique_tasks = set((f['user_id'], f['task_id']) for f in file_info_list)
    print(f"Users processed: {len(unique_users)} ({', '.join(sorted(unique_users))})")
    print(f"User/Task combinations: {len(unique_tasks)}")
    
    if successful_files > 0:
        print(f"\n✅ Processing completed successfully!")
        return 0
    else:
        print(f"\n❌ All files failed to process!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
