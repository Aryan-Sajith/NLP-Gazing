#!/usr/bin/env python3
"""
CSV Data Formatter Script

This script fixes formatting issues in CSV files where:
1. Text columns may contain unescaped double quotes
2. Text columns may be split due to commas, causing more than 6 columns per row

The script processes files to ensure exactly 6 columns per row with proper escaping.
"""

import argparse
from pathlib import Path

def escape_quotes(text):
    """
    Properly escape double quotes in text for CSV format.
    Replaces single " with "" and wraps the entire text in quotes.
    """
    if isinstance(text, str) and text:
        # Replace any existing double quotes with doubled quotes
        escaped_text = text.replace('"', '""')
        # Wrap the entire text in double quotes
        return f'"{escaped_text}"'
    return text


def fix_csv_row(row):
    """
    Fix a CSV row to ensure exactly 6 columns.
    
    Expected format:
    [x_coord, y_coord, text_content, id_number, timestamp1, timestamp2]
    
    Args:
        row (list): List of CSV fields from a row
        
    Returns:
        list: Fixed row with exactly 6 columns
    """
    if len(row) == 6:
        # Row is correct length, just escape quotes in text column (index 2)
        if len(row) > 2:
            row[2] = escape_quotes(row[2])
        return row
    
    elif len(row) > 6:
        # Row has too many columns - need to concatenate split text columns
        
        # First two columns are coordinates
        x_coord = row[0]
        y_coord = row[1]
        
        # Last three columns are id, timestamp1, timestamp2
        id_number = row[-3]
        timestamp1 = row[-2]
        timestamp2 = row[-1]
        
        # Everything in between should be concatenated as the text content
        text_parts = row[2:-3]
        
        # Concatenate with ", " (comma + space) as specified
        combined_text = ", ".join(text_parts)
        
        # Escape quotes in the combined text
        combined_text = escape_quotes(combined_text)
        
        return [x_coord, y_coord, combined_text, id_number, timestamp1, timestamp2]
    
    else:
        # Row has fewer than 6 columns - handle missing data
        # Pad with empty strings to reach 6 columns
        while len(row) < 6:
            row.append("")
        
        # Escape quotes in text column if it exists
        if len(row) > 2:
            row[2] = escape_quotes(row[2])
        
        return row


def process_csv_file(input_file_path, output_file_path):
    """
    Process a single CSV file and create a corrected version.
    
    Args:
        input_file_path (str): Path to input CSV file
        output_file_path (str): Path to output CSV file
    """
    print(f"Processing: {input_file_path}")
    
    processed_rows = []
    
    try:
        # Read the file line by line to handle malformed CSV
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                # Strip whitespace and skip empty lines
                line = line.strip()
                if not line:
                    continue
                
                # Split by comma (basic CSV parsing)
                # Note: This assumes no quoted fields with commas in the original
                row = [field.strip() for field in line.split(',')]
                
                # Fix the row
                fixed_row = fix_csv_row(row)
                processed_rows.append(fixed_row)
        
        # Write the corrected CSV
        with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
            for row in processed_rows:
                # Join the row with commas manually since we're handling quoting ourselves
                outfile.write(','.join(str(field) for field in row) + '\n')
        
        print(f"✓ Created corrected file: {output_file_path}")
        print(f"  Processed {len(processed_rows)} rows")
        
    except Exception as e:
        print(f"✗ Error processing {input_file_path}: {str(e)}")


def find_and_process_files(root_directory, target_filenames, dry_run=False):
    """
    Find and process all target files in subdirectories.
    
    Args:
        root_directory (str): Root directory to search
        target_filenames (list): List of filenames to look for
        dry_run (bool): If True, only show what would be processed
    """
    root_path = Path(root_directory)
    
    if not root_path.exists():
        print(f"Error: Directory {root_directory} does not exist")
        return
    
    found_files = []
    
    # Walk through all subdirectories
    for target_filename in target_filenames:
        for file_path in root_path.rglob(target_filename):
            if file_path.is_file():
                found_files.append(file_path)
    
    if not found_files:
        print(f"No target files found in {root_directory}")
        print(f"Searched for: {target_filenames}")
        return
    
    print(f"Found {len(found_files)} file(s) to process:")
    for file_path in found_files:
        print(f"  - {file_path}")
    
    if dry_run:
        print("\nDry run mode - no files will be modified")
        return
    
    print("\nProcessing files...")
    
    # Process each file
    for file_path in found_files:
        # Create output filename with "_formatted" suffix
        output_path = file_path.parent / f"{file_path.stem}_formatted{file_path.suffix}"
        process_csv_file(str(file_path), str(output_path))


def main():
    parser = argparse.ArgumentParser(
        description='Fix CSV formatting issues in gaze tracking data files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python data-format.py --file rel_gaze.csv
  
  # Process all rel_gaze.csv files in current directory and subdirectories
  python data-format.py --directory . --filenames rel_gaze.csv
  
  # Process multiple file types
  python data-format.py --directory /path/to/data --filenames rel_gaze.csv rel_mouse.csv
  
  # Dry run to see what would be processed
  python data-format.py --directory . --filenames rel_gaze.csv --dry-run
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', '-f', 
                      help='Process a single CSV file')
    group.add_argument('--directory', '-d', 
                      help='Root directory to search for files')
    
    parser.add_argument('--filenames', '-n', nargs='+', 
                       default=['rel_gaze.csv'],
                       help='List of filenames to process (default: rel_gaze.csv)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without making changes')
    
    args = parser.parse_args()
    
    if args.file:
        # Process single file
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Error: File {args.file} does not exist")
            return
        
        output_path = input_path.parent / f"{input_path.stem}_formatted{input_path.suffix}"
        
        if args.dry_run:
            print(f"Would process: {input_path} -> {output_path}")
        else:
            process_csv_file(str(input_path), str(output_path))
    
    else:
        # Process directory
        find_and_process_files(args.directory, args.filenames, args.dry_run)


if __name__ == "__main__":
    main()
