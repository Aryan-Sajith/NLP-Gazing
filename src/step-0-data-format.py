"""
CSV Data Formatter Script

⚠️ WARNING: DO NOT USE THIS SCRIPT! ⚠️

This script was found to CORRUPT data rather than fix it.
The original CSV files from user_behavior_org are already properly formatted.

This script uses naive string splitting that breaks CSV quoting rules.
Running it will corrupt properly-formatted CSV files.

SOLUTION: Skip this step entirely. Use the original files directly.
See PIPELINE_INSTRUCTIONS.md for correct pipeline execution.

---

Original description (DO NOT USE):
This script recursively finds and processes CSV files in a directory,
fixing formatting issues and replacing the original files.
"""

import argparse
import os
from pathlib import Path

def escape_quotes(text):
    """Escape double quotes in text for CSV format."""
    if isinstance(text, str) and text:
        escaped_text = text.replace('"', '""')
        return f'"{escaped_text}"'
    return text

def fix_csv_row(row):
    """Fix a CSV row to ensure exactly 6 columns."""
    if len(row) == 6:
        if len(row) > 2:
            row[2] = escape_quotes(row[2])
        return row
    elif len(row) > 6:
        # Concatenate split text columns
        x_coord = row[0]
        y_coord = row[1]
        id_number = row[-3]
        timestamp1 = row[-2]
        timestamp2 = row[-1]
        text_parts = row[2:-3]
        combined_text = ",".join(text_parts)
        combined_text = escape_quotes(combined_text)
        return [x_coord, y_coord, combined_text, id_number, timestamp1, timestamp2]
    else:
        # Pad with empty strings
        while len(row) < 6:
            row.append("")
        if len(row) > 2:
            row[2] = escape_quotes(row[2])
        return row

def process_file(file_path):
    """Process a single CSV file and replace it with the corrected version."""
    print(f"Processing: {file_path}")
    
    # Create temporary file
    temp_path = file_path.parent / f"{file_path.stem}_temp{file_path.suffix}"
    
    try:
        processed_rows = []
        
        # Read and process the file
        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                row = [field for field in line.split(',')]
                fixed_row = fix_csv_row(row)
                processed_rows.append(fixed_row)
        
        # Write to temporary file
        with open(temp_path, 'w', newline='', encoding='utf-8') as outfile:
            for row in processed_rows:
                outfile.write(','.join(str(field) for field in row) + '\n')
        
        # Replace original file
        os.remove(file_path)
        os.rename(temp_path, file_path)
        
        print(f"✓ Processed and replaced: {file_path} ({len(processed_rows)} rows)")
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {str(e)}")
        # Clean up temp file if it exists
        if temp_path.exists():
            os.remove(temp_path)

def find_and_process_files(directory, filenames):
    """Find and process all target files recursively in directory."""
    root_path = Path(directory)
    
    if not root_path.exists():
        print(f"Error: Directory {directory} does not exist")
        return
    
    found_files = []
    
    # Find all matching files recursively
    for target_filename in filenames:
        for file_path in root_path.rglob(target_filename):
            if file_path.is_file():
                found_files.append(file_path)
    
    if not found_files:
        print(f"No files found matching: {filenames}")
        return
    
    print(f"Found {len(found_files)} file(s) to process")
    
    # Process each file
    for file_path in found_files:
        process_file(file_path)

def main():
    parser = argparse.ArgumentParser(
        description='Recursively process and replace CSV files with formatted versions'
    )
    default_directory = "user_behavior/"
    # Only process pairwise files - exclude pointwise files (rel_gaze.csv, rel_mouse.csv)
    default_filenames = [
        'rel_gaze_one.csv', 
        'rel_gaze_two.csv', 
        'rel_mouse_left.csv', 
        'rel_mouse_right.csv'
    ]
    parser.add_argument('directory', nargs='?',
                       help='Directory to search recursively',
                       default=default_directory)
    parser.add_argument('--filenames', '-f', nargs='+', 
                       default=default_filenames,
                       help='Filenames to process (default: pairwise files only)')
    
    args = parser.parse_args()
    
    print(f"Processing pairwise data only (excluding pointwise files)")
    print(f"Target files: {args.filenames}")
    find_and_process_files(args.directory, args.filenames)

if __name__ == "__main__":
    main()
