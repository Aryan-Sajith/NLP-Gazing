import pandas as pd
import json
import pytz
from dateutil import parser
import os

def convert_to_gmt(timestamp_str, timezone_str):
    """
    Convert a timestamp string from a given timezone to GMT.
    
    Args:
        timestamp_str: String in any common datetime format
        timezone_str: Timezone string (e.g., "America/New_York")
    
    Returns:
        GMT datetime string in ISO format
    """
    if pd.isna(timestamp_str) or not timestamp_str:
        return None
    
    if not timezone_str:
        # If no timezone is specified, assume UTC
        timezone_str = 'UTC'
    
    try:
        # Parse the timestamp using dateutil.parser (format-agnostic)
        dt = parser.parse(timestamp_str)
        
        # Create timezone object
        local_tz = pytz.timezone(timezone_str)
        
        # Localize the datetime to the user's timezone
        local_dt = local_tz.localize(dt)
        
        # Convert to GMT (UTC)
        gmt_dt = local_dt.astimezone(pytz.UTC)
        
        # Return in ISO format
        return gmt_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    except Exception as e:
        print(f"Error converting timestamp '{timestamp_str}' with timezone '{timezone_str}': {e}")
        return None

def main(input_file='task_table.csv', user_id_col='user_id', timestamp_col='finished'):
    # Read the timezone mapping
    with open('user_timezones.json', 'r') as f:
        user_timezones = json.load(f)
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Create a list to store GMT times
    gmt_times = []
    
    # Process each row
    for idx, row in df.iterrows():
        user_id = row[user_id_col]
        timestamp = row[timestamp_col]
        
        # Get user's timezone (default to UTC if not found or null)
        user_tz = user_timezones.get(user_id, 'UTC')
        if user_tz is None:
            user_tz = 'UTC'
        
        # Convert to GMT
        gmt_time = convert_to_gmt(timestamp, user_tz)
        gmt_times.append(gmt_time)
        
        # Print progress every 10 rows
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} rows...")
    
    # Add the new column to the dataframe
    df['gmt_time'] = gmt_times
    
    # Save the updated CSV
    # Ensure the output file name is derived from the input file name
    output_file = f"{input_file.split('.')[0]}_with_gmt.csv"
    df.to_csv(output_file, index=False)
    print(f"\nConversion complete! Processed {len(df)} rows.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Start by moving to the script's directory for relative file access
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # Run the main function
    main()