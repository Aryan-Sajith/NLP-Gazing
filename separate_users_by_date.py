import csv
import json
from datetime import datetime
from collections import defaultdict

# Read the CSV file
csv_file = 'full_query_logs_table.csv'
output_file = 'users_by_date.json'

# Dictionary to store each user's earliest timestamp
user_first_timestamp = {}

# Read CSV and find first timestamp for each user
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        user_id = row['user_id']
        timestamp_str = row['query_timestamp']
        
        # Parse the timestamp
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            # Store the earliest timestamp for each user
            if user_id not in user_first_timestamp:
                user_first_timestamp[user_id] = timestamp
            else:
                if timestamp < user_first_timestamp[user_id]:
                    user_first_timestamp[user_id] = timestamp
        except ValueError:
            print(f"Warning: Could not parse timestamp for user {user_id}: {timestamp_str}")
            continue

# Cutoff date: December 1, 2025
cutoff_date = datetime(2025, 12, 1, 0, 0, 0)

# Separate users into two lists
users_before_dec1 = []
users_after_dec1 = []

for user_id, first_timestamp in user_first_timestamp.items():
    if first_timestamp < cutoff_date:
        users_before_dec1.append(user_id)
    else:
        users_after_dec1.append(user_id)

# Create output dictionary
output = {
    "users_before_december_1_2025": sorted(users_before_dec1),
    "users_after_december_1_2025": sorted(users_after_dec1),
    "statistics": {
        "total_users": len(user_first_timestamp),
        "users_before_dec1": len(users_before_dec1),
        "users_after_dec1": len(users_after_dec1)
    }
}

# Write to JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)

print(f"Processing complete!")
print(f"Total users: {output['statistics']['total_users']}")
print(f"Users before December 1, 2025: {output['statistics']['users_before_dec1']}")
print(f"Users after December 1, 2025: {output['statistics']['users_after_dec1']}")
print(f"Output saved to: {output_file}")
