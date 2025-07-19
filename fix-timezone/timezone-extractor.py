import csv
import json
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from functools import lru_cache

# Initialize geocoder and timezone finder
geolocator = Nominatim(user_agent="timezone_extractor")
tf = TimezoneFinder()

@lru_cache(maxsize=128)
def get_timezone(location):
    """Get timezone for a location string. Cached to avoid repeated API calls."""
    if not location or location.upper() == 'NULL':
        return None
    
    try:
        # Geocode the location
        geo = geolocator.geocode(location)
        if geo:
            # Get timezone from coordinates
            tz = tf.timezone_at(lat=geo.latitude, lng=geo.longitude)
            return tz
    except Exception as e:
        print(f"Error processing {location}: {e}")
        return None # Handles errors gracefully by mapping failed user timezone to None
    
    return None

def process_csv(input_file='record_info_table.csv', output_file='user_timezones.json'):
    """Process CSV and create timezone mapping."""
    user_timezones = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            user_id = row['user_id']
            location = row['location']
            
            # Get timezone for location
            timezone = get_timezone(location)
            user_timezones[user_id] = timezone
            
            print(f"Processed {user_id}: {location} -> {timezone}")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(user_timezones, f, indent=2)
    
    print(f"\nSaved {len(user_timezones)} user timezones to {output_file}")

if __name__ == "__main__":
    process_csv()