#!/usr/bin/env python3
"""
Unified Feature Extraction Pipeline for LLM Response Preference Prediction

This script processes multiple user/task combinations to extract behavioral features
from gaze and mouse tracking data for pairwise LLM response comparison.
Outputs a single CSV file with one row per valid pairwise comparison.

Author: Generated for batch processing of user/task combinations
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics

class FeatureExtractionPipeline:
    """Extract behavioral features for multiple user/task combinations"""
    
    def __init__(self, data_dir: str, query_logs_file: str, output_csv: str):
        self.data_dir = Path(data_dir)
        self.query_logs_file = Path(query_logs_file)
        self.output_csv = Path(output_csv)
        self.query_data_cache = {}
        self.load_all_query_data()
        
    def load_all_query_data(self):
        """Load all query data into memory for faster access"""
        try:
            with open(self.query_logs_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    query_id = int(row['query_ID'])
                    self.query_data_cache[query_id] = row
            print(f"Loaded {len(self.query_data_cache)} queries from logs")
        except Exception as e:
            print(f"Error loading query logs: {e}")
            self.query_data_cache = {}
    
    def find_user_task_combinations(self) -> List[Tuple[str, str]]:
        """Discover all user/task combinations in the data directory"""
        combinations = []
        try:
            for user_dir in self.data_dir.iterdir():
                print(f"Checking user directory: {user_dir}")
                if user_dir.is_dir():
                    user_id = user_dir.name
                    for task_dir in user_dir.iterdir():
                        if task_dir.is_dir():
                            task_id = task_dir.name
                            # Check if this combination has the required files
                            if self.has_required_files(user_id, task_id):
                                combinations.append((user_id, task_id))
            print(f"Found {len(combinations)} user/task combinations with required files")
            return combinations
        except Exception as e:
            print(f"Error discovering user/task combinations: {e}")
            return []
    
    def has_required_files(self, user_id: str, task_id: str) -> bool:
        """Check if a user/task combination has the required behavioral data files"""
        task_path = self.data_dir / user_id / task_id
        required_files = [
            "rel_gaze_one_query_id_assigned.csv",
            "rel_gaze_two_query_id_assigned.csv", 
            "rel_mouse_left_query_id_assigned.csv",
            "rel_mouse_right_query_id_assigned.csv"
        ]
        
        for file_name in required_files:
            if not (task_path / file_name).exists():
                return False
        return True
    
    def find_pairwise_queries(self, user_id: str, task_id: str) -> List[int]:
        """Find queries with both responses for a given user/task combination"""
        pairwise_queries = []
        
        for query_id, query_data in self.query_data_cache.items():
            if (query_data['user_id'] == user_id and 
                query_data['task_id'] == task_id and
                query_data.get('llm_response_2') and 
                query_data['llm_response_2'].strip() not in ['NULL', '', 'null']):
                pairwise_queries.append(query_id)
        
        return pairwise_queries
    
    def load_behavioral_data(self, file_path: Path, query_id: int) -> List[Dict]:
        """Load and filter behavioral data for a specific query"""
        data = []
        try:
            if not file_path.exists():
                return data
                
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Filter for valid data points and specific query
                    if (row.get('query_id') == str(query_id) and 
                        row.get('x') != '-1' and row.get('y') != '-1' and
                        row.get('centre_idx') and row['centre_idx'].strip()):
                        
                        try:
                            data.append({
                                'x': float(row['x']),
                                'y': float(row['y']),
                                'window': row.get('window', ''),
                                'centre_idx': int(row['centre_idx']),
                                'rel_ts': float(row['rel_ts']),
                                'abs_ts': float(row['abs_ts']),
                                'is_experimental_text': row.get('is_experimental_text', '').lower() == 'true'
                            })
                        except (ValueError, KeyError):
                            continue  # Skip malformed rows
                            
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
        
        return data
    
    def calculate_response_length(self, response_text: str) -> int:
        """Calculate the total character length of the response"""
        return len(response_text) if response_text else 0
    
    def extract_core_features(self, data: List[Dict], response_length: int) -> Dict:
        """Extract the 5 core features for a single modality"""
        if not data or response_length == 0:
            return {
                'active_engagement_ratio': 0.0,
                'normalized_avg_char_position': 0.0,
                'reading_completion_ratio': 0.0,
                'normalized_char_position_variance': 0.0,
                'windowed_features': [0.0] * 100
            }
        
        # Sort data by timestamp for temporal analysis
        data = sorted(data, key=lambda x: x['rel_ts'])
        
        # Extract timestamps and character positions
        timestamps = [d['rel_ts'] for d in data]
        char_positions = [d['centre_idx'] for d in data]
        
        # Calculate session duration
        total_session_time = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        
        # 1. Active Engagement Ratio
        # Estimate active time based on data point frequency
        if len(timestamps) > 1:
            avg_interval = total_session_time / (len(timestamps) - 1)
            active_time = len(data) * avg_interval
            active_engagement_ratio = min(1.0, active_time / total_session_time) if total_session_time > 0 else 0
        else:
            active_engagement_ratio = 0.0
        
        # 2. Normalized Average Character Position
        normalized_positions = [pos / response_length for pos in char_positions]
        normalized_avg_char_position = statistics.mean(normalized_positions)
        
        # 3. Reading Completion Ratio
        furthest_char = max(char_positions)
        reading_completion_ratio = min(1.0, furthest_char / response_length)
        
        # 4. Normalized Character Position Variance
        normalized_char_position_variance = statistics.variance(normalized_positions) if len(normalized_positions) > 1 else 0
        
        # 5. Normalized Character Sequence Windowing (100 segments)
        windowed_features = self.calculate_windowed_features(data, response_length, 100)
        
        return {
            'active_engagement_ratio': active_engagement_ratio,
            'normalized_avg_char_position': normalized_avg_char_position,
            'reading_completion_ratio': reading_completion_ratio,
            'normalized_char_position_variance': normalized_char_position_variance,
            'windowed_features': windowed_features
        }
    
    def calculate_windowed_features(self, data: List[Dict], response_length: int, num_windows: int) -> List[float]:
        """Split session into time windows and calculate average normalized character position per window"""
        if not data or len(data) < 2:
            return [0.0] * num_windows
        
        # Sort by timestamp
        data = sorted(data, key=lambda x: x['rel_ts'])
        
        min_time = min(d['rel_ts'] for d in data)
        max_time = max(d['rel_ts'] for d in data)
        time_span = max_time - min_time
        
        if time_span == 0:
            return [0.0] * num_windows
        
        window_size = time_span / num_windows
        windowed_features = []
        
        for i in range(num_windows):
            window_start = min_time + i * window_size
            window_end = min_time + (i + 1) * window_size
            
            # Get data points in this window
            window_data = [d for d in data if window_start <= d['rel_ts'] < window_end]
            
            if window_data:
                # Calculate average normalized character position for this window
                normalized_positions = [d['centre_idx'] / response_length for d in window_data]
                avg_normalized_pos = statistics.mean(normalized_positions)
                windowed_features.append(avg_normalized_pos)
            else:
                windowed_features.append(0.0)
        
        return windowed_features
    
    def extract_features_for_response(self, user_id: str, task_id: str, query_id: int, 
                                    response_num: int, response_text: str) -> Optional[Dict]:
        """Extract all features for a single response (both gaze and mouse)"""
        try:
            # Determine which files to use based on response number
            if response_num == 1:
                gaze_file = self.data_dir / user_id / task_id / "rel_gaze_one_query_id_assigned.csv"
                mouse_file = self.data_dir / user_id / task_id / "rel_mouse_left_query_id_assigned.csv"
            else:
                gaze_file = self.data_dir / user_id / task_id / "rel_gaze_two_query_id_assigned.csv"
                mouse_file = self.data_dir / user_id / task_id / "rel_mouse_right_query_id_assigned.csv"
            
            # Load behavioral data
            gaze_data = self.load_behavioral_data(gaze_file, query_id)
            mouse_data = self.load_behavioral_data(mouse_file, query_id)
            
            # Calculate response length
            response_length = self.calculate_response_length(response_text)
            
            if response_length == 0:
                return None
            
            # Extract features for both modalities
            gaze_features = self.extract_core_features(gaze_data, response_length)
            mouse_features = self.extract_core_features(mouse_data, response_length)
            
            # Combine features with modality prefixes
            combined_features = {}
            
            # Basic features (4 per modality = 8 total)
            combined_features['gaze_active_engagement_ratio'] = gaze_features['active_engagement_ratio']
            combined_features['gaze_normalized_avg_char_position'] = gaze_features['normalized_avg_char_position']
            combined_features['gaze_reading_completion_ratio'] = gaze_features['reading_completion_ratio']
            combined_features['gaze_normalized_char_position_variance'] = gaze_features['normalized_char_position_variance']
            
            combined_features['mouse_active_engagement_ratio'] = mouse_features['active_engagement_ratio']
            combined_features['mouse_normalized_avg_char_position'] = mouse_features['normalized_avg_char_position']
            combined_features['mouse_reading_completion_ratio'] = mouse_features['reading_completion_ratio']
            combined_features['mouse_normalized_char_position_variance'] = mouse_features['normalized_char_position_variance']
            
            # Windowed features (100 per modality = 200 total)
            for i, val in enumerate(gaze_features['windowed_features']):
                combined_features[f'gaze_window_{i:03d}'] = val
            
            for i, val in enumerate(mouse_features['windowed_features']):
                combined_features[f'mouse_window_{i:03d}'] = val
            
            # Add metadata
            combined_features['response_length'] = response_length
            combined_features['gaze_data_points'] = len(gaze_data)
            combined_features['mouse_data_points'] = len(mouse_data)
            
            return combined_features
            
        except Exception as e:
            print(f"Error extracting features for {user_id}/{task_id}/query_{query_id}/response_{response_num}: {e}")
            return None
    
    def create_pairwise_features(self, user_id: str, task_id: str, query_id: int) -> Optional[Dict]:
        """Create pairwise feature vector for a query with both responses"""
        try:
            # Get query data
            query_data = self.query_data_cache.get(query_id)
            if not query_data:
                return None
            
            # Check if both responses exist and are valid
            if (not query_data.get('llm_response_2') or 
                query_data['llm_response_2'].strip() in ['NULL', '', 'null']):
                return None
            
            # Extract features for both responses
            response_1_features = self.extract_features_for_response(
                user_id, task_id, query_id, 1, query_data['llm_response_1']
            )
            response_2_features = self.extract_features_for_response(
                user_id, task_id, query_id, 2, query_data['llm_response_2']
            )
            
            if not response_1_features or not response_2_features:
                return None
            
            # Create pairwise feature vector
            pairwise_features = {}
            
            # Add metadata first
            pairwise_features['query_id'] = query_id
            pairwise_features['user_id'] = user_id
            pairwise_features['task_id'] = task_id
            pairwise_features['user_query'] = query_data['user_query']
            pairwise_features['llm_name_1'] = query_data.get('llm_name_1', '')
            pairwise_features['llm_name_2'] = query_data.get('llm_name_2', '')
            
            # Add target variables
            try:
                pairwise_features['likert_1'] = float(query_data.get('likert_1', 0)) if query_data.get('likert_1') else 0
                pairwise_features['likert_2'] = float(query_data.get('likert_2', 0)) if query_data.get('likert_2') else 0
                pairwise_features['preference'] = int(query_data.get('preference', 0)) if query_data.get('preference') else 0
            except (ValueError, TypeError):
                pairwise_features['likert_1'] = 0
                pairwise_features['likert_2'] = 0
                pairwise_features['preference'] = 0
            
            pairwise_features['normalized_likert_1'] = pairwise_features['likert_1'] / 5.0
            pairwise_features['normalized_likert_2'] = pairwise_features['likert_2'] / 5.0
            
            # Binary preference (0 if response A preferred, 1 if response B preferred)
            pairwise_features['binary_preference'] = 1 if pairwise_features['preference'] == 2 else 0
            
            # Add response A features (response 1)
            for key, value in response_1_features.items():
                pairwise_features[f'response_A_{key}'] = value
            
            # Add response B features (response 2)
            for key, value in response_2_features.items():
                pairwise_features[f'response_B_{key}'] = value
            
            return pairwise_features
            
        except Exception as e:
            print(f"Error creating pairwise features for {user_id}/{task_id}/query_{query_id}: {e}")
            return None
    
    def get_csv_headers(self) -> List[str]:
        """Generate CSV headers for the feature vector"""
        headers = [
            # Metadata
            'query_id', 'user_id', 'task_id', 'user_query', 'llm_name_1', 'llm_name_2',
            
            # Target variables
            'likert_1', 'likert_2', 'preference', 'normalized_likert_1', 'normalized_likert_2', 'binary_preference',
            
            # Response A basic features
            'response_A_gaze_active_engagement_ratio', 'response_A_gaze_normalized_avg_char_position',
            'response_A_gaze_reading_completion_ratio', 'response_A_gaze_normalized_char_position_variance',
            'response_A_mouse_active_engagement_ratio', 'response_A_mouse_normalized_avg_char_position',
            'response_A_mouse_reading_completion_ratio', 'response_A_mouse_normalized_char_position_variance',
            'response_A_response_length', 'response_A_gaze_data_points', 'response_A_mouse_data_points',
            
            # Response B basic features
            'response_B_gaze_active_engagement_ratio', 'response_B_gaze_normalized_avg_char_position',
            'response_B_gaze_reading_completion_ratio', 'response_B_gaze_normalized_char_position_variance',
            'response_B_mouse_active_engagement_ratio', 'response_B_mouse_normalized_avg_char_position',
            'response_B_mouse_reading_completion_ratio', 'response_B_mouse_normalized_char_position_variance',
            'response_B_response_length', 'response_B_gaze_data_points', 'response_B_mouse_data_points'
        ]
        
        # Add windowed features for both responses
        for response in ['A', 'B']:
            for modality in ['gaze', 'mouse']:
                for i in range(100):
                    headers.append(f'response_{response}_{modality}_window_{i:03d}')
        
        return headers
    
    def process_all_combinations(self) -> int:
        """Process all user/task combinations and write results to CSV"""
        combinations = self.find_user_task_combinations()
        total_processed = 0
        total_successful = 0
        
        if not combinations:
            print("No valid user/task combinations found")
            return 0
        
        # Create output directory if it doesn't exist
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Open CSV file for writing
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            headers = self.get_csv_headers()
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for user_id, task_id in combinations:
                print(f"Processing {user_id}/{task_id}...")
                
                # Find pairwise queries for this combination
                pairwise_queries = self.find_pairwise_queries(user_id, task_id)
                
                if not pairwise_queries:
                    print(f"  No pairwise queries found for {user_id}/{task_id}")
                    continue
                
                for query_id in pairwise_queries:
                    total_processed += 1
                    
                    # Extract features for this query
                    features = self.create_pairwise_features(user_id, task_id, query_id)
                    
                    if features:
                        # Write row to CSV
                        writer.writerow(features)
                        total_successful += 1
                        print(f"  ✓ Query {query_id}: Features extracted successfully")
                    else:
                        print(f"  ✗ Query {query_id}: Failed to extract features")
        
        print(f"\nProcessing complete!")
        print(f"Total queries processed: {total_processed}")
        print(f"Successful extractions: {total_successful}")
        print(f"Output file: {self.output_csv}")
        
        return total_successful

def main():
    """Main function to run the feature extraction pipeline"""
    
    # Configuration
    data_dir = "/Users/aryan-sajith/Downloads/NLP-Gaze-Feature-Eng/small-scale-test/data"
    query_logs_file = "/Users/aryan-sajith/Downloads/NLP-Gaze-Feature-Eng/small-scale-test/full_query_logs_table.csv"
    output_csv = "/Users/aryan-sajith/Downloads/NLP-Gaze-Feature-Eng/small-scale-test/extracted_features.csv"

    # Initialize and run pipeline
    pipeline = FeatureExtractionPipeline(data_dir, query_logs_file, output_csv)
    successful_extractions = pipeline.process_all_combinations()
    
    if successful_extractions > 0:
        print(f"\n🎉 Successfully extracted features for {successful_extractions} pairwise comparisons")
        print(f"📁 Output saved to: {output_csv}")
    else:
        print("\n❌ No features were successfully extracted")

if __name__ == "__main__":
    main()
