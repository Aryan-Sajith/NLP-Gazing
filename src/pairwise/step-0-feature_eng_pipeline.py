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
        """Load ALL behavioral data for a specific query (including looking away periods)"""
        data = []
        try:
            if not file_path.exists():
                return data
                
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Include ALL rows for this query, even -1,-1 (looking away)
                    if row.get('query_id') == str(query_id):
                        try:
                            # Handle -1,-1 rows (user looking away) specially
                            x = float(row['x']) if row.get('x') != '-1' else -1
                            y = float(row['y']) if row.get('y') != '-1' else -1
                            centre_idx = int(row['centre_idx']) if (row.get('centre_idx') and 
                                                                     row['centre_idx'].strip() and 
                                                                     row['centre_idx'] != '-1') else -1
                            
                            data.append({
                                'x': x,
                                'y': y,
                                'window': row.get('window', ''),
                                'centre_idx': centre_idx,
                                'rel_ts': float(row['rel_ts']),
                                'abs_ts': float(row['abs_ts']),
                                'is_looking_at_text': x != -1 and y != -1 and centre_idx != -1,
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
        """Extract dual engagement metrics for a single modality"""
        if not data or response_length == 0:
            return {
                'focused_engagement_ratio': 0.0,
                'overall_attention_ratio': 0.0,
                'normalized_avg_char_position': 0.0,
                'reading_completion_ratio': 0.0,
                'normalized_char_position_variance': 0.0,
                'windowed_features': [0.0] * 100
            }
        
        # Sort data by timestamp for temporal analysis
        data = sorted(data, key=lambda x: x['rel_ts'])
        
        # Separate looking vs not-looking data
        looking_data = [d for d in data if d['is_looking_at_text']]
        
        # Extract timestamps
        all_timestamps = [d['rel_ts'] for d in data]
        looking_timestamps = [d['rel_ts'] for d in looking_data]
        
        # Calculate total session duration
        total_session_time = max(all_timestamps) - min(all_timestamps) if len(all_timestamps) > 1 else 0
        
        # 1A. Focused Engagement Ratio (gaps while looking at text)
        # Measures: "When looking at the text, how continuously did they read?"
        if len(looking_timestamps) > 1:
            inactivity_threshold = 2000.0  # milliseconds
            intervals = [looking_timestamps[i+1] - looking_timestamps[i] for i in range(len(looking_timestamps) - 1)]
            active_time = sum(min(interval, inactivity_threshold) for interval in intervals)
            looking_session_time = max(looking_timestamps) - min(looking_timestamps)
            focused_engagement_ratio = active_time / looking_session_time if looking_session_time > 0 else 0
        else:
            focused_engagement_ratio = 0.0
        
        # 1B. Overall Attention Ratio
        # Measures: "What % of total session time was spent looking at the text?"
        if total_session_time > 0:
            # Calculate time spent looking using eye tracker sampling rate
            # Approximate: (number of looking samples / total samples) * total time
            overall_attention_ratio = len(looking_data) / len(data)
        else:
            overall_attention_ratio = 0.0
        
        # For remaining features, only use looking_data (when user was engaged)
        if not looking_data:
            return {
                'focused_engagement_ratio': focused_engagement_ratio,
                'overall_attention_ratio': overall_attention_ratio,
                'normalized_avg_char_position': 0.0,
                'reading_completion_ratio': 0.0,
                'normalized_char_position_variance': 0.0,
                'windowed_features': [0.0] * 100
            }
        
        char_positions = [d['centre_idx'] for d in looking_data]
        
        # 2. Normalized Average Character Position
        normalized_positions = [pos / response_length for pos in char_positions]
        normalized_avg_char_position = statistics.mean(normalized_positions)
        
        # 3. Reading Completion Ratio
        furthest_char = max(char_positions)
        reading_completion_ratio = min(1.0, furthest_char / response_length)
        
        # 4. Normalized Character Position Variance
        normalized_char_position_variance = statistics.variance(normalized_positions) if len(normalized_positions) > 1 else 0
        
        # 5. Normalized Character Sequence Windowing (100 segments)
        windowed_features = self.calculate_windowed_features(looking_data, response_length, 100)
        
        return {
            'focused_engagement_ratio': focused_engagement_ratio,
            'overall_attention_ratio': overall_attention_ratio,
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
            
            # Engagement features (2 per modality = 4 total)
            combined_features['gaze_focused_engagement_ratio'] = gaze_features['focused_engagement_ratio']
            combined_features['gaze_overall_attention_ratio'] = gaze_features['overall_attention_ratio']
            combined_features['mouse_focused_engagement_ratio'] = mouse_features['focused_engagement_ratio']
            combined_features['mouse_overall_attention_ratio'] = mouse_features['overall_attention_ratio']
            
            # Other basic features (3 per modality = 6 total)
            combined_features['gaze_normalized_avg_char_position'] = gaze_features['normalized_avg_char_position']
            combined_features['gaze_reading_completion_ratio'] = gaze_features['reading_completion_ratio']
            combined_features['gaze_normalized_char_position_variance'] = gaze_features['normalized_char_position_variance']
            
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
            pairwise_features['llm_response_1'] = query_data.get('llm_response_1', '')
            pairwise_features['llm_response_2'] = query_data.get('llm_response_2', '')
            
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
            'llm_response_1', 'llm_response_2',
            
            # Target variables
            'likert_1', 'likert_2', 'preference', 'normalized_likert_1', 'normalized_likert_2', 'binary_preference',
            
            # Response A engagement features (2 per modality)
            'response_A_gaze_focused_engagement_ratio', 'response_A_gaze_overall_attention_ratio',
            'response_A_mouse_focused_engagement_ratio', 'response_A_mouse_overall_attention_ratio',
            
            # Response A other basic features
            'response_A_gaze_normalized_avg_char_position',
            'response_A_gaze_reading_completion_ratio', 'response_A_gaze_normalized_char_position_variance',
            'response_A_mouse_normalized_avg_char_position',
            'response_A_mouse_reading_completion_ratio', 'response_A_mouse_normalized_char_position_variance',
            'response_A_response_length', 'response_A_gaze_data_points', 'response_A_mouse_data_points',
            
            # Response B engagement features (2 per modality)
            'response_B_gaze_focused_engagement_ratio', 'response_B_gaze_overall_attention_ratio',
            'response_B_mouse_focused_engagement_ratio', 'response_B_mouse_overall_attention_ratio',
            
            # Response B other basic features
            'response_B_gaze_normalized_avg_char_position',
            'response_B_gaze_reading_completion_ratio', 'response_B_gaze_normalized_char_position_variance',
            'response_B_mouse_normalized_avg_char_position',
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
    data_dir = "/Users/mehulpatwari/Code/cics/ciir/research/NLP-Gazing/user_behavior"
    query_logs_file = "/Users/mehulpatwari/Code/cics/ciir/research/NLP-Gazing/full_query_logs_table.csv"
    output_csv = "/Users/mehulpatwari/Code/cics/ciir/research/NLP-Gazing/extracted_features.csv"

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
