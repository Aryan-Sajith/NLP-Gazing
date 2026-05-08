"""Extract behavioral features from annotated data"""

import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.config.constants import INACTIVITY_THRESHOLD_MS, NUM_TIME_WINDOWS
from src.models.features import ResponseFeatures, ModalityFeatures, ComparisonFeatures
from src.io.behavioral_reader import BehavioralReader
from src.io.query_reader import QueryReader
from src.io.feature_writer import FeatureWriter
from src.utils.file_discovery import discover_user_task_combinations, detect_comparison_type
from src.processing.phase_feature_extractor import PhaseFeatureExtractor


class FeatureExtractor:
    """Extracts behavioral features from gaze and mouse data"""
    
    def __init__(self, data_dir: Path, query_logs_path: Path, output_path: Path):
        self.data_dir = data_dir
        self.output_path = output_path
        self.behavioral_reader = BehavioralReader()
        self.feature_writer = FeatureWriter(output_path)
        self.phase_extractor = PhaseFeatureExtractor()
        
        # Load query cache
        query_reader = QueryReader(query_logs_path)
        self.query_cache = query_reader.read_query_cache()
    
    def extract_all_features(self) -> int:
        """
        Process all user/task combinations and extract features.
        
        Returns:
            Number of successful feature extractions
        """
        print("Discovering user/task combinations...")
        combinations = discover_user_task_combinations(self.data_dir)
        
        if not combinations:
            print("No valid combinations found")
            return 0
        
        print(f"Found {len(combinations)} user/task combinations")
        
        all_features = []
        total_processed = 0
        successful = 0
        
        for user_id, task_id in combinations:
            comparison_type = detect_comparison_type(self.data_dir, user_id, task_id)
            
            if not comparison_type:
                continue
            
            print(f"Processing {user_id}/{task_id} ({comparison_type})...")
            
            # Find queries for this task
            queries = self._find_queries_for_task(user_id, task_id, comparison_type)
            
            if not queries:
                print(f"  No valid queries found")
                continue
            
            for query_id in queries:
                total_processed += 1
                features = self._extract_query_features(
                    user_id, task_id, query_id, comparison_type
                )
                
                if features:
                    all_features.append(features.to_flat_dict())
                    successful += 1
                    print(f"  ✓ Query {query_id}")
                else:
                    print(f"  ✗ Query {query_id}: Failed")
        
        # Write all features
        if all_features:
            headers = FeatureWriter.get_feature_headers()
            self.feature_writer.write_features(all_features, headers)
        
        print(f"\nProcessing complete!")
        print(f"Total queries: {total_processed}")
        print(f"Successful: {successful}")
        
        return successful
    
    def _find_queries_for_task(self, user_id: str, task_id: str, 
                               comparison_type: str) -> List[int]:
        """Find all valid queries for a user/task"""
        queries = []
        
        for query_id, query_data in self.query_cache.items():
            if query_data['user_id'] != user_id or query_data['task_id'] != task_id:
                continue
            
            # Check if query has required responses
            if comparison_type == 'pairwise':
                if (query_data.get('llm_response_2') and
                    query_data['llm_response_2'].strip() not in ['NULL', '', 'null']):
                    queries.append(query_id)
            else:  # pointwise
                if (query_data.get('llm_response_1') and
                    query_data['llm_response_1'].strip() not in ['NULL', '', 'null']):
                    queries.append(query_id)
        
        return queries
    
    def _extract_query_features(self, user_id: str, task_id: str, 
                                query_id: int, comparison_type: str) -> Optional[ComparisonFeatures]:
        """Extract features for a single query"""
        try:
            query_data = self.query_cache.get(query_id)
            if not query_data:
                return None
            
            # Extract Response A features
            response_a = self._extract_response_features(
                user_id, task_id, query_id, comparison_type, 1,
                query_data['llm_response_1']
            )
            
            if not response_a:
                return None
            
            # Extract Response B features (if pairwise)
            response_b = None
            if comparison_type == 'pairwise':
                response_b = self._extract_response_features(
                    user_id, task_id, query_id, comparison_type, 2,
                    query_data['llm_response_2']
                )
                if not response_b:
                    return None
            
            # Extract phase features for pairwise
            if comparison_type == 'pairwise' and response_b:
                # Get response lengths from query data
                left_response_length = len(query_data.get('llm_response_1', ''))
                right_response_length = len(query_data.get('llm_response_2', ''))
                
                # Extract pairwise phase features
                gaze_phase = self.phase_extractor.extract_pairwise_features(
                    response_a.gaze_df, response_b.gaze_df, 'gaze',
                    left_response_length, right_response_length
                )
                mouse_phase = self.phase_extractor.extract_pairwise_features(
                    response_a.mouse_df, response_b.mouse_df, 'mouse',
                    left_response_length, right_response_length
                )
                cross_phase = self.phase_extractor.extract_cross_modality_features(
                    gaze_phase, mouse_phase, task_type='pairwise'
                )
                phase_features = {**gaze_phase, **mouse_phase, **cross_phase}
            else:
                # For pointwise, features already extracted
                phase_features = getattr(response_a, 'phase_features', {})
            
            # Parse target variables
            likert_1 = self._parse_float(query_data.get('likert_1'))
            likert_2 = self._parse_float(query_data.get('likert_2')) if comparison_type == 'pairwise' else None
            preference = self._parse_int(query_data.get('preference')) if comparison_type == 'pairwise' else None
            
            comparison_features = ComparisonFeatures(
                comparison_type=comparison_type,
                query_id=query_id,
                user_id=user_id,
                task_id=task_id,
                user_query=query_data['user_query'],
                llm_name_1=query_data.get('llm_name_1'),
                llm_name_2=query_data.get('llm_name_2') if comparison_type == 'pairwise' else None,
                llm_response_1=query_data.get('llm_response_1'),
                llm_response_2=query_data.get('llm_response_2') if comparison_type == 'pairwise' else None,
                response_a=response_a,
                response_b=response_b,
                likert_1=likert_1,
                likert_2=likert_2,
                preference=preference
            )
            comparison_features.phase_features = phase_features
            
            return comparison_features
            
        except Exception as e:
            print(f"Error extracting features for query {query_id}: {e}")
            return None
    
    def _extract_response_features(self, user_id: str, task_id: str, query_id: int,
                                   comparison_type: str, response_num: int,
                                   response_text: str) -> Optional[ModalityFeatures]:
        """Extract features for a single response (both gaze and mouse)"""
        try:
            # Determine file paths
            gaze_file, mouse_file = self._get_file_paths(
                user_id, task_id, comparison_type, response_num
            )
            
            response_length = len(response_text) if response_text else 0
            
            if response_length == 0:
                return None
            
            import pandas as pd
            
            if comparison_type == 'pointwise':
                # NEW: For pointwise, read full file and extract using timeline approach
                gaze_data_full = self.behavioral_reader.read_full_csv(gaze_file)
                mouse_data_full = self.behavioral_reader.read_full_csv(mouse_file)
                
                # Convert to DataFrames
                gaze_df = pd.DataFrame([vars(d) for d in gaze_data_full]) if gaze_data_full else pd.DataFrame()
                mouse_df = pd.DataFrame([vars(d) for d in mouse_data_full]) if mouse_data_full else pd.DataFrame()
                
                # Extract behavioral features (still need these from filtered data)
                gaze_data_filtered = self.behavioral_reader.read_annotated_csv(gaze_file, query_id)
                mouse_data_filtered = self.behavioral_reader.read_annotated_csv(mouse_file, query_id)
                gaze_features = self._extract_modality_features(gaze_data_filtered, response_length)
                mouse_features = self._extract_modality_features(mouse_data_filtered, response_length)
                
                # Extract phase features using full timeline
                gaze_phase = self.phase_extractor.extract_query_features(gaze_df, query_id, 'gaze')
                mouse_phase = self.phase_extractor.extract_query_features(mouse_df, query_id, 'mouse')
                cross_phase = self.phase_extractor.extract_cross_modality_features(
                    gaze_phase, mouse_phase, task_type='pointwise'
                )
                
                modality_features = ModalityFeatures(gaze=gaze_features, mouse=mouse_features)
                modality_features.phase_features = {**gaze_phase, **mouse_phase, **cross_phase}
                
                return modality_features
            
            else:
                # Pairwise: Read full files (like pointwise)
                gaze_data_full = self.behavioral_reader.read_full_csv(gaze_file)
                mouse_data_full = self.behavioral_reader.read_full_csv(mouse_file)
                
                # Convert to DataFrames
                gaze_df = pd.DataFrame([vars(d) for d in gaze_data_full]) if gaze_data_full else pd.DataFrame()
                mouse_df = pd.DataFrame([vars(d) for d in mouse_data_full]) if mouse_data_full else pd.DataFrame()
                
                # Still need filtered data for basic behavioral features
                gaze_data_filtered = self.behavioral_reader.read_annotated_csv(gaze_file, query_id)
                mouse_data_filtered = self.behavioral_reader.read_annotated_csv(mouse_file, query_id)
                gaze_features = self._extract_modality_features(gaze_data_filtered, response_length)
                mouse_features = self._extract_modality_features(mouse_data_filtered, response_length)
                
                # Store full DataFrames for pairwise phase extraction
                modality_features = ModalityFeatures(gaze=gaze_features, mouse=mouse_features)
                modality_features.gaze_df = gaze_df
                modality_features.mouse_df = mouse_df
                modality_features.response_num = response_num
                
                return modality_features
            
        except Exception as e:
            print(f"Error extracting response features: {e}")
            return None
    
    def _get_file_paths(self, user_id: str, task_id: str, 
                       comparison_type: str, response_num: int) -> Tuple[Path, Path]:
        """Get paths to gaze and mouse files for a response"""
        base_path = self.data_dir / user_id / task_id
        
        if comparison_type == 'pairwise':
            if response_num == 1:
                gaze_file = base_path / "rel_gaze_one_query_id_assigned.csv"
                mouse_file = base_path / "rel_mouse_left_query_id_assigned.csv"
            else:
                gaze_file = base_path / "rel_gaze_two_query_id_assigned.csv"
                mouse_file = base_path / "rel_mouse_right_query_id_assigned.csv"
        else:  # pointwise
            gaze_file = base_path / "rel_gaze_query_id_assigned.csv"
            mouse_file = base_path / "rel_mouse_query_id_assigned.csv"
        
        return gaze_file, mouse_file
    
    def _extract_modality_features(self, data_points: List, 
                                   response_length: int) -> ResponseFeatures:
        """Extract features for a single modality (gaze or mouse)"""
        if not data_points or response_length == 0:
            return ResponseFeatures(
                focused_engagement_ratio=0.0,
                focused_engagement_time=0.0,
                overall_attention_ratio=0.0,
                normalized_avg_char_position=0.0,
                reading_completion_ratio=0.0,
                normalized_char_position_variance=0.0,
                windowed_features=[0.0] * NUM_TIME_WINDOWS,
                response_length=response_length,
                data_points=0
            )

        # Sort by timestamp
        data_points = sorted(data_points, key=lambda x: x.rel_ts)

        # Separate looking vs not-looking
        looking_data = [d for d in data_points if d.is_looking_at_text()]

        # Calculate engagement metrics
        focused_engagement_ratio, focused_engagement_time = self._calculate_focused_engagement(looking_data)
        overall_attention = len(looking_data) / len(data_points) if data_points else 0.0
        
        # Calculate reading metrics from looking data
        if looking_data:
            char_positions = [d.centre_idx for d in looking_data]
            normalized_positions = [pos / response_length for pos in char_positions]
            
            avg_position = statistics.mean(normalized_positions)
            completion = min(1.0, max(char_positions) / response_length)
            variance = statistics.variance(normalized_positions) if len(normalized_positions) > 1 else 0.0
            windowed = self._calculate_windowed_features(looking_data, response_length)
        else:
            avg_position = 0.0
            completion = 0.0
            variance = 0.0
            windowed = [0.0] * NUM_TIME_WINDOWS
        
        return ResponseFeatures(
            focused_engagement_ratio=focused_engagement_ratio,
            focused_engagement_time=focused_engagement_time,
            overall_attention_ratio=overall_attention,
            normalized_avg_char_position=avg_position,
            reading_completion_ratio=completion,
            normalized_char_position_variance=variance,
            windowed_features=windowed,
            response_length=response_length,
            data_points=len(data_points)
        )
    
    def _calculate_focused_engagement(self, looking_data: List) -> Tuple[float, float]:
        """Calculate focused engagement ratio and active time (ms)"""
        if len(looking_data) <= 1:
            return 0.0, 0.0

        timestamps = sorted([d.rel_ts for d in looking_data])
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)]

        active_time = sum(min(interval, INACTIVITY_THRESHOLD_MS) for interval in intervals)
        session_time = max(timestamps) - min(timestamps)

        ratio = active_time / session_time if session_time > 0 else 0.0
        return ratio, active_time
    
    def _calculate_windowed_features(self, data_points: List, 
                                    response_length: int) -> List[float]:
        """Split session into time windows and calculate average position per window"""
        if not data_points or len(data_points) < 2:
            return [0.0] * NUM_TIME_WINDOWS
        
        data_points = sorted(data_points, key=lambda x: x.rel_ts)
        
        min_time = min(d.rel_ts for d in data_points)
        max_time = max(d.rel_ts for d in data_points)
        time_span = max_time - min_time
        
        if time_span == 0:
            return [0.0] * NUM_TIME_WINDOWS
        
        window_size = time_span / NUM_TIME_WINDOWS
        windowed = []
        
        for i in range(NUM_TIME_WINDOWS):
            window_start = min_time + i * window_size
            window_end = min_time + (i + 1) * window_size
            
            window_data = [d for d in data_points 
                          if window_start <= d.rel_ts < window_end]
            
            if window_data:
                normalized_positions = [d.centre_idx / response_length for d in window_data]
                windowed.append(statistics.mean(normalized_positions))
            else:
                windowed.append(0.0)
        
        return windowed
    
    @staticmethod
    def _parse_float(value) -> Optional[float]:
        """Safely parse float value"""
        try:
            return float(value) if value else None
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _parse_int(value) -> Optional[int]:
        """Safely parse int value"""
        try:
            return int(value) if value else None
        except (ValueError, TypeError):
            return None
