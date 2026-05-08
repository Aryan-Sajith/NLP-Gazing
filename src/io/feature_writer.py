"""Write feature data to CSV files"""

import csv
from pathlib import Path
from typing import List, Optional
from ..config.constants import NUM_TIME_WINDOWS

class FeatureWriter:
    """Writes extracted features to CSV"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
    
    def write_features(self, features: List[dict], headers: Optional[List[str]] = None):
        """
        Write feature vectors to CSV.

        Args:
            features: List of feature dictionaries
            headers: Ignored; column names are derived from the feature dicts
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Derive column names from the actual keys present in the feature dicts
            fieldnames = list(dict.fromkeys(k for f in features for k in f))

            with open(self.output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()

                for feature_dict in features:
                    writer.writerow(feature_dict)

            print(f"Features written to {self.output_path}")
            
        except Exception as e:
            print(f"Error writing features: {e}")
    
    @staticmethod
    def get_feature_headers() -> List[str]:
        """Generate standard CSV headers for feature vectors"""
        headers = [
            # Metadata
            'comparison_type', 'query_id', 'user_id', 'task_id', 'user_query',
            
            # LLM information
            'llm_name_1', 'llm_name_2', 'llm_response_1', 'llm_response_2',
            
            # Target variables
            'likert_1', 'likert_2', 'preference', 
            'normalized_likert_1', 'normalized_likert_2', 'binary_preference',
        ]
        
        # Response features for A and B
        for response in ['A', 'B']:
            for modality in ['gaze', 'mouse']:
                prefix = f'response_{response}_{modality}'
                headers.extend([
                    f'{prefix}_focused_engagement_ratio',
                    f'{prefix}_focused_engagement_time',
                    f'{prefix}_overall_attention_ratio',
                    f'{prefix}_normalized_avg_char_position',
                    f'{prefix}_reading_completion_ratio',
                    f'{prefix}_normalized_char_position_variance',
                    f'{prefix}_data_points',
                ])
            
            # Response length (same for both modalities)
            headers.append(f'response_{response}_response_length')
        
        # Windowed features
        for response in ['A', 'B']:
            for modality in ['gaze', 'mouse']:
                for i in range(NUM_TIME_WINDOWS):
                    headers.append(f'response_{response}_{modality}_window_{i:03d}')
        
        # Pointwise phase features (for each modality)
        for modality in ['gaze', 'mouse']:
            prefix = f'{modality}_'
            headers.extend([
                # Phase timing (6)
                f'{prefix}reviewing_duration_s',
                f'{prefix}composing_duration_s',
                f'{prefix}reviewing_pct',
                f'{prefix}composing_pct',
                f'{prefix}detection_method',
                f'{prefix}max_char_position_reached',
                # Activity ratios (6)
                f'{prefix}reviewing_active_ratio',
                f'{prefix}reviewing_offscreen_ratio',
                f'{prefix}composing_active_ratio',
                f'{prefix}composing_offscreen_ratio',
                f'{prefix}composing_lookback_ratio',
                f'{prefix}composing_thinking_ratio',
                # Comparison features (10)
                f'{prefix}reviewing_composing_duration_ratio',
                f'{prefix}reviewing_composing_activity_ratio',
                f'{prefix}composing_reviewing_activity_diff',
                f'{prefix}offscreen_increase',
                f'{prefix}active_time_reviewing_s',
                f'{prefix}active_time_composing_s',
                f'{prefix}offscreen_time_reviewing_s',
                f'{prefix}offscreen_time_composing_s',
                f'{prefix}lookback_time_s',
                f'{prefix}thinking_time_s',
            ])
        
        # Pairwise phase features
        for modality in ['gaze', 'mouse']:
            # Per-side features (left and right)
            for side in ['left', 'right']:
                prefix = f'{modality}_{side}_'
                headers.extend([
                    f'{prefix}reviewing_engaged_time_s',
                    f'{prefix}reviewing_engaged_pct',
                    f'{prefix}reviewing_active_ratio',
                    f'{prefix}reviewing_offscreen_ratio',
                    f'{prefix}reviewing_onscreen_ratio',
                    f'{prefix}max_char_position_reached',
                    f'{prefix}composing_lookback_time_s',
                    f'{prefix}composing_lookback_ratio',
                ])
            
            # Global composing features
            headers.extend([
                f'{modality}_composing_duration_s',
                f'{modality}_composing_pct',
                f'{modality}_detection_method',
            ])
            
            # Comparison features
            prefix = f'{modality}_comparison_'
            headers.extend([
                f'{prefix}reviewing_time_ratio',
                f'{prefix}reviewing_time_diff',
                f'{prefix}which_side_longer_reviewing',
                f'{prefix}reviewing_activity_ratio',
                f'{prefix}reviewing_activity_diff',
                f'{prefix}which_side_more_active_reviewing',
                f'{prefix}composing_lookback_ratio',
                f'{prefix}composing_lookback_diff',
                f'{prefix}which_side_read_further',
            ])
        
        # Cross-modality features
        prefix = 'cross_modality_'
        # Pointwise cross-modality
        headers.extend([
            f'{prefix}reviewing_duration_ratio_gaze_mouse',
            f'{prefix}composing_duration_ratio_gaze_mouse',
            f'{prefix}reviewing_activity_ratio_gaze_mouse',
            f'{prefix}composing_activity_ratio_gaze_mouse',
        ])
        # Pairwise cross-modality
        headers.extend([
            f'{prefix}left_reviewing_duration_ratio_gaze_mouse',
            f'{prefix}right_reviewing_duration_ratio_gaze_mouse',
            f'{prefix}preference_agreement_gaze_mouse',
        ])
        
        return headers
