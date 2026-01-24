"""Write feature data to CSV files"""

import csv
from pathlib import Path
from typing import List


class FeatureWriter:
    """Writes extracted features to CSV"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
    
    def write_features(self, features: List[dict], headers: List[str]):
        """
        Write feature vectors to CSV.
        
        Args:
            features: List of feature dictionaries
            headers: Column headers
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
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
                for i in range(100):
                    headers.append(f'response_{response}_{modality}_window_{i:03d}')
        
        return headers
