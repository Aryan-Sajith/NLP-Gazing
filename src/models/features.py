"""Feature data structures and models"""

from dataclasses import dataclass, field
from typing import List, Optional
from ..config.constants import NUM_TIME_WINDOWS

@dataclass
class ResponseFeatures:
    """Behavioral features extracted from interaction with a single response"""
    
    # Engagement metrics
    focused_engagement_ratio: float
    focused_engagement_time: float
    overall_attention_ratio: float
    
    # Reading pattern metrics
    normalized_avg_char_position: float
    reading_completion_ratio: float
    normalized_char_position_variance: float
    
    # Temporal features
    windowed_features: List[float] = field(default_factory=list)
    
    # Metadata
    response_length: int = 0
    data_points: int = 0


@dataclass
class ModalityFeatures:
    """Features from both gaze and mouse modalities"""
    gaze: ResponseFeatures
    mouse: ResponseFeatures


@dataclass
class ComparisonFeatures:
    """Complete feature set for a query (pairwise or pointwise)"""
    
    # Metadata
    comparison_type: str  # 'pairwise' or 'pointwise'
    query_id: int
    user_id: str
    task_id: str
    user_query: str
    
    # LLM information
    llm_name_1: Optional[str] = None
    llm_name_2: Optional[str] = None
    llm_response_1: Optional[str] = None
    llm_response_2: Optional[str] = None
    
    # Response A
    response_a: ModalityFeatures = None
    
    # Response B (None for pointwise)
    response_b: Optional[ModalityFeatures] = None
    
    # Target variables
    likert_1: Optional[float] = None
    likert_2: Optional[float] = None
    preference: Optional[int] = None
    
    def to_flat_dict(self) -> dict:
        """Convert to flat dictionary for CSV export"""
        result = {
            'comparison_type': self.comparison_type,
            'query_id': self.query_id,
            'user_id': self.user_id,
            'task_id': self.task_id,
            'user_query': self.user_query,
            'llm_name_1': self.llm_name_1,
            'llm_name_2': self.llm_name_2,
            'llm_response_1': self.llm_response_1,
            'llm_response_2': self.llm_response_2,
        }
        
        # Add Response A features
        result['user_query_length'] = len(self.user_query)
        self._add_modality_features(result, 'response_A', self.response_a)
        
        # Add Response B features (or None for pointwise)
        if self.response_b:
            self._add_modality_features(result, 'response_B', self.response_b)
        else:
            self._add_none_features(result, 'response_B')
        
        # Add target variables
        result['likert_1'] = self.likert_1
        result['likert_2'] = self.likert_2
        result['preference'] = self.preference
        
        if self.likert_1 is not None:
            result['normalized_likert_1'] = self.likert_1 / 5.0
        else:
            result['normalized_likert_1'] = None
            
        if self.likert_2 is not None:
            result['normalized_likert_2'] = self.likert_2 / 5.0
            result['binary_preference'] = 1 if self.preference == 2 else 0
        else:
            result['normalized_likert_2'] = None
            result['binary_preference'] = None
        
        # Add phase features if they exist
        if hasattr(self, 'phase_features') and self.phase_features:
            result.update(self.phase_features)
        
        return result
    
    def _add_modality_features(self, result: dict, prefix: str, modality: ModalityFeatures):
        """Add features from both gaze and mouse to result dict"""
        for mod_name, mod_features in [('gaze', modality.gaze), ('mouse', modality.mouse)]:
            result[f'{prefix}_{mod_name}_focused_engagement_ratio'] = mod_features.focused_engagement_ratio
            result[f'{prefix}_{mod_name}_focused_engagement_time'] = mod_features.focused_engagement_time
            result[f'{prefix}_{mod_name}_overall_attention_ratio'] = mod_features.overall_attention_ratio
            result[f'{prefix}_{mod_name}_normalized_avg_char_position'] = mod_features.normalized_avg_char_position
            result[f'{prefix}_{mod_name}_reading_completion_ratio'] = mod_features.reading_completion_ratio
            result[f'{prefix}_{mod_name}_normalized_char_position_variance'] = mod_features.normalized_char_position_variance
            result[f'{prefix}_{mod_name}_data_points'] = mod_features.data_points
            
            # Add windowed features
            for i, val in enumerate(mod_features.windowed_features):
                result[f'{prefix}_{mod_name}_window_{i:03d}'] = val
        
        # Add response length (same for both modalities)
        result[f'{prefix}_response_length'] = modality.gaze.response_length
    
    def _add_none_features(self, result: dict, prefix: str):
        """Add None placeholders for missing response"""
        for mod_name in ['gaze', 'mouse']:
            result[f'{prefix}_{mod_name}_focused_engagement_ratio'] = None
            result[f'{prefix}_{mod_name}_focused_engagement_time'] = None
            result[f'{prefix}_{mod_name}_overall_attention_ratio'] = None
            result[f'{prefix}_{mod_name}_normalized_avg_char_position'] = None
            result[f'{prefix}_{mod_name}_reading_completion_ratio'] = None
            result[f'{prefix}_{mod_name}_normalized_char_position_variance'] = None
            result[f'{prefix}_{mod_name}_data_points'] = None
            
            for i in range(NUM_TIME_WINDOWS):
                result[f'{prefix}_{mod_name}_window_{i:03d}'] = None
        
        result[f'{prefix}_response_length'] = None
