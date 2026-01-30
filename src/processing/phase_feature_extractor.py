"""Extract features based on reviewing and composing phases"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from .stage_detector import StageDetector


class PhaseFeatureExtractor:
    """
    Extract behavioral features separately for reviewing and composing phases.
    
    This extractor identifies the boundary between phases using StageDetector,
    then computes phase-specific and comparative features.
    """
    
    # Constants for safe ratio calculation
    EPSILON = 0.001  # Small value to avoid division by zero
    MAX_RATIO = 100.0  # Cap ratios to avoid extreme values
    
    def __init__(self, plateau_threshold_pct: float = 0.90, min_composing_duration_s: float = 2.0):
        """Initialize with stage detection parameters."""
        self.stage_detector = StageDetector(plateau_threshold_pct, min_composing_duration_s)
    
    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        """
        Compute ratio with safeguards against division by zero and extreme values.
        
        Args:
            numerator: Top value
            denominator: Bottom value
            
        Returns:
            Capped ratio, max of MAX_RATIO if denominator is very small
        """
        return min(numerator / (denominator + self.EPSILON), self.MAX_RATIO)
    
    def extract_pointwise_features(self, query_data: pd.DataFrame, modality: str) -> Dict[str, float]:
        """
        Extract phase features for pointwise tasks (single LLM response).
        
        Args:
            query_data: DataFrame with behavioral data
            modality: 'gaze' or 'mouse'
            
        Returns:
            Dictionary of features with prefix {modality}_
        """
        features = {}
        prefix = f"{modality}_"
        
        # Detect phase boundary
        boundary_time, metadata = self.stage_detector.detect_boundary(query_data)
        
        if boundary_time is None:
            # Return empty features on error
            return self._empty_pointwise_features(prefix)
        
        # Split into phases
        reviewing_data, composing_data = self.stage_detector.split_phases(query_data, boundary_time)
        
        # Phase timing features (8)
        features[f"{prefix}reviewing_duration_s"] = metadata['reviewing_duration_s']
        features[f"{prefix}composing_duration_s"] = metadata['composing_duration_s']
        features[f"{prefix}reviewing_pct"] = (metadata['reviewing_duration_s'] / metadata['total_duration_s'] * 100) if metadata['total_duration_s'] > 0 else 0
        features[f"{prefix}composing_pct"] = metadata['composing_pct']
        features[f"{prefix}plateau_time_pct"] = metadata.get('plateau_time_pct', 0)
        features[f"{prefix}time_after_plateau_s"] = metadata['composing_duration_s']
        features[f"{prefix}detection_method"] = self._encode_method(metadata['method'])
        features[f"{prefix}max_char_position_reached"] = metadata.get('max_char_position', 0)
        
        # Activity ratio features (6)
        if len(reviewing_data) > 0:
            features[f"{prefix}reviewing_active_ratio"] = (~reviewing_data['is_not_looking']).mean()
            features[f"{prefix}reviewing_offscreen_ratio"] = reviewing_data['is_not_looking'].mean()
        else:
            features[f"{prefix}reviewing_active_ratio"] = 0
            features[f"{prefix}reviewing_offscreen_ratio"] = 0
        
        if len(composing_data) > 0:
            features[f"{prefix}composing_active_ratio"] = (~composing_data['is_not_looking']).mean()
            features[f"{prefix}composing_offscreen_ratio"] = composing_data['is_not_looking'].mean()
            
            # Lookback = looking at experimental text during composing
            features[f"{prefix}composing_lookback_ratio"] = (
                composing_data['is_experimental_text'].sum() / len(composing_data)
            ) if len(composing_data) > 0 else 0
            
            # "Thinking" = looking at screen but not at experimental text
            thinking_time = composing_data[
                ~composing_data['is_not_looking'] & ~composing_data['is_experimental_text']
            ]
            features[f"{prefix}composing_thinking_ratio"] = len(thinking_time) / len(composing_data) if len(composing_data) > 0 else 0
        else:
            features[f"{prefix}composing_active_ratio"] = 0
            features[f"{prefix}composing_offscreen_ratio"] = 0
            features[f"{prefix}composing_lookback_ratio"] = 0
            features[f"{prefix}composing_thinking_ratio"] = 0
        
        # Comparison features (10)
        features[f"{prefix}reviewing_composing_duration_ratio"] = self._safe_ratio(
            metadata['reviewing_duration_s'], metadata['composing_duration_s']
        )
        
        reviewing_active = features[f"{prefix}reviewing_active_ratio"]
        composing_active = features[f"{prefix}composing_active_ratio"]
        features[f"{prefix}reviewing_composing_activity_ratio"] = self._safe_ratio(
            reviewing_active, composing_active
        )
        
        features[f"{prefix}composing_reviewing_activity_diff"] = composing_active - reviewing_active
        
        reviewing_offscreen = features[f"{prefix}reviewing_offscreen_ratio"]
        composing_offscreen = features[f"{prefix}composing_offscreen_ratio"]
        features[f"{prefix}offscreen_increase"] = composing_offscreen - reviewing_offscreen
        
        features[f"{prefix}active_time_reviewing_s"] = metadata['reviewing_duration_s'] * reviewing_active
        features[f"{prefix}active_time_composing_s"] = metadata['composing_duration_s'] * composing_active
        
        features[f"{prefix}offscreen_time_reviewing_s"] = metadata['reviewing_duration_s'] * reviewing_offscreen
        features[f"{prefix}offscreen_time_composing_s"] = metadata['composing_duration_s'] * composing_offscreen
        
        features[f"{prefix}lookback_time_s"] = metadata['composing_duration_s'] * features[f"{prefix}composing_lookback_ratio"]
        features[f"{prefix}thinking_time_s"] = metadata['composing_duration_s'] * features[f"{prefix}composing_thinking_ratio"]
        
        return features
    
    def extract_pairwise_features(self, 
                                  left_data: pd.DataFrame, 
                                  right_data: pd.DataFrame,
                                  modality: str,
                                  left_response_length: int,
                                  right_response_length: int) -> Dict[str, float]:
        """
        Extract phase features for pairwise tasks (comparing two LLM responses).
        
        For pairwise, user reviews BOTH responses together, then composes.
        We detect ONE boundary that applies to both left and right.
        All times are relative to the global comparison timeline.
        
        Args:
            left_data: DataFrame for left response
            right_data: DataFrame for right response
            modality: 'gaze' or 'mouse'
            left_response_length: Character length of left response
            right_response_length: Character length of right response
            
        Returns:
            Dictionary of features with prefixes {modality}_left_, {modality}_right_, {modality}_comparison_
        """
        features = {}
        
        # Detect SINGLE boundary using merged data
        boundary_time, metadata = self.stage_detector.detect_pairwise_boundary(
            left_data, right_data, left_response_length, right_response_length
        )
        
        if boundary_time is None:
            return self._empty_pairwise_features(modality)
        
        # Global timeline (same for both sides)
        merged_data = pd.concat([left_data, right_data], ignore_index=True).sort_values('rel_ts')
        global_start = merged_data['rel_ts'].min()
        global_end = merged_data['rel_ts'].max()
        total_duration_s = metadata['total_duration_s']
        reviewing_duration_s = metadata['reviewing_duration_s']
        composing_duration_s = metadata['composing_duration_s']
        
        # Split BOTH left and right at the SAME boundary
        left_reviewing, left_composing = self.stage_detector.split_phases(left_data, boundary_time)
        right_reviewing, right_composing = self.stage_detector.split_phases(right_data, boundary_time)
        
        # Extract per-side features during reviewing phase
        left_features = self._extract_pairwise_side_features(
            left_reviewing, left_composing, 
            total_duration_s, reviewing_duration_s, composing_duration_s,
            f"{modality}_left"
        )
        features.update(left_features)
        
        right_features = self._extract_pairwise_side_features(
            right_reviewing, right_composing,
            total_duration_s, reviewing_duration_s, composing_duration_s,
            f"{modality}_right"
        )
        features.update(right_features)
        
        # Global composing features (shared, not per-side)
        features[f"{modality}_composing_duration_s"] = composing_duration_s
        features[f"{modality}_composing_pct"] = metadata['composing_pct']
        features[f"{modality}_detection_method"] = self._encode_method(metadata['method'])
        features[f"{modality}_plateau_time_pct"] = metadata.get('plateau_time_pct', 0)
        
        # Comparison features between left and right
        comparison_features = self._extract_pairwise_comparison_features(
            left_features, right_features, f"{modality}"
        )
        features.update(comparison_features)
        
        return features
    
    def _extract_pairwise_side_features(self, reviewing_data: pd.DataFrame, composing_data: pd.DataFrame,
                                        total_duration_s: float, reviewing_duration_s: float, 
                                        composing_duration_s: float, prefix: str) -> Dict[str, float]:
        """
        Extract features for one side of pairwise comparison.
        
        Calculates actual engaged time on this side during the shared reviewing/composing phases.
        All percentages are relative to GLOBAL total time.
        
        Args:
            reviewing_data: Data points on this side during reviewing phase
            composing_data: Data points on this side during composing phase
            total_duration_s: Global total comparison duration
            reviewing_duration_s: Global reviewing phase duration
            composing_duration_s: Global composing phase duration
            prefix: Feature name prefix (e.g., 'gaze_left')
            
        Returns:
            Dictionary of features for this side
        """
        from src.config.constants import INACTIVITY_THRESHOLD_MS
        
        features = {}
        
        # Calculate actual engaged time on this side during reviewing
        reviewing_looking = reviewing_data[~reviewing_data['is_not_looking']].copy() if len(reviewing_data) > 0 else pd.DataFrame()
        
        if len(reviewing_looking) > 1:
            # Calculate engaged time using windowing (sum of active intervals)
            timestamps = reviewing_looking['rel_ts'].values
            intervals = [(timestamps[i+1] - timestamps[i]) for i in range(len(timestamps) - 1)]
            engaged_time_ms = sum(min(interval, INACTIVITY_THRESHOLD_MS) for interval in intervals)
            engaged_time_s = engaged_time_ms / 1000
        elif len(reviewing_looking) == 1:
            engaged_time_s = 0
        else:
            engaged_time_s = 0
        
        # Reviewing features
        features[f"{prefix}_reviewing_engaged_time_s"] = engaged_time_s
        features[f"{prefix}_reviewing_engaged_pct"] = (engaged_time_s / total_duration_s * 100) if total_duration_s > 0 else 0
        features[f"{prefix}_reviewing_active_ratio"] = (engaged_time_s / reviewing_duration_s) if reviewing_duration_s > 0 else 0
        features[f"{prefix}_reviewing_offscreen_ratio"] = reviewing_data['is_not_looking'].mean() if len(reviewing_data) > 0 else 0
        
        # Max character position reached on this side
        if len(reviewing_looking) > 0:
            features[f"{prefix}_max_char_position_reached"] = reviewing_looking['centre_idx'].max()
        else:
            features[f"{prefix}_max_char_position_reached"] = 0
        
        # Composing lookback features (per-side)
        composing_looking = composing_data[~composing_data['is_not_looking']].copy() if len(composing_data) > 0 else pd.DataFrame()
        composing_looking_at_text = composing_data[composing_data['is_experimental_text']].copy() if len(composing_data) > 0 else pd.DataFrame()
        
        if len(composing_looking_at_text) > 1:
            timestamps = composing_looking_at_text['rel_ts'].values
            intervals = [(timestamps[i+1] - timestamps[i]) for i in range(len(timestamps) - 1)]
            lookback_time_ms = sum(min(interval, INACTIVITY_THRESHOLD_MS) for interval in intervals)
            lookback_time_s = lookback_time_ms / 1000
        else:
            lookback_time_s = 0
        
        features[f"{prefix}_composing_lookback_time_s"] = lookback_time_s
        features[f"{prefix}_composing_lookback_ratio"] = (lookback_time_s / composing_duration_s) if composing_duration_s > 0 else 0
        
        return features
    
    def _extract_pairwise_comparison_features(self, left_features: Dict, right_features: Dict, 
                                             modality: str) -> Dict[str, float]:
        """
        Extract comparison features between left and right sides.
        
        Args:
            left_features: Features for left side
            right_features: Features for right side
            modality: 'gaze' or 'mouse'
            
        Returns:
            Dictionary of comparison features
        """
        features = {}
        prefix = f"{modality}_comparison_"
        
        # Engaged time comparisons during reviewing
        left_engaged = left_features[f"{modality}_left_reviewing_engaged_time_s"]
        right_engaged = right_features[f"{modality}_right_reviewing_engaged_time_s"]
        
        features[f"{prefix}reviewing_time_ratio"] = self._safe_ratio(left_engaged, right_engaged)
        features[f"{prefix}reviewing_time_diff"] = left_engaged - right_engaged
        features[f"{prefix}which_side_longer_reviewing"] = 1 if left_engaged > right_engaged else -1
        
        # Active ratio comparisons
        left_active = left_features[f"{modality}_left_reviewing_active_ratio"]
        right_active = right_features[f"{modality}_right_reviewing_active_ratio"]
        
        features[f"{prefix}reviewing_activity_ratio"] = self._safe_ratio(left_active, right_active)
        features[f"{prefix}reviewing_activity_diff"] = left_active - right_active
        features[f"{prefix}which_side_more_active_reviewing"] = 1 if left_active > right_active else -1
        
        # Lookback comparisons during composing
        left_lookback = left_features[f"{modality}_left_composing_lookback_time_s"]
        right_lookback = right_features[f"{modality}_right_composing_lookback_time_s"]
        
        features[f"{prefix}composing_lookback_ratio"] = self._safe_ratio(left_lookback, right_lookback)
        features[f"{prefix}composing_lookback_diff"] = left_lookback - right_lookback
        
        # Character position comparison
        left_max_char = left_features[f"{modality}_left_max_char_position_reached"]
        right_max_char = right_features[f"{modality}_right_max_char_position_reached"]
        
        features[f"{prefix}which_side_read_further"] = 1 if left_max_char > right_max_char else -1
        
        return features
    
    def extract_cross_modality_features(self,
                                       gaze_features: Dict[str, float],
                                       mouse_features: Dict[str, float],
                                       task_type: str = 'pointwise') -> Dict[str, float]:
        """
        Extract features comparing gaze and mouse behavior across phases.
        
        Args:
            gaze_features: Features from gaze modality
            mouse_features: Features from mouse modality
            task_type: 'pointwise' or 'pairwise'
            
        Returns:
            Dictionary of cross-modality features
        """
        features = {}
        prefix = "cross_modality_"
        
        if task_type == 'pointwise':
            # Reviewing phase correlation
            gaze_rev_dur = gaze_features.get('gaze_reviewing_duration_s', 0)
            mouse_rev_dur = mouse_features.get('mouse_reviewing_duration_s', 0)
            features[f"{prefix}reviewing_duration_ratio_gaze_mouse"] = self._safe_ratio(
                gaze_rev_dur, mouse_rev_dur
            )
            
            # Composing phase correlation
            gaze_comp_dur = gaze_features.get('gaze_composing_duration_s', 0)
            mouse_comp_dur = mouse_features.get('mouse_composing_duration_s', 0)
            features[f"{prefix}composing_duration_ratio_gaze_mouse"] = self._safe_ratio(
                gaze_comp_dur, mouse_comp_dur
            )
            
            # Activity correlation
            gaze_rev_active = gaze_features.get('gaze_reviewing_active_ratio', 0)
            mouse_rev_active = mouse_features.get('mouse_reviewing_active_ratio', 0)
            features[f"{prefix}reviewing_activity_ratio_gaze_mouse"] = self._safe_ratio(
                gaze_rev_active, mouse_rev_active
            )
            
            gaze_comp_active = gaze_features.get('gaze_composing_active_ratio', 0)
            mouse_comp_active = mouse_features.get('mouse_composing_active_ratio', 0)
            features[f"{prefix}composing_activity_ratio_gaze_mouse"] = self._safe_ratio(
                gaze_comp_active, mouse_comp_active
            )
        
        else:  # pairwise
            # Left side gaze/mouse correlation
            gaze_left_rev = gaze_features.get('gaze_left_reviewing_engaged_time_s', 0)
            mouse_left_rev = mouse_features.get('mouse_left_reviewing_engaged_time_s', 0)
            features[f"{prefix}left_reviewing_duration_ratio_gaze_mouse"] = self._safe_ratio(
                gaze_left_rev, mouse_left_rev
            )
            
            # Right side gaze/mouse correlation
            gaze_right_rev = gaze_features.get('gaze_right_reviewing_engaged_time_s', 0)
            mouse_right_rev = mouse_features.get('mouse_right_reviewing_engaged_time_s', 0)
            features[f"{prefix}right_reviewing_duration_ratio_gaze_mouse"] = self._safe_ratio(
                gaze_right_rev, mouse_right_rev
            )
            
            # Which modality shows stronger left preference
            gaze_pref = gaze_features.get('gaze_comparison_reviewing_duration_ratio', 1)
            mouse_pref = mouse_features.get('mouse_comparison_reviewing_duration_ratio', 1)
            features[f"{prefix}preference_agreement_gaze_mouse"] = (
                1 if (gaze_pref > 1 and mouse_pref > 1) or (gaze_pref < 1 and mouse_pref < 1) else -1
            )
        
        return features
    
    def _encode_method(self, method: str) -> int:
        """Encode detection method as numeric feature."""
        method_encoding = {
            'last_reading_after_plateau': 1,
            'plateau_preferred': 2,
            'plateau_then_no_reading': 3,
            'plateau_at_end': 4,
            'no_reading_detected': 5
        }
        return method_encoding.get(method, 0)
    
    def _empty_pointwise_features(self, prefix: str) -> Dict[str, float]:
        """Return empty feature dict for error cases."""
        return {
            f"{prefix}reviewing_duration_s": 0,
            f"{prefix}composing_duration_s": 0,
            f"{prefix}reviewing_pct": 0,
            f"{prefix}composing_pct": 0,
            f"{prefix}plateau_time_pct": 0,
            f"{prefix}time_after_plateau_s": 0,
            f"{prefix}detection_method": 0,
            f"{prefix}max_char_position_reached": 0,
            f"{prefix}reviewing_active_ratio": 0,
            f"{prefix}reviewing_offscreen_ratio": 0,
            f"{prefix}composing_active_ratio": 0,
            f"{prefix}composing_offscreen_ratio": 0,
            f"{prefix}composing_lookback_ratio": 0,
            f"{prefix}composing_thinking_ratio": 0,
            f"{prefix}reviewing_composing_duration_ratio": 0,
            f"{prefix}reviewing_composing_activity_ratio": 0,
            f"{prefix}composing_reviewing_activity_diff": 0,
            f"{prefix}offscreen_increase": 0,
            f"{prefix}active_time_reviewing_s": 0,
            f"{prefix}active_time_composing_s": 0,
            f"{prefix}offscreen_time_reviewing_s": 0,
            f"{prefix}offscreen_time_composing_s": 0,
            f"{prefix}lookback_time_s": 0,
            f"{prefix}thinking_time_s": 0
        }
    
    def _empty_pairwise_features(self, modality: str) -> Dict[str, float]:
        """Return empty feature dict for pairwise error cases."""
        features = {}
        
        # Per-side features (left and right)
        for side in ['left', 'right']:
            prefix = f"{modality}_{side}_"
            features.update({
                f"{prefix}reviewing_engaged_time_s": 0,
                f"{prefix}reviewing_engaged_pct": 0,
                f"{prefix}reviewing_active_ratio": 0,
                f"{prefix}reviewing_offscreen_ratio": 0,
                f"{prefix}max_char_position_reached": 0,
                f"{prefix}composing_lookback_time_s": 0,
                f"{prefix}composing_lookback_ratio": 0
            })
        
        # Global composing features
        features.update({
            f"{modality}_composing_duration_s": 0,
            f"{modality}_composing_pct": 0,
            f"{modality}_detection_method": 0,
            f"{modality}_plateau_time_pct": 0
        })
        
        # Comparison features
        prefix = f"{modality}_comparison_"
        features.update({
            f"{prefix}reviewing_time_ratio": 0,
            f"{prefix}reviewing_time_diff": 0,
            f"{prefix}which_side_longer_reviewing": 0,
            f"{prefix}reviewing_activity_ratio": 0,
            f"{prefix}reviewing_activity_diff": 0,
            f"{prefix}which_side_more_active_reviewing": 0,
            f"{prefix}composing_lookback_ratio": 0,
            f"{prefix}composing_lookback_diff": 0,
            f"{prefix}which_side_read_further": 0
        })
        
        return features
