"""Detect reviewing and composing phase boundaries in query data"""

from typing import Tuple, Dict, Optional
import pandas as pd


class StageDetector:
    """
    Detects the boundary between reviewing and composing phases within a query.
    
    Uses a hybrid method combining character position plateau detection with
    actual reading behavior to identify when the user stopped reviewing the
    response and started composing the next question.
    """
    
    def __init__(self, 
                 plateau_threshold_pct: float = 0.90,
                 min_composing_duration_s: float = 2.0):
        """
        Initialize stage detector with parameters.
        
        Args:
            plateau_threshold_pct: What % of max character position = "finished reading"
            min_composing_duration_s: Minimum duration to consider as composing phase
        """
        self.plateau_threshold_pct = plateau_threshold_pct
        self.min_composing_duration_s = min_composing_duration_s
    
    def detect_boundary(self, query_data: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Detect the boundary between reviewing and composing phases.
        
        Args:
            query_data: DataFrame with behavioral data for a single query
            
        Returns:
            Tuple of (boundary_timestamp, metadata_dict)
            
        Algorithm:
            1. Find when user reached 90% through response (plateau)
            2. Find last actual reading activity after plateau
            3. Choose boundary based on time remaining after last reading
            4. Return boundary timestamp and metadata
        """
        if len(query_data) == 0:
            return None, {'error': 'empty_data'}
        
        query_data = query_data.copy().sort_values('rel_ts').reset_index(drop=True)
        
        start_time = query_data['rel_ts'].min()
        end_time = query_data['rel_ts'].max()
        total_duration_s = (end_time - start_time) / 1000
        
        # Get reading points (not looking off-screen)
        reading_points = query_data[~query_data['is_not_looking']].copy()
        
        if len(reading_points) == 0:
            # No reading detected - entire period is composing
            return start_time, {
                'method': 'no_reading_detected',
                'total_duration_s': total_duration_s,
                'reviewing_duration_s': 0,
                'composing_duration_s': total_duration_s,
                'composing_pct': 100.0
            }
        
        # Step 1: Find character position plateau
        max_char_pos = reading_points['centre_idx'].max()
        
        if max_char_pos <= 0:
            # Invalid character positions - use last reading time
            plateau_time = reading_points['rel_ts'].max()
        else:
            plateau_target = max_char_pos * self.plateau_threshold_pct
            reached_plateau = reading_points[reading_points['centre_idx'] >= plateau_target]
            
            if len(reached_plateau) > 0:
                # User reached 90% threshold
                plateau_time = reached_plateau['rel_ts'].min()
            else:
                # User never reached 90% - use when they reached maximum position
                # This is the fallback for incomplete reading
                max_pos_points = reading_points[reading_points['centre_idx'] == max_char_pos]
                plateau_time = max_pos_points['rel_ts'].min()
        
        # Step 2: Find last reading activity after plateau
        after_plateau = query_data[query_data['rel_ts'] >= plateau_time].copy()
        
        if len(after_plateau) == 0:
            # Reached plateau at the very end
            boundary_time = end_time
            method = 'plateau_at_end'
        else:
            after_plateau_reading = after_plateau[~after_plateau['is_not_looking']]
            
            if len(after_plateau_reading) == 0:
                # No reading after plateau
                boundary_time = plateau_time
                method = 'plateau_then_no_reading'
            else:
                # Find last reading point
                last_reading_time = after_plateau_reading['rel_ts'].max()
                time_after_reading = (end_time - last_reading_time) / 1000
                
                # Step 3: Choose boundary based on remaining time
                if time_after_reading >= self.min_composing_duration_s:
                    boundary_time = last_reading_time
                    method = 'last_reading_after_plateau'
                else:
                    boundary_time = plateau_time
                    method = 'plateau_preferred'
        
        # Calculate metadata
        reviewing_duration_s = (boundary_time - start_time) / 1000
        composing_duration_s = (end_time - boundary_time) / 1000
        
        metadata = {
            'method': method,
            'total_duration_s': total_duration_s,
            'reviewing_duration_s': reviewing_duration_s,
            'composing_duration_s': composing_duration_s,
            'composing_pct': (composing_duration_s / total_duration_s * 100) if total_duration_s > 0 else 0,
            'plateau_time_pct': ((plateau_time - start_time) / (end_time - start_time) * 100) if end_time > start_time else 0,
            'max_char_position': max_char_pos,
            'total_reading_points': len(reading_points),
            'reading_ratio': len(reading_points) / len(query_data)
        }
        
        return boundary_time, metadata
    
    def detect_pairwise_boundary(self, left_data: pd.DataFrame, right_data: pd.DataFrame,
                                 left_response_length: int, right_response_length: int) -> Tuple[float, Dict]:
        """
        Detect boundary for pairwise tasks by considering both responses together.
        
        For pairwise, user reviews BOTH responses simultaneously, then composes for both.
        We find when they stopped reading EITHER response.
        
        Args:
            left_data: DataFrame for left response
            right_data: DataFrame for right response  
            left_response_length: Character length of left response
            right_response_length: Character length of right response
            
        Returns:
            Tuple of (boundary_timestamp, metadata_dict)
        """
        # Tag data to distinguish left from right before merging
        left_tagged = left_data.copy()
        left_tagged['_side'] = 'left'
        right_tagged = right_data.copy()
        right_tagged['_side'] = 'right'
        
        # Merge both datasets
        merged_data = pd.concat([left_tagged, right_tagged], ignore_index=True).sort_values('rel_ts')
        
        if len(merged_data) == 0:
            return None, {'error': 'empty_data'}
        
        start_time = merged_data['rel_ts'].min()
        end_time = merged_data['rel_ts'].max()
        total_duration_s = (end_time - start_time) / 1000
        
        # Get reading points from merged data
        reading_points = merged_data[~merged_data['is_not_looking']].copy()
        
        if len(reading_points) == 0:
            return start_time, {
                'method': 'no_reading_detected',
                'total_duration_s': total_duration_s,
                'reviewing_duration_s': 0,
                'composing_duration_s': total_duration_s,
                'composing_pct': 100.0
            }
        
        # Step 1: Find plateau for pairwise - check both responses separately
        left_reading = reading_points[reading_points['_side'] == 'left'].copy() if len(left_data) > 0 else pd.DataFrame()
        right_reading = reading_points[reading_points['_side'] == 'right'].copy() if len(right_data) > 0 else pd.DataFrame()
        
        # Calculate 90% thresholds
        left_plateau_target = left_response_length * self.plateau_threshold_pct
        right_plateau_target = right_response_length * self.plateau_threshold_pct
        
        left_plateau_time = None
        right_plateau_time = None
        
        # Check if left response reached 90%
        if len(left_reading) > 0:
            max_left_pos = left_reading['centre_idx'].max()
            if max_left_pos >= left_plateau_target:
                left_reached = left_reading[left_reading['centre_idx'] >= left_plateau_target]
                left_plateau_time = left_reached['rel_ts'].min() if len(left_reached) > 0 else None
            elif max_left_pos > 0:
                # Never reached 90% - use max position reached
                max_left_points = left_reading[left_reading['centre_idx'] == max_left_pos]
                left_plateau_time = max_left_points['rel_ts'].min()
        
        # Check if right response reached 90%
        if len(right_reading) > 0:
            max_right_pos = right_reading['centre_idx'].max()
            if max_right_pos >= right_plateau_target:
                right_reached = right_reading[right_reading['centre_idx'] >= right_plateau_target]
                right_plateau_time = right_reached['rel_ts'].min() if len(right_reached) > 0 else None
            elif max_right_pos > 0:
                # Never reached 90% - use max position reached  
                max_right_points = right_reading[right_reading['centre_idx'] == max_right_pos]
                right_plateau_time = max_right_points['rel_ts'].min()
        
        # Choose plateau time: use LAST timestamp (both finished) or whichever exists
        if left_plateau_time is not None and right_plateau_time is not None:
            plateau_time = max(left_plateau_time, right_plateau_time)
        elif left_plateau_time is not None:
            plateau_time = left_plateau_time
        elif right_plateau_time is not None:
            plateau_time = right_plateau_time
        else:
            # Neither response had any reading - use last reading time
            plateau_time = reading_points['rel_ts'].max()
        
        # Step 2: Find last reading activity after plateau (on merged data)
        after_plateau = merged_data[merged_data['rel_ts'] >= plateau_time].copy()
        
        if len(after_plateau) == 0:
            boundary_time = end_time
            method = 'plateau_at_end'
        else:
            after_plateau_reading = after_plateau[~after_plateau['is_not_looking']]
            
            if len(after_plateau_reading) == 0:
                boundary_time = plateau_time
                method = 'plateau_then_no_reading'
            else:
                last_reading_time = after_plateau_reading['rel_ts'].max()
                time_after_reading = (end_time - last_reading_time) / 1000
                
                if time_after_reading >= self.min_composing_duration_s:
                    boundary_time = last_reading_time
                    method = 'last_reading_after_plateau'
                else:
                    boundary_time = plateau_time
                    method = 'plateau_preferred'
        
        # Calculate metadata
        reviewing_duration_s = (boundary_time - start_time) / 1000
        composing_duration_s = (end_time - boundary_time) / 1000
        
        metadata = {
            'method': method,
            'total_duration_s': total_duration_s,
            'reviewing_duration_s': reviewing_duration_s,
            'composing_duration_s': composing_duration_s,
            'composing_pct': (composing_duration_s / total_duration_s * 100) if total_duration_s > 0 else 0,
            'plateau_time_pct': ((plateau_time - start_time) / (end_time - start_time) * 100) if end_time > start_time else 0,
            'total_reading_points': len(reading_points),
            'reading_ratio': len(reading_points) / len(merged_data)
        }
        
        return boundary_time, metadata
    
    def split_phases(self, query_data: pd.DataFrame, boundary_time: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split query data into reviewing and composing phases.
        
        Args:
            query_data: DataFrame with behavioral data
            boundary_time: Timestamp marking the phase boundary
            
        Returns:
            Tuple of (reviewing_data, composing_data)
        """
        query_data = query_data.sort_values('rel_ts')
        reviewing_data = query_data[query_data['rel_ts'] < boundary_time].copy()
        composing_data = query_data[query_data['rel_ts'] >= boundary_time].copy()
        
        return reviewing_data, composing_data
