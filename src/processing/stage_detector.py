"""Detect reviewing and composing phase boundaries based on query transitions"""

from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np


class StageDetector:
    """
    Detects reviewing and composing phases by analyzing the full timeline of query responses.
    
    New approach (Feb 2026):
    - Works on FULL file without filtering by query_id
    - Identifies query sequence based on order of appearance
    - Composing phase = time between responses (or before first response)
    - Reviewing phase = time when actively looking at response text
    - Boundaries determined by valid centre_idx timestamps
    """
    
    def __init__(self):
        """Initialize stage detector."""
        pass
    
    def detect_query_phases(self, data: pd.DataFrame) -> Dict[int, Dict]:
        """
        Detect reviewing and composing phases for all queries in the full timeline.
        
        Args:
            data: Full DataFrame with all behavioral data (NOT filtered by query_id)
                  Must have columns: rel_ts, query_id, centre_idx
        
        Returns:
            Dictionary mapping query_id -> phase info:
            {
                query_id: {
                    'composing_start': timestamp,
                    'composing_end': timestamp,
                    'reviewing_start': timestamp,
                    'reviewing_end': timestamp,
                    'query_order': int (0-based position in sequence)
                }
            }
        """
        if len(data) == 0 or 'query_id' not in data.columns:
            return {}
        
        data = data.copy().sort_values('rel_ts').reset_index(drop=True)
        
        # Find all valid query IDs (> 0) in order of first appearance
        valid_queries = data[data['query_id'] > 0].copy()
        if len(valid_queries) == 0:
            return {}
        
        # Get query sequence in order of first appearance
        query_sequence = valid_queries.groupby('query_id')['rel_ts'].min().sort_values().index.tolist()
        
        # File boundaries
        file_start = data['rel_ts'].min()
        file_end = data['rel_ts'].max()
        
        # Build phase info for each query
        phases = {}
        
        for idx, query_id in enumerate(query_sequence):
            phase_info = {'query_order': idx}
            
            # Find first and last valid centre_idx for this query
            query_data = data[data['query_id'] == query_id].copy()
            valid_reading = query_data[
                (query_data['centre_idx'].notna()) & 
                (query_data['centre_idx'] != -1)
            ]
            
            if len(valid_reading) == 0:
                # No valid reading for this query - skip it
                continue
            
            first_reading_ts = valid_reading['rel_ts'].min()
            last_reading_ts = valid_reading['rel_ts'].max()
            
            # COMPOSING PHASE
            if idx == 0:
                # First query: composing starts at file start
                phase_info['composing_start'] = file_start
            else:
                # Subsequent queries: composing starts at last valid centre_idx of previous query
                # that occurred BEFORE this query's first reading. The boundary between queries
                # is fuzzy in the raw data — stray readings near the transition can be labeled
                # with the previous query_id even after the current query has started. Without
                # this filter, composing_start could be pushed past composing_end, producing a
                # negative composing duration.
                prev_query_id = query_sequence[idx - 1]
                if prev_query_id in phases:
                    prev_data = data[data['query_id'] == prev_query_id].copy()
                    prev_valid = prev_data[
                        (prev_data['centre_idx'].notna()) &
                        (prev_data['centre_idx'] != -1) &
                        (prev_data['rel_ts'] < first_reading_ts)
                    ]
                    if len(prev_valid) > 0:
                        phase_info['composing_start'] = prev_valid['rel_ts'].max()
                    else:
                        phase_info['composing_start'] = first_reading_ts
                else:
                    phase_info['composing_start'] = first_reading_ts
            
            # Composing ends when this query's reviewing starts
            phase_info['composing_end'] = first_reading_ts
            
            # REVIEWING PHASE
            phase_info['reviewing_start'] = first_reading_ts
            
            if idx == len(query_sequence) - 1:
                # Last query: reviewing extends to EOF
                phase_info['reviewing_end'] = file_end
            else:
                # Other queries: reviewing ends at last valid centre_idx
                phase_info['reviewing_end'] = last_reading_ts
            
            phases[query_id] = phase_info
        
        return phases
    
    def split_phases(self, data: pd.DataFrame, composing_start: float, composing_end: float,
                     reviewing_start: float, reviewing_end: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into reviewing and composing phases based on timestamps.
        
        Args:
            data: Full DataFrame
            composing_start: Timestamp where composing starts
            composing_end: Timestamp where composing ends
            reviewing_start: Timestamp where reviewing starts
            reviewing_end: Timestamp where reviewing ends
        
        Returns:
            Tuple of (reviewing_data, composing_data)
        """
        if len(data) == 0:
            return pd.DataFrame(), pd.DataFrame()
        
        data = data.sort_values('rel_ts')
        
        reviewing_data = data[
            (data['rel_ts'] >= reviewing_start) & 
            (data['rel_ts'] <= reviewing_end)
        ].copy()
        
        composing_data = data[
            (data['rel_ts'] >= composing_start) & 
            (data['rel_ts'] < composing_end)
        ].copy()
        
        return reviewing_data, composing_data
