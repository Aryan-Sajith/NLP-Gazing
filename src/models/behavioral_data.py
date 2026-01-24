"""Behavioral data structures and models"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BehavioralDataPoint:
    """Single data point from gaze or mouse tracking"""
    x: float
    y: float
    window: str
    centre_idx: int
    rel_ts: float
    abs_ts: float
    query_id: int
    is_experimental_text: bool
    is_not_looking: bool
    
    @classmethod
    def from_csv_row(cls, row: dict, query_id: int, 
                     is_experimental_text: bool, is_not_looking: bool):
        """Create instance from CSV row data"""
        try:
            x = float(row['x']) if row.get('x') != '-1' else -1.0
            y = float(row['y']) if row.get('y') != '-1' else -1.0
            centre_idx = int(row['centre_idx']) if (
                row.get('centre_idx') and 
                row['centre_idx'].strip() and 
                row['centre_idx'] != '-1'
            ) else -1
            
            return cls(
                x=x,
                y=y,
                window=row.get('window', ''),
                centre_idx=centre_idx,
                rel_ts=float(row['rel_ts']),
                abs_ts=float(row['abs_ts']),
                query_id=query_id,
                is_experimental_text=is_experimental_text,
                is_not_looking=is_not_looking
            )
        except (ValueError, KeyError) as e:
            return None
    
    def is_looking_at_text(self) -> bool:
        """Check if user is actively looking at text"""
        return self.x != -1 and self.y != -1 and self.centre_idx != -1
