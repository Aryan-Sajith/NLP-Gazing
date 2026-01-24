"""File system utilities for discovering data files"""

from pathlib import Path
from typing import List, Tuple, Optional
from src.config.constants import PAIRWISE_FILES, POINTWISE_FILES


def discover_user_task_combinations(data_dir: Path) -> List[Tuple[str, str]]:
    """
    Discover all user/task directory combinations.
    
    Args:
        data_dir: Base directory containing user_behavior data
    
    Returns:
        List of (user_id, task_id) tuples
    """
    combinations = []
    
    try:
        for user_dir in data_dir.iterdir():
            if not user_dir.is_dir() or user_dir.name.startswith('.'):
                continue
            
            user_id = user_dir.name
            for task_dir in user_dir.iterdir():
                if task_dir.is_dir():
                    task_id = task_dir.name
                    combinations.append((user_id, task_id))
    
    except Exception as e:
        print(f"Error discovering combinations: {e}")
    
    return combinations


def detect_comparison_type(data_dir: Path, user_id: str, task_id: str) -> Optional[str]:
    """
    Detect if task has pairwise or pointwise data based on files present.
    
    Args:
        data_dir: Base directory
        user_id: User identifier
        task_id: Task identifier
    
    Returns:
        'pairwise', 'pointwise', or None if neither
    """
    task_path = data_dir / user_id / task_id
    
    # Check for pairwise files
    pairwise_annotated = [f.replace('.csv', '_query_id_assigned.csv') for f in PAIRWISE_FILES]
    has_pairwise = all((task_path / f).exists() for f in pairwise_annotated)
    
    # Check for pointwise files
    pointwise_annotated = [f.replace('.csv', '_query_id_assigned.csv') for f in POINTWISE_FILES]
    has_pointwise = all((task_path / f).exists() for f in pointwise_annotated)
    
    if has_pairwise:
        return 'pairwise'
    elif has_pointwise:
        return 'pointwise'
    else:
        return None


def find_behavioral_files(data_dir: Path, user_id: str, task_id: str, 
                          file_patterns: List[str]) -> List[Path]:
    """
    Find all matching behavioral CSV files for a user/task.
    
    Args:
        data_dir: Base directory
        user_id: User identifier
        task_id: Task identifier
        file_patterns: List of filename patterns to match
    
    Returns:
        List of paths to matching files
    """
    task_path = data_dir / user_id / task_id
    files = []
    
    for pattern in file_patterns:
        file_path = task_path / pattern
        if file_path.exists():
            files.append(file_path)
    
    return files
