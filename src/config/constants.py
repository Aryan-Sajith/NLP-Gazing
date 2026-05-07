"""Application-wide constants and configuration defaults"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "user_behavior"
OUTPUT_DIR = PROJECT_ROOT / "output"
QUERY_LOGS_FILE = PROJECT_ROOT / "query_logs_table.csv"

# Special query IDs for non-standard entries
QUERY_ID_NOT_LOOKING = -2  # User not looking at screen (x=-1, y=-1)
QUERY_ID_PROMPT = -1  # User looking at experimental prompt
QUERY_ID_DEFAULT = 0  # Default when no match found

# Experimental prompt text
EXPERIMENTAL_PROMPT = (
    "After you type your question, please wait patiently. It might take up to 1 "
    "minute for AI to finish generating their answer. The AI's response will "
    "show up here."
)

# File patterns for behavioral data
PAIRWISE_FILES = [
    "rel_gaze_one.csv",
    "rel_gaze_two.csv",
    "rel_mouse_left.csv",
    "rel_mouse_right.csv"
]

POINTWISE_FILES = [
    "rel_gaze.csv",
    "rel_mouse.csv"
]

# Text matching parameters
WINDOW_CONTEXT_SIZE = 15  # Characters before/after index for matching

# Quality standards for worker analysis
QUALITY_STANDARDS = {
    'min_total_rows': 2000,
    'min_text_rows': 500,
    'min_duration_minutes': 2.5,
    'min_text_percentage': 10.0
}

# Feature extraction parameters
#INACTIVITY_THRESHOLD_MS = 2000.0  # Milliseconds
INACTIVITY_THRESHOLD_MS = 1000.0  # Milliseconds
NUM_TIME_WINDOWS = 20  # For temporal windowing features
