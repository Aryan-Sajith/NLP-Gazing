"""Text matching utilities for gaze data"""

from src.config.constants import WINDOW_CONTEXT_SIZE


def match_window_to_text(text: str, window: str, char_index: int) -> bool:
    """
    Check if window text appears in the message at the specified index.
    
    Both texts are cleaned of newlines/carriage returns before checking.
    Uses context window around the index to handle text formatting variations.
    
    Args:
        text: Full text to search in
        window: Small window of text to find
        char_index: Character index where window should appear
    
    Returns:
        True if window found near the index
    """
    # Clean both strings
    text_cleaned = text.replace('\n', ' ').replace('\r', ' ')
    window_cleaned = window.replace('\n', ' ').replace('\r', ' ')
    
    # Calculate context window boundaries
    start_idx = max(0, char_index - WINDOW_CONTEXT_SIZE)
    end_idx = min(len(text_cleaned), char_index + WINDOW_CONTEXT_SIZE)
    
    # Check if window appears in the context
    return window_cleaned in text_cleaned[start_idx:end_idx]
