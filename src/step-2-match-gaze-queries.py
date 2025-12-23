"""
Annotate every   data/<user>/<task>/rel_*.csv   file and write the results to a
new file whose name is the original stem plus "-query-id-assigned".

Example:
    rel_gaze_two.csv  ->  rel_gaze_two-query-id-assigned.csv
The original files are left untouched.

METHOD: Sequential context-based matching
This uses a sequential processing approach where "not looking" events are assigned
to the query the user was JUST reading, not based on time ranges (which causes bugs
when users look at the same query multiple times with breaks in between).
"""

import csv
import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
BASE_DIR   = Path("user_behavior")
QUERY_JSON = Path("query_data.json")
PROMPT = (
    "After you type your question, please wait patiently. It might take up to 1 "
    "minute for AI to finish generating their answer. The AI's response will "
    "show up here."
)
NO_GAZE_QUERY_ID = -2  # special value for "not looking at the screen"
PROMPT_GAZE_QUERY_ID = -1  # special value for "looking at the prompt"
BASE_QUERY_ID = 0  # default value for query_id if no match is found

# Process both pairwise and pointwise files
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

ALL_FILES = PAIRWISE_FILES + POINTWISE_FILES

# --------------------------------------------------------------------------- #
# UTILITIES
# --------------------------------------------------------------------------- #
def match_window(msg: str, window: str, raw_idx: int) -> bool:
    """Check if the given message contains the window text at the specified index.
    The message is cleaned of newlines and carriage returns before checking.
    The window is 10 characters before and after the index but clipped at the edges of the message.
    We use 15 characters before and after the index to allow for more context to handle edge cases where
    the window might not be exactly at the index due to text formatting or other issues.
    """
    # Clean both message and window of newlines and carriage returns
    msg_cleaned = msg.replace('\n', ' ').replace('\r', ' ')
    window_cleaned = window.replace('\n', ' ').replace('\r', ' ')
    
    window_length = 15  # characters before and after the index
    start_idx_inclusive = raw_idx - window_length if raw_idx - window_length >= 0 else 0
    end_idx_exclusive = raw_idx + window_length if raw_idx + window_length < len(msg_cleaned) else len(msg_cleaned)
    return window_cleaned in msg_cleaned[start_idx_inclusive:end_idx_exclusive]

# --------------------------------------------------------------------------- #
# LOAD QUERY DATA
# --------------------------------------------------------------------------- #
with QUERY_JSON.open(encoding="utf-8") as fh:
    QUERY_DATA: dict = json.load(fh)

# --------------------------------------------------------------------------- #
# MAIN WALK
# --------------------------------------------------------------------------- #
for src in BASE_DIR.rglob("rel_*.csv"):
    # Process both pairwise and pointwise files
    if src.name not in ALL_FILES:
        continue
        
    # derive user_id / task_id from path:  user_behavior/user_id/task_id/file.csv
    try:
        _, user_id, task_id, _ = src.parts[-4:]
    except ValueError:
        continue

    # locate task block in JSON
    task_block = QUERY_DATA.get(user_id, {}).get(task_id, [])
    if not task_block:
        continue

    # Determine if this is a pointwise or pairwise file and which response(s) to use
    is_pointwise = src.name in POINTWISE_FILES
    
    if is_pointwise:
        # For pointwise files, use llm_response_1 (the only response)
        responses = [
            (q["query_id"], q.get("llm_response_1", ""))
            for q in task_block
            if q.get("llm_response_1")
        ]
    else:
        # For pairwise files, pick which response column matters
        resp_key = "llm_response_2" if src.stem.endswith("_two") or src.stem.endswith("_right") else "llm_response_1"
        responses = [
            (q["query_id"], q.get(resp_key, ""))
            for q in task_block
            if q.get(resp_key)
        ]
    
    if not responses:
        continue

    # read the file
    with src.open(encoding="utf-8") as fh:
        raw_rows = list(csv.reader(fh))
    
    # NEW METHOD: Sequential processing with context tracking
    out_rows = [
        [
            "x",
            "y",
            "window",
            "centre_idx",
            "rel_ts",
            "abs_ts",
            "query_id",
            "is_experimental_text",
            "is_not_looking",
        ]
    ]

    # Track the last query the user was actually reading
    last_query_id = NO_GAZE_QUERY_ID  # Start with "not looking"

    for row in raw_rows:
        # csv.reader handles CSV quoting properly
        # Some files have 6 fields: x, y, window, idx, rel_ts, abs_ts
        # Some files have 7 fields: x, y, window, idx, rel_ts, unknown, abs_ts
        
        if len(row) == 7:
            x, y, window, idx_str, rel_ts, _, abs_ts = row
        elif len(row) == 6:
            x, y, window, idx_str, rel_ts, abs_ts = row
        else:
            # Skip invalid rows (header or malformed)
            continue
        
        # Parse numeric values
        try:
            x_f, y_f = float(x), float(y)
        except (ValueError, TypeError):
            continue
        
        # Parse index
        try:
            idx_i = int(float(idx_str)) if idx_str.strip() else -1
        except (ValueError, TypeError):
            idx_i = -1
        
        # Clean window text
        window = window.strip()

        is_not_looking = x_f == -1 and y_f == -1
        is_exp_text = False
        query_id = BASE_QUERY_ID  # default value if no match is found

        if is_not_looking:
            # NEW METHOD: Assign to the last query the user was reading
            query_id = last_query_id
        elif match_window(PROMPT, window, idx_i):
            # Looking at prompt - assign -1 but DON'T update last_query_id
            # (we want "not looking" after prompt to go to the last real query)
            query_id = PROMPT_GAZE_QUERY_ID
            is_exp_text = True
            # Don't update last_query_id here
        else:
            # Try to match to a query response
            matched = False
            for qid, text in responses:
                if match_window(text, window, idx_i):
                    query_id = qid
                    last_query_id = qid  # Update context
                    matched = True
                    break
            
            # If no match found, keep query_id as BASE_QUERY_ID (0)
            # and don't update last_query_id

        out_rows.append(
            [
                x,
                y,
                window,
                idx_i,
                rel_ts,
                abs_ts,
                query_id,
                str(is_exp_text).lower(),
                str(is_not_looking).lower(),
            ]
        )

    # ----------------------------------------------------------------------- #
    # WRITE NEW FILE (do NOT overwrite original)
    # ----------------------------------------------------------------------- #
    dst = src.with_name(f"{src.stem}_query_id_assigned.csv")
    with dst.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(out_rows)

print("Finished: new files suffixed with '_query_id_assigned' created.")
