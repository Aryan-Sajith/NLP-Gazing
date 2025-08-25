"""
Annotate every   data/<user>/<task>/rel_*.csv   file and write the results to a
new file whose name is the original stem plus “-query-id-assigned”.

Example:
    rel_gaze_two.csv  ->  rel_gaze_two-query-id-assigned.csv
The original files are left untouched.
"""

import csv
import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
BASE_DIR   = Path("to-fix-data")
QUERY_JSON = Path("query_data.json")
PROMPT = (
    "After you type your question, please wait patiently. It might take up to 1 "
    "minute for AI to finish generating their answer. The AI's response will "
    "show up here."
)
NO_GAZE_QUERY_ID = -2  # special value for "not looking at the screen"
PROMPT_GAZE_QUERY_ID = -1  # special value for "looking at the prompt"
BASE_QUERY_ID = 0  # default value for query_id if no match is found

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
    window_length = 15  # characters before and after the index
    start_idx_inclusive = raw_idx - window_length if raw_idx - window_length >= 0 else 0
    end_idx_exclusive = raw_idx + window_length if raw_idx + window_length < len(msg) else len(msg)
    return window in msg[start_idx_inclusive:end_idx_exclusive]

# --------------------------------------------------------------------------- #
# LOAD QUERY DATA
# --------------------------------------------------------------------------- #
with QUERY_JSON.open(encoding="utf-8") as fh:
    QUERY_DATA: dict = json.load(fh)

# --------------------------------------------------------------------------- #
# MAIN WALK
# --------------------------------------------------------------------------- #
for src in BASE_DIR.rglob("rel_*.csv"):
    # derive user_id / task_id from path:  data/user_id/task_id/file.csv
    try:
        _, user_id, task_id, _ = src.parts[-4:]
    except ValueError:
        continue

    # locate task block in JSON
    task_block = QUERY_DATA.get(user_id, {}).get(task_id, [])
    if not task_block:
        continue

    # pick which response column matters for this file
    resp_key = "llm_response_2" if src.stem.endswith("_two") or src.stem.endswith("_right") else "llm_response_1"

    # build (query_id, response_text) pairs
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

    # process rows
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

    for x, y, window, idx, rel_ts, abs_ts, *rest in raw_rows:
        x_f, y_f = float(x), float(y)
        idx_i = int(idx) if idx else -1
        window = window or ""

        is_not_looking = x_f == -1 and y_f == -1
        is_exp_text = False
        query_id = BASE_QUERY_ID # default value if no match is found

        if is_not_looking:
            query_id = NO_GAZE_QUERY_ID # special value for "not looking at the screen"
        elif match_window(PROMPT, window, idx_i):
            query_id = PROMPT_GAZE_QUERY_ID # special value for "looking at the prompt"
            is_exp_text = True
        else:
            for qid, text in responses:
                if match_window(text, window, idx_i):
                    query_id = qid
                    break

        out_rows.append(
            [
                x,
                y,
                window,
                idx,
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