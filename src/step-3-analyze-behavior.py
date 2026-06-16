"""
Annotate every   data/<user>/<task>/rel_*.csv   file and write the results to a
new file whose name is the original stem plus "-query-id-assigned".

Example:
    rel_gaze_two.csv  ->  rel_gaze_two-query-id-assigned.csv
The original files are left untouched.
"""

import csv
import pandas as pd
import json
import math
from pathlib import Path
from collections import defaultdict

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

# Only process pairwise files (exclude pointwise files)
PAIRWISE_FILES = [
    "rel_gaze_one.csv",
    "rel_gaze_two.csv",
    "rel_mouse_left.csv",
    "rel_mouse_right.csv"
]

OUTPUT_RESPONSE_FILE = "all_final.txt"
OUTPUT_GREEN_BOX_FILE = "green_box.txt"
OUTPUT_IDX_FILE = "max_index.txt"

TASK_TABLE_FILE = Path("task_table.csv")
EDITED_TASK_TABLE_FILE = Path("edit_task_table.csv")
START_DATE = pd.to_datetime("2024-01-19 00:00:01")
END_DATE = pd.to_datetime("2027-11-30 14:24:56")

OUTPUT_CSV = Path("output/all_metrics.csv")
TO_OUTPUT_CSV = True #False gives txt file 
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
    if not msg or not window:
        return False

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
#user -> task_id -> query_id -> {camera_green, gaze_*, mouse_*}
def make_leaf():
    return {
        "camera_green": 0,
        "gaze_total_entries_left": 0,
        "gaze_total_entries_right": 0,
        "gaze_onscreen_points_left": 0,
        "gaze_onscreen_points_right": 0,
        "gaze_max_idx_left": -1,
        "gaze_max_idx_right": -1,
        "mouse_total_entries_left": 0,
        "mouse_total_entries_right": 0,
        "mouse_onscreen_points_left": 0,
        "mouse_onscreen_points_right": 0,
        "mouse_max_idx_left": -1,
        "mouse_max_idx_right": -1,
        "response_length_left": -1,
        "response_length_right": -1,
    }

analysis_dict = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(make_leaf)
    )
)

#gets all user_ids within timerange to filter out everything that isnt
user_ids_within_time_range = []


# for src in TASK_TABLE_FILE.rglob("Batch_*.csv"):
df_database = pd.read_csv(TASK_TABLE_FILE)
df_database["user_id"] = df_database["user_id"].astype(str).str.strip()
df_database["finished"] = pd.to_datetime(df_database["finished"], format="%Y-%m-%d %H:%M:%S")
df_database = df_database[df_database["finished"].notna() & (df_database["finished"].between(START_DATE, END_DATE))]
df_database.to_csv(EDITED_TASK_TABLE_FILE, index=False)

user_ids_within_time_range.extend(df_database['user_id'].unique())


gaze_sources = list(BASE_DIR.rglob("rel_gaze*.csv"))
mouse_sources = list(BASE_DIR.rglob("rel_mouse_left.csv")) + list(BASE_DIR.rglob("rel_mouse_right.csv")) + list(BASE_DIR.rglob("rel_mouse.csv"))

for src in gaze_sources + mouse_sources:
    try:
        _, user_id, task_id, _ = src.parts[-4:]
    except ValueError:
        continue

    if user_id not in user_ids_within_time_range:
        continue

    # locate task block in JSON
    task_block = QUERY_DATA.get(user_id, {}).get(task_id, [])
    if not task_block:
        continue

    # pick which response column matters for this file
    resp_key = 'llm_response_2' if src.stem.endswith("_two") or src.stem.endswith("_right") else 'llm_response_1'
    prefix = "mouse_" if src.stem.startswith("rel_mouse") else "gaze_"

    # build (query_id, response_text) pairs
    responses = [
        (q["query_id"], q.get(resp_key, ""))
        for q in task_block
        if q.get(resp_key)
    ]
    # if not responses:
    #     continue

    # read the file
    with src.open(encoding="utf-8") as fh:
        raw_rows = list(csv.reader(fh))


    # First pass: Build timestamp ranges for each query_id
    # We'll use this to assign query_ids to not_looking events based on their timestamps
    query_time_ranges = {}  # {query_id: min_ts}

    # Temporary first pass to find time ranges
    for row in raw_rows:
        if len(row) == 7:
            x, y, window, idx_str, rel_ts_str, _, abs_ts = row
        elif len(row) == 6:
            x, y, window, idx_str, rel_ts_str, abs_ts = row
        else:
            continue

        try:
            x_f, y_f = float(x), float(y)
            rel_ts = float(rel_ts_str)
        except (ValueError, TypeError):
            continue

        try:
            idx_i = int(float(idx_str)) if idx_str.strip() else -1
        except (ValueError, TypeError):
            idx_i = -1

        window = window.strip()
        is_not_looking = x_f == -1 and y_f == -1

        # Only process looking events for time range calculation
        if not is_not_looking:
            # Check if this matches any query
            for qid, text in responses:
                if match_window(text, window, idx_i):
                    if qid not in query_time_ranges:
                        query_time_ranges[qid] = rel_ts
                        if resp_key == 'llm_response_2':
                            analysis_dict[user_id][task_id][qid]["response_length_right"] = len(text)
                        else:
                            analysis_dict[user_id][task_id][qid]["response_length_left"] = len(text)
                    else:
                        # query_id =
                        query_time_ranges[qid] = min(query_time_ranges[qid], rel_ts)
                    break

    query_time_ranges = sorted(query_time_ranges.items(), key=lambda x: x[1], reverse=True)

    # processes rows
    processed_rows = []
    # query_stats = defaultdict(lambda: {'total': 0, 'response_looks': 0})

    for row in raw_rows:
        # csv.reader handles CSV quoting properly
        # Some files have 6 fields: x, y, window, idx, rel_ts, abs_ts
        # Some files have 7 fields: x, y, window, idx, rel_ts, unknown, abs_ts

        if len(row) == 7:
            x, y, window, idx_str, rel_ts, camera_green, abs_ts = row
            camera_green = True if camera_green == '1' else False
        elif len(row) == 6:
            x, y, window, idx_str, rel_ts, abs_ts = row
            camera_green = True
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
        query_id = BASE_QUERY_ID # default value if no match is found

        rel_ts_float = float(rel_ts)
        # best_min_ts = float('-inf')

        for qid, min_ts in query_time_ranges:
            if min_ts <= rel_ts_float:
                query_id = qid
                break


        if query_id == BASE_QUERY_ID:
            continue


        #if right side
        if resp_key == 'llm_response_2':
            analysis_dict[user_id][task_id][query_id][f"{prefix}total_entries_right"] += 1
            if not is_not_looking:
                analysis_dict[user_id][task_id][query_id][f"{prefix}onscreen_points_right"] += 1
            if analysis_dict[user_id][task_id][query_id][f"{prefix}max_idx_right"] < idx_i:
                analysis_dict[user_id][task_id][query_id][f"{prefix}max_idx_right"] = idx_i
        else:
            analysis_dict[user_id][task_id][query_id][f"{prefix}total_entries_left"] += 1
            if not is_not_looking:
                analysis_dict[user_id][task_id][query_id][f"{prefix}onscreen_points_left"] += 1
            if analysis_dict[user_id][task_id][query_id][f"{prefix}max_idx_left"] < idx_i:
                analysis_dict[user_id][task_id][query_id][f"{prefix}max_idx_left"] = idx_i

        if camera_green and prefix == "gaze_":
            analysis_dict[user_id][task_id][query_id]["camera_green"] += 1

        # print()

summary_query_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
summary_task_dict = defaultdict(lambda: defaultdict(float))
# summary_user_dict = defaultdict(float)

if TO_OUTPUT_CSV:
    rows = []

    for k1, d1 in analysis_dict.items():
        for k2, d2 in d1.items():
            for k3, leaf in d2.items():
                row = {
                    "user_id": k1,
                    "task_id": k2,
                    "query_id": k3,
                    **leaf
                }
                rows.append(row)

    analysis_df = pd.DataFrame(rows)
    analysis_df['gaze_max_idx_ratio_left'] = analysis_df.apply(lambda row: row['gaze_max_idx_left'] / row['response_length_left'] if row['response_length_left'] > 0 else 0, axis=1)
    analysis_df['gaze_max_idx_ratio_right'] = analysis_df.apply(lambda row: row['gaze_max_idx_right'] / row['response_length_right'] if row['response_length_right'] > 0 else 0, axis=1)
    analysis_df['mouse_max_idx_ratio_left'] = analysis_df.apply(lambda row: row['mouse_max_idx_left'] / row['response_length_left'] if row['response_length_left'] > 0 else 0, axis=1)
    analysis_df['mouse_max_idx_ratio_right'] = analysis_df.apply(lambda row: row['mouse_max_idx_right'] / row['response_length_right'] if row['response_length_right'] > 0 else 0, axis=1)
    analysis_df['camera_green_ratio'] = analysis_df.apply(lambda row: row['camera_green'] / (row['gaze_total_entries_left'] + row['gaze_total_entries_right']) if (row['gaze_total_entries_left'] + row['gaze_total_entries_right']) > 0 else 0, axis=1)
    analysis_df.to_csv(OUTPUT_CSV, index=False)
else:
    with open(OUTPUT_RESPONSE_FILE, mode="w") as f:
        # ------------------------------------------------------------------- #
        # Gaze Response
        # ------------------------------------------------------------------- #
        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Queries (Gaze Response)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in analysis_dict.items():
            f.write(f'User ID: {user_id}\n')
            for task_id, query_id_dict in task_id_dict.items():
                f.write(f'\tTask_ID: {task_id}\n')
                for query_id, data_dict in query_id_dict.items():
                    if data_dict['gaze_total_entries_right'] == 0:
                        percentage = 0 if data_dict['gaze_total_entries_left'] == 0 else data_dict['gaze_onscreen_points_left'] / data_dict["gaze_total_entries_left"]
                        f.write(f'\t\tQuery_ID {query_id}: {percentage}\n')
                        summary_query_dict[user_id][task_id][query_id] = percentage
                    else:
                        left_percentage = 0 if data_dict['gaze_total_entries_left'] == 0 else data_dict['gaze_onscreen_points_left'] / data_dict['gaze_total_entries_left']
                        right_percentage = data_dict["gaze_onscreen_points_right"] / data_dict['gaze_total_entries_right']
                        overall_percentage = (data_dict["gaze_onscreen_points_left"] + data_dict['gaze_onscreen_points_right']) / (data_dict["gaze_total_entries_left"] + data_dict["gaze_total_entries_right"])

                        f.write(f'\t\tQuery_ID (Left) {query_id}: {left_percentage}\n')
                        f.write(f'\t\tQuery_ID (Right) {query_id}: {right_percentage}\n')
                        f.write(f'\t\tQuery_ID (Overall) {query_id}: {overall_percentage}\n')
                        f.write(f'\t\tQuery_ID (Sum) {query_id}: {left_percentage + right_percentage}\n')

                        summary_query_dict[user_id][task_id][query_id] = overall_percentage

                    f.write('\n')

                f.write('\n')
            # f.write('\n')

        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Tasks (Gaze Response)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in summary_query_dict.items():
            f.write(f'User ID: {user_id}\n')
            for task_id, query_id_dict in task_id_dict.items():
                percentage = sum(query_id_dict.values()) / len(query_id_dict)
                summary_task_dict[user_id][task_id] = percentage
                f.write(f'\tTask_ID {task_id}: {percentage}\n\n')
            f.write('\n')

        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Users (Gaze Response)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in summary_task_dict.items():
            percentage = sum(task_id_dict.values()) / len(task_id_dict)
            f.write(f'User ID {user_id}: {percentage}\n\n')
        f.write('\n')

        summary_query_dict.clear()
        summary_task_dict.clear()

        # ------------------------------------------------------------------- #
        # Mouse Response
        # ------------------------------------------------------------------- #
        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Queries (Mouse Response)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in analysis_dict.items():
            f.write(f'User ID: {user_id}\n')
            for task_id, query_id_dict in task_id_dict.items():
                f.write(f'\tTask_ID: {task_id}\n')
                for query_id, data_dict in query_id_dict.items():
                    if data_dict['mouse_total_entries_right'] == 0:
                        percentage = 0 if data_dict['mouse_total_entries_left'] == 0 else data_dict['mouse_onscreen_points_left'] / data_dict["mouse_total_entries_left"]
                        f.write(f'\t\tQuery_ID {query_id}: {percentage}\n')
                        summary_query_dict[user_id][task_id][query_id] = percentage
                    else:
                        left_percentage = 0 if data_dict['mouse_total_entries_left'] == 0 else data_dict['mouse_onscreen_points_left'] / data_dict['mouse_total_entries_left']
                        right_percentage = data_dict["mouse_onscreen_points_right"] / data_dict['mouse_total_entries_right']
                        overall_percentage = (data_dict["mouse_onscreen_points_left"] + data_dict['mouse_onscreen_points_right']) / (data_dict["mouse_total_entries_left"] + data_dict["mouse_total_entries_right"])

                        f.write(f'\t\tQuery_ID (Left) {query_id}: {left_percentage}\n')
                        f.write(f'\t\tQuery_ID (Right) {query_id}: {right_percentage}\n')
                        f.write(f'\t\tQuery_ID (Overall) {query_id}: {overall_percentage}\n')
                        f.write(f'\t\tQuery_ID (Sum) {query_id}: {left_percentage + right_percentage}\n')

                        summary_query_dict[user_id][task_id][query_id] = overall_percentage

                    f.write('\n')

                f.write('\n')

        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Tasks (Mouse Response)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in summary_query_dict.items():
            f.write(f'User ID: {user_id}\n')
            for task_id, query_id_dict in task_id_dict.items():
                percentage = sum(query_id_dict.values()) / len(query_id_dict)
                summary_task_dict[user_id][task_id] = percentage
                f.write(f'\tTask_ID {task_id}: {percentage}\n\n')
            f.write('\n')

        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Users (Mouse Response)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in summary_task_dict.items():
            percentage = sum(task_id_dict.values()) / len(task_id_dict)
            f.write(f'User ID {user_id}: {percentage}\n\n')
        f.write('\n')

        summary_query_dict.clear()
        summary_task_dict.clear()

    # with open(OUTPUT_GREEN_BOX_FILE, mode="w") as f:
        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Queries (Green Box)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in analysis_dict.items():
            f.write(f'User ID: {user_id}\n')
            for task_id, query_id_dict in task_id_dict.items():
                f.write(f'\tTask_ID: {task_id}\n')
                for query_id, data_dict in query_id_dict.items():
                    if data_dict['gaze_total_entries_left'] + data_dict['gaze_total_entries_right'] > 0:
                        percentage = data_dict['camera_green'] / (data_dict["gaze_total_entries_left"] + data_dict['gaze_total_entries_right'])
                        f.write(f'\t\tQuery_ID {query_id}: {percentage}\n')
                        summary_query_dict[user_id][task_id][query_id] = percentage
                    f.write('\n')
                f.write('\n')
            # f.write('\n')

        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Tasks (Green Box)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in summary_query_dict.items():
            f.write(f'User ID: {user_id}\n')
            for task_id, query_id_dict in task_id_dict.items():
                percentage = sum(query_id_dict.values()) / len(query_id_dict)
                summary_task_dict[user_id][task_id] = percentage
                f.write(f'\tTask_ID {task_id}: {percentage}\n\n')
            f.write('\n')

        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Users (Green Box)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in summary_task_dict.items():
            percentage = sum(task_id_dict.values()) / len(task_id_dict)
            f.write(f'User ID {user_id}: {percentage}\n\n')
        f.write('\n')

        summary_query_dict.clear()
        summary_task_dict.clear()

    # with open(OUTPUT_IDX_FILE, mode="w") as f:
        # ------------------------------------------------------------------- #
        # Gaze Maximum Index
        # ------------------------------------------------------------------- #
        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Maximum Index For Each Query (Gaze)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in analysis_dict.items():
            f.write(f'User ID: {user_id}\n')
            for task_id, query_id_dict in task_id_dict.items():
                f.write(f'\tTask_ID: {task_id}\n')
                for query_id, data_dict in query_id_dict.items():
                    if data_dict["response_length_right"] > 0:
                        left_ratio = data_dict["gaze_max_idx_left"] / data_dict["response_length_left"] if data_dict["gaze_max_idx_left"] > 0 else 0
                        f.write(f'\t\tQuery_ID (Left) {query_id}: {left_ratio}')
                        f.write('\n')
                        right_ratio = data_dict["gaze_max_idx_right"] / data_dict["response_length_right"] if data_dict["gaze_max_idx_right"] > 0 else 0
                        f.write(f'\t\tQuery_ID (Right) {query_id}: {right_ratio}')
                        f.write('\n')

                        avg_ratio = (left_ratio + right_ratio) / 2
                        f.write(f'\t\tQuery_ID (Avg) {query_id}: {avg_ratio}')
                        summary_query_dict[user_id][task_id][query_id] = avg_ratio
                    else:
                        left_ratio = data_dict["gaze_max_idx_left"] / data_dict["response_length_left"] if data_dict["gaze_max_idx_left"] > 0 else 0
                        f.write(f'\t\tQuery_ID {query_id}: {left_ratio}')

                        summary_query_dict[user_id][task_id][query_id] = left_ratio

                    f.write('\n\n')

                f.write('\n')

        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Tasks (Maximum Index Gaze)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in summary_query_dict.items():
            f.write(f'User ID: {user_id}\n')
            for task_id, query_id_dict in task_id_dict.items():
                avg_task = sum(query_id_dict.values()) / len(query_id_dict)
                summary_task_dict[user_id][task_id] = avg_task
                f.write(f'\tTask_ID {task_id}: {avg_task}\n\n')
            f.write('\n')

        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Users (Maximum Index Gaze)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in summary_task_dict.items():
            avg_user = sum(task_id_dict.values()) / len(task_id_dict)
            f.write(f'User ID {user_id}: {avg_user}\n\n')
        f.write('\n')

        summary_query_dict.clear()
        summary_task_dict.clear()

        # ------------------------------------------------------------------- #
        # Mouse Maximum Index
        # ------------------------------------------------------------------- #
        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Maximum Index For Each Query (Mouse)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in analysis_dict.items():
            f.write(f'User ID: {user_id}\n')
            for task_id, query_id_dict in task_id_dict.items():
                f.write(f'\tTask_ID: {task_id}\n')
                for query_id, data_dict in query_id_dict.items():
                    if data_dict["response_length_right"] > 0:
                        left_ratio = data_dict["mouse_max_idx_left"] / data_dict["response_length_left"] if data_dict["mouse_max_idx_left"] > 0 else 0
                        f.write(f'\t\tQuery_ID (Left) {query_id}: {left_ratio}')
                        f.write('\n')
                        right_ratio = data_dict["mouse_max_idx_right"] / data_dict["response_length_right"] if data_dict["mouse_max_idx_right"] > 0 else 0
                        f.write(f'\t\tQuery_ID (Right) {query_id}: {right_ratio}')
                        f.write('\n')

                        avg_ratio = (left_ratio + right_ratio) / 2
                        f.write(f'\t\tQuery_ID (Avg) {query_id}: {avg_ratio}')
                        summary_query_dict[user_id][task_id][query_id] = avg_ratio
                    else:
                        left_ratio = data_dict["mouse_max_idx_left"] / data_dict["response_length_left"] if data_dict["mouse_max_idx_left"] > 0 else 0
                        f.write(f'\t\tQuery_ID {query_id}: {left_ratio}')

                        summary_query_dict[user_id][task_id][query_id] = left_ratio

                    f.write('\n\n')

                f.write('\n')

        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Tasks (Maximum Index Mouse)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in summary_query_dict.items():
            f.write(f'User ID: {user_id}\n')
            for task_id, query_id_dict in task_id_dict.items():
                avg_task = sum(query_id_dict.values()) / len(query_id_dict)
                summary_task_dict[user_id][task_id] = avg_task
                f.write(f'\tTask_ID {task_id}: {avg_task}\n\n')
            f.write('\n')

        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Average Across Users (Maximum Index Mouse)\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")

        for user_id, task_id_dict in summary_task_dict.items():
            avg_user = sum(task_id_dict.values()) / len(task_id_dict)
            f.write(f'User ID {user_id}: {avg_user}\n\n')
        f.write('\n')
            # f.write('\n')

        f.write("# ----------------------------------------------------------------------- #\n")
        f.write("# Num Tasks Per User\n")
        f.write("# ----------------------------------------------------------------------- #\n\n")
        for user_id, task_id_dict in summary_task_dict.items():
            f.write(f'User ID {user_id} completed: {len(task_id_dict)}\n\n')
