# Explanation: `src/step-3-analyze-behavior.py`

## Purpose

This script analyzes user **eye-gaze and mouse-tracking data** collected during a pairwise LLM evaluation study. Users were shown two AI-generated responses side by side and asked questions; their gaze positions were recorded. The script's job is to:

1. Assign each gaze data point to the specific query the user was reading at that moment.
2. Aggregate statistics per (user, task, query) tuple.
3. Output the aggregated metrics to either a CSV file or a structured text report.

---

## Inputs

| File / Directory | Description |
|---|---|
| `user_behavior/<user_id>/<task_id>/rel_gaze*.csv` | Raw gaze data files, one per user per task. Each row is one gaze sample. |
| `query_data.json` | Maps each `(user_id, task_id)` pair to a list of queries, each carrying the actual LLM response text (`llm_response_1`, `llm_response_2`) and a `query_id`. |
| `task_table.csv` | Records when each user finished each task. Used to filter users to a valid date range. |

### Gaze CSV Row Format

Each row in the gaze CSV files has either 6 or 7 fields:

```
x, y, window_text, char_index, relative_timestamp, [camera_green,] absolute_timestamp
```

- `x, y` — gaze coordinates. `(-1, -1)` means the user was **not looking at the screen**.
- `window_text` — a short excerpt of text that was visible at the gaze point.
- `char_index` — the character offset of the gaze point within the displayed response text.
- `camera_green` — optional `1/0` flag indicating whether the camera detected a "green" signal (e.g., attention indicator).

---

## Key Design Decisions

### Left vs. Right Response

The script processes `rel_gaze_one.csv` / `rel_gaze_two.csv` (and mouse equivalents). Filenames ending in `_two` or `_right` are mapped to `llm_response_2` (the right panel); all others use `llm_response_1` (the left panel).

### Query ID Assignment (Two-Pass Algorithm)

Because a single session contains multiple queries presented sequentially, each gaze sample must be matched to the query the user was reading at that moment.

**Pass 1 — find each query's earliest timestamp:**
For every gaze sample that has a real screen position (not `-1, -1`), the script tries to match `window_text` (at `char_index`) against each query's LLM response using `match_window()`. When a match is found, the sample's `relative_timestamp` is recorded as the earliest known time for that query.

**Pass 2 — assign query IDs by timestamp:**
The query time ranges are sorted in reverse order. For each gaze sample, the script walks this sorted list and assigns the sample to the query whose earliest timestamp is the closest one that is ≤ the sample's own timestamp. This is a "most-recent query wins" assignment — any gaze activity is attributed to the last query whose content started appearing on screen before that moment.

### `match_window()` — Text Matching

```python
def match_window(msg, window, raw_idx):
```

Given a full response text (`msg`), a short visible excerpt (`window`), and a character index (`raw_idx`), this function checks whether `window` appears within ±15 characters of `raw_idx` in the cleaned response. Newlines are normalized to spaces before matching to handle text reflow.

---

## Per-Query Statistics Collected

For each `(user_id, task_id, query_id)` triple, the script accumulates:

| Field | Meaning |
|---|---|
| `total_entries_left` / `total_entries_right` | Total gaze samples attributed to this query from the left / right response file. |
| `response_left` / `response_right` | Samples where the user **was** looking at the screen (i.e., coordinates ≠ `-1, -1`). |
| `camera_green` | Samples where the `camera_green` flag was `1`. |
| `max_idx_left` / `max_idx_right` | Highest character index ever gazed at — a proxy for **how deep** into the response the user read. |
| `query_length_left` / `query_length_right` | Total character length of the LLM response, set on the first match. |

---

## Outputs

The output format is controlled by the `TO_OUTPUT_CSV` flag.

### `TO_OUTPUT_CSV = True` → `all_metrics.csv`

A flat CSV with one row per `(user_id, task_id, query_id)` and all eight metric columns listed above. This is the default and most analysis-friendly format.

### `TO_OUTPUT_CSV = False` → `all_final.txt`

A human-readable text report with three sections, each aggregated at query → task → user levels:

1. **Response Rate** — fraction of gaze samples where the user was actually looking at the LLM response text (not off-screen). Reported separately for left, right, and combined.

2. **Green Box Rate** — fraction of total gaze samples where `camera_green = 1`, indicating some form of visual attention signal from the camera.

3. **Maximum Index Ratio** — `max_idx / query_length`, measuring how far into the response the user read (0 = nothing, 1 = all the way to the end). Averaged across left and right for pairwise queries.

---

## Data Flow Summary

```
task_table.csv
    └─ filter by date range → valid user_ids

user_behavior/**/*.csv  +  query_data.json
    └─ for each gaze file:
        1. Pass 1: match gaze windows to responses → query earliest timestamps
        2. Pass 2: assign each gaze row a query_id by timestamp
        3. Accumulate (total_entries, response_looks, camera_green, max_idx)

analysis_dict  [user_id][task_id][query_id]
    └─ TO_OUTPUT_CSV=True  → all_metrics.csv  (flat table)
    └─ TO_OUTPUT_CSV=False → all_final.txt    (hierarchical text report)
```

---

## Configuration Constants

| Constant | Default | Purpose |
|---|---|---|
| `BASE_DIR` | `user_behavior/` | Root of the gaze data tree |
| `QUERY_JSON` | `query_data.json` | LLM response lookup table |
| `START_DATE` / `END_DATE` | 2024-01-19 / 2027-11-30 | Filter window for valid sessions |
| `NO_GAZE_QUERY_ID` | `-2` | Sentinel for off-screen samples (defined but not currently used in output) |
| `PROMPT_GAZE_QUERY_ID` | `-1` | Sentinel for prompt-area gazes (defined but not currently used in output) |
| `BASE_QUERY_ID` | `0` | Default when no query match is found; rows with this ID are skipped |
| `TO_OUTPUT_CSV` | `True` | Switch between CSV and text output |
