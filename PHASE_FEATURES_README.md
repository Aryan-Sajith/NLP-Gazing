# Phase-Based Feature Extraction

## Overview

When users interact with LLM outputs, they go through two distinct behavioral phases **within the same query**:

1. **Reviewing Phase**: User actively reads and engages with the LLM's response
2. **Composing Phase**: User stops reading, rates the response, and formulates their next question (all while still on the same query_id)

We detect this transition point and extract separate behavioral features for each phase, capturing how users engage with different responses.

---

## Phase Detection Method

We use a **timeline-based approach** that analyzes the full behavioral timeline:

**Algorithm**:
1. Read full behavioral file (not filtered by query_id)
2. Identify all queries in chronological order
3. For each query:
   - **Reviewing phase**: First to last valid `centre_idx` timestamp (when user is looking at text)
   - **Composing phase**: Time after previous query ended until reviewing starts (or before first query)
4. Boundaries based on actual reading activity, not arbitrary percentages

**Key insight**: Composing happens BETWEEN queries, not after reaching a percentage threshold. Users compose their next query after finishing the previous response.

---

## Extracted Features

We extract **48 new features for pointwise** and **~42 new features for pairwise** tasks (added to existing 426 features).

### Pointwise Features

**Phase Timing** (6 features per modality):
- Duration and percentage of reviewing vs composing
- Detection method (1 = timeline-based)
- Max character position reached

**Activity Metrics** (6 features per modality):
- Active engagement ratios during each phase
- Offscreen time, lookback behavior, thinking time

**Phase Comparisons** (10 features per modality):
- Reviewing/composing duration and activity ratios
- Absolute time in different states

**Cross-Modality** (4 features):
- Gaze vs mouse behavior correlation

---

## How Pairwise is Handled

For **pairwise** tasks, users compare two LLM responses (left vs right) simultaneously. We use **one global timeline** to detect phase boundaries consistently across both responses.

### Global Timeline Approach

**Boundary Detection**:
1. Merge left + right behavioral data into a single timeline
2. Apply timeline-based phase detection to the merged data
3. Use the latest reviewing end time across all queries as the global boundary
4. Split both left and right responses at the same boundary point

This ensures both responses share the same reviewing/composing phases, avoiding overlapping timelines.

### Per-Side Engagement Calculation

During the global reviewing phase, we calculate **actual engaged time** on each side separately:

- Track when user looks at left vs right response data
- Sum active intervals (< INACTIVITY_THRESHOLD) for each side
- Per-side engaged times sum to ≤ total reviewing duration
- Difference indicates relative preference between responses

### Pairwise Feature Structure

**Per-Side Reviewing** (7 features × 2 sides × 2 modalities = 28 features):
- `{mod}_{side}_reviewing_engaged_time_s`: Actual seconds on this side
- `{mod}_{side}_reviewing_engaged_pct`: Engaged time / global total
- `{mod}_{side}_reviewing_active_ratio`: Engaged time / global reviewing
- `{mod}_{side}_reviewing_offscreen_ratio`: Offscreen proportion
- `{mod}_{side}_max_char_position_reached`: Furthest character read

**Global Composing** (3 features × 2 modalities = 6 features):
- `{mod}_composing_duration_s`: Total composing time (shared)
- `{mod}_composing_pct`: Composing / total
- `{mod}_detection_method`: Timeline-based detection (1)

**Per-Side Lookback** (2 features × 2 sides × 2 modalities = 8 features):
- `{mod}_{side}_composing_lookback_time_s`: Looked back at this side
- `{mod}_{side}_composing_lookback_ratio`: Lookback / composing duration

**Comparison** (9 features × 2 modalities = 18 features):
- `{mod}_comparison_reviewing_time_ratio`: Left / right engaged time
- `{mod}_comparison_reviewing_time_diff`: Left - right engaged time
- `{mod}_comparison_which_side_longer_reviewing`: 1 (left) or -1 (right)
- `{mod}_comparison_reviewing_activity_ratio`: Left / right activity
- `{mod}_comparison_reviewing_activity_diff`: Left - right activity
- `{mod}_comparison_which_side_more_active_reviewing`: 1 or -1
- `{mod}_comparison_composing_lookback_ratio`: Left / right lookback
- `{mod}_comparison_composing_lookback_diff`: Left - right lookback
- `{mod}_comparison_which_side_read_further`: Based on max char position

**Cross-Modality** (5 features):
- Gaze vs mouse correlation
- Preference agreement between modalities


---

## Usage

Run the main feature extraction pipeline:

```bash
python src/pipelines/extract_features.py
```

**Output**: `output/extracted_features.csv` containing all 426 original features + phase features in a single file.

Phase features are automatically extracted during normal feature extraction - no separate steps required.