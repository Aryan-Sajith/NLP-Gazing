# Phase-Based Feature Extraction

## Overview

When users interact with LLM outputs, they go through two distinct behavioral phases **within the same query**:

1. **Reviewing Phase**: User actively reads and engages with the LLM's response
2. **Composing Phase**: User stops reading, rates the response, and formulates their next question (all while still on the same query_id)

We detect this transition point and extract separate behavioral features for each phase, capturing how users engage with different responses.

---

## Phase Detection Method

We use a **hybrid method** combining two signals:

1. **Character Position Plateau**: When user reaches ~90% through the response
2. **Last Reading Activity**: The final moment user actively looked at text

**Algorithm**: Find when user reaches 90% → track last reading point → if user spent ≥2s after last reading, use that as boundary; otherwise use 90% plateau point.

**Fallback**: If user never reaches 90%, use maximum position actually reached as plateau.

**Validation** (303 queries, 24 users, 99 tasks):
- Median composing time: 25 seconds
- Mean composing time: 48 seconds
- Detects two user types: "quick composers" (8.5s) vs "deliberate composers" (68s)

---

## Extracted Features

We extract **52 new features for pointwise** and **~45 new features for pairwise** tasks (added to existing 426 features).

### Pointwise Features

**Phase Timing** (8 features per modality):
- Duration and percentage of reviewing vs composing
- Detection method and plateau timing
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

For **pairwise** tasks, users compare two LLM responses (left vs right) **simultaneously**. Key difference from pointwise: **ONE global timeline with per-side engagement tracking**.

### Global Timeline Approach

**Single Boundary Detection**:
1. Merge left + right data into one timeline
2. Find 90% plateau considering BOTH responses:
   - Both reach 90% → use LAST timestamp (finished reading both)
   - Only one reaches 90% → use that timestamp
   - Neither reaches 90% → use max position reached
3. Find last reading activity on merged data
4. Split both left and right at same boundary

**Example Timeline**:
```
0s ────────────────── 80s ──── 95s
    Reviewing (global)   Composing
    ↓                    ↓
    User switches        User rates &
    between left/right   composes
```

### Per-Side Engagement Calculation

During the global reviewing phase (0-80s), we calculate **actual engaged time** on each side using windowing:

**Method**: Sum of active intervals (like `focused_engagement` in original features)
- Track when user looks at left data vs right data
- Calculate engaged time by summing intervals < INACTIVITY_THRESHOLD
- These times can sum to ≤ global reviewing duration

**Example**:
- Global reviewing: 80s
- Left engaged time: 45s (56% of reviewing time)
- Right engaged time: 30s (38% of reviewing time)
- Remaining: 5s offscreen/switching

### Pairwise Feature Structure

**Per-Side Reviewing** (7 features × 2 sides × 2 modalities = 28 features):
- `{mod}_{side}_reviewing_engaged_time_s`: Actual seconds on this side
- `{mod}_{side}_reviewing_engaged_pct`: Engaged time / global total
- `{mod}_{side}_reviewing_active_ratio`: Engaged time / global reviewing
- `{mod}_{side}_reviewing_offscreen_ratio`: Offscreen proportion
- `{mod}_{side}_max_char_position_reached`: Furthest character read

**Global Composing** (4 features × 2 modalities = 8 features):
- `{mod}_composing_duration_s`: Total composing time (shared)
- `{mod}_composing_pct`: Composing / total
- `{mod}_detection_method`: Which method used
- `{mod}_plateau_time_pct`: When reached plateau

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

### Why This Approach?

**Problem with separate timelines**: Left (0-80s) and right (30-80s) would overlap and sum incorrectly.

**Solution**: Global phases + per-side engagement metrics properly represent reality:
- ONE reviewing phase where user switches between responses
- Engaged time on each side sums to ≤ total reviewing time
- Clear preference signal from relative engagement

---

## Usage

Run the main feature extraction pipeline:

```bash
python src/pipelines/extract_features.py
```

**Output**: `output/extracted_features.csv` containing all 426 original features + phase features in a single file.

Phase features are automatically extracted during normal feature extraction - no separate steps required.