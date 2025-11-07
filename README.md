# User Metrics and LLM Output Preference Alignment Research

## Overview

The main goal of this project is to determine if user usage metrics(like gazing, mouse movements, and such) can be utilized to better predict user preferences for LLM outputs.

## ⚡ Quick Start

```bash
# IMPORTANT: Skip step-0! It corrupts data.

# Step 1: Extract queries
python3 src/step-1-extract-queries.py

# Step 2: Match behavioral data to queries
python3 src/step-2-match-gaze-queries.py

# Step 3: Extract features
cd src/pairwise && python3 step-0-feature_eng_pipeline.py
```

## 📁 Required Files and Folder Structure

### Input Files Required:
1. **`full_query_logs_table.csv`** - Master CSV log file containing all user queries and LLM responses (place in project root)
2. **User behavioral data** - Raw `rel_*.csv` files (gaze and mouse tracking data)

### Folder Structure:
- **User data location**: `user_behavior/` - User behavioral data directory
- Each user has their own subdirectory (e.g., `user_behavior/A1IZ4NX41GKU4X/`)
- Within each user directory, task-specific CSV files contain the behavioral data

### ⚠️ Important Notes:
- **Do NOT run `src/step-0-data-format.py`** - it corrupts data
- Zeros in features are legitimate user behavior, not bugs

## Pre-Experiment: Text-Only Baseline

As a first step, we evaluated whether user preferences for LLM outputs could be predicted using only the text of the query and response.

- **Model**: 
Fine-tuned distilbert-base-uncased for regression (predicting a 1–5 Likert rating).

- **Input**: 
Concatenated query and response text; no user behavior data used.

- **Evaluation**: 
5-fold cross-validation.

- **Results**:

| Metric        | Value (Mean ± Std) |
|---------------|------------------|
| MSE           | 2.44 ± 1.14      |
| MAE           | 1.45 ± 0.39      |
| Pearson R     | 0.61 ± 0.35      |
| Accuracy (±10)| 13% ± 12%     |
| R²            | -6.53            |

Takeaway: Text alone, especially with a small dataset, is insufficient to predict user preferences, motivating the use of behavioral metrics in the main experiment. Consider the extremely low and subrandom accuracy(around 13 - 25%) of the model, which highlights the need for additional data in terms of both quantity and quality.

## Main Experiment: Incorporating User Interaction Data

This project processes raw user interaction data (gaze and mouse movements) and aligns it with corresponding Large Language Model (LLM) query-response logs. The goal is to produce a clean, annotated dataset where each moment of user gaze is mapped to a specific query they were viewing, followed by behavioral feature extraction for preference prediction.

The processing pipeline consists of three main steps:

1.  **Query Extraction**: Parses a master log of all LLM queries and organizes them into a structured JSON file.
2.  **Gaze-Query Matching**: Annotates the cleaned interaction data with query IDs by matching the text users were looking at with the text from the query logs.
3.  **Feature Extraction**: Extracts behavioral features from the annotated data for pairwise preference prediction.

---

## Data Processing Pipeline

### Step 1: Extracting Query Logs

-   **Script**: `step-1-extract-queries.py`
-   **Input**: A master CSV log file containing all user queries and LLM responses (`full_query_logs_table.csv`). This helps structurally organize and efficiently access the associated queries and metadata associated with each user and task combination without having to re-read our original query logs table.
-   **Process**: This script reads the master log and extracts all relevant fields for each query (`user_id`, `task_id`, `query_id`, `user_query`, `llm_response_1`, `llm_response_2`, and `query_timestamp`). It then organizes this information into a structured JSON file, grouped by user and task, and sorted by timestamp.
-   **Output**: `query_data.json`

### Step 2: Matching Gaze Data with Queries

-   **Script**: `step-2-match-gaze-queries.py`
-   **Input**:
    1.  The original `rel_*.csv` files from `user_behavior/`.
    2.  The `query_data.json` file from Step 1.
-   **Process**: This is the core matching script. It iterates through each row of the interaction data. Using the character index and a small window of surrounding text provided in the gaze data, it finds the corresponding LLM response text in `query_data.json`. Each row is then annotated with the matched `query_id`. For non-standard entries like when the user isn't looking at the screen or when they look at our experimentally provided prompt(instructing them how to perform their tasks), I used clearly defined query_ids like -1 and -2 which don't occur in the true dataset and additional boolean flags to properly convey this binary information for better model training.
-   **Output**: The script generates new annotated CSV files with the suffix `-query_id_assigned.csv`. These files are placed in the same directory as the input files and contain the original data plus additional columns for analysis.

### Step 3: Feature Extraction

-   **Script**: `src/pairwise/step-0-feature_eng_pipeline.py`
-   **Input**: The annotated `*-query_id_assigned.csv` files from Step 2
-   **Process**: Extracts 426 behavioral features per pairwise comparison from user gaze and mouse tracking data
-   **Output**: `extracted_features.csv` - A consolidated CSV file with one row per pairwise comparison

---

## Final Output Schema

The final `-query_id_assigned.csv` files contain the following columns:

| Column Name            | Description                                                                                              | Data Type |
| ---------------------- | -------------------------------------------------------------------------------------------------------- | --------- |
| `x`                    | The x-coordinate of the gaze/mouse.                                                                      | float     |
| `y`                    | The y-coordinate of the gaze/mouse.                                                                      | float     |
| `window`               | A small snippet of text the user was looking at.                                                         | string    |
| `centre_idx`           | The character index at the center of the user's gaze within the full text.                               | integer   |
| `rel_ts`               | Relative timestamp.                                                                                      | integer   |
| `abs_ts`               | Absolute timestamp.                                                                                      | integer   |
| `query_id`             | The ID of the query the user was viewing.                                                                | integer   |
| `is_experimental_text` | A boolean flag that is `true` if the user was looking at the static instructional prompt.                | boolean   |
| `is_not_looking`       | A boolean flag that is `true` if the user's gaze was off-screen (typically at coordinates -1, -1).         | boolean   |
| `response_gaze_percentage` | Percentage of entries per query per user spent gazing at the response                                | float    |
---

## Pairwise Feature Engineering Pipeline

After completing the core data processing pipeline, the project includes a specialized **pairwise feature engineering pipeline** located in `src/pairwise/` for predicting user preferences between competing LLM responses.

### Overview

-   **Scripts**: `step-0-feature_eng_pipeline.py`, `step-1-analyze_results.py`
-   **Objective**: Extract behavioral features from user gaze and mouse tracking data to predict which of two LLM responses a user prefers
-   **Output**: A single consolidated CSV file (`extracted_features.csv`) with one row per pairwise comparison

### Feature Categories

The pipeline extracts **426 total features** per pairwise comparison:

#### Core Behavioral Features (8 features per response = 16 total)
- **Active Engagement Ratio**: Proportion of time user actively engaged with response
- **Normalized Average Character Position**: Mean reading position relative to response length
- **Reading Completion Ratio**: How far through the response the user read
- **Normalized Character Position Variance**: Variability in reading positions

*Extracted separately for gaze and mouse modalities for both Response A and Response B*

#### Temporal Windowing Features (400 features)
- **Gaze Windows**: 100 time-based segments tracking average normalized character position over time for each response
- **Mouse Windows**: 100 time-based segments tracking average normalized character position over time for each response

#### Metadata Features (10 features)
- Response lengths, data point counts, query/user/task identifiers

### Target Variables

## Pairwise Data
For now the pipeline predicts a single preference metric:
- **Binary Preference**: Which response the user preferred (0 = Response A, 1 = Response B)

In the future, the pipeline may also predict other preference metrics, particularly for point-wise comparisons.

---

## How to Run the Pipeline

```bash
# Step 1: Extract queries from the master log
python3 src/step-1-extract-queries.py

# Step 2: Match gaze data to queries and generate annotated files
python3 src/step-2-match-gaze-queries.py

# Step 3: Extract pairwise behavioral features
cd src/pairwise && python3 step-0-feature_eng_pipeline.py
```

### Output Files:
- `query_data.json` - Structured query data (from Step 1)
- `*-query_id_assigned.csv` - Annotated behavioral data files (from Step 2)
- `extracted_features.csv` - Pairwise features for preference prediction (from Step 3)

---
