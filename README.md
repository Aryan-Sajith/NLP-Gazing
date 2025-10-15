# User Metrics and LLM Output Preference Alignment Research

## Overview

The main goal of this project is to determine if user usage metrics(like gazing, mouse movements, and such) can be utilized to better predict user preferences for LLM outputs.

For the first phase of this project I developed a set of scripts to process raw user interaction data (gaze and mouse movements) and align it with corresponding Large Language Model (LLM) query-response logs. The goal is to produce a clean, annotated dataset where each moment of user gaze is mapped to a specific query they were viewing.

The user metrics and query data processing and fusion pipeline:

0.  **(Optional) Timezone Fixing**: Was used before to fix timezones across database entries to follow a consistent GMT-timezone as opposed to varying local timestamps.
1.  **Initial Data Formatting**: Cleans and corrects raw, malformed CSV interaction files.
2.  **Query Extraction**: Parses a master log of all LLM queries and organizes them into a structured JSON file.
3.  **Gaze-Query Matching**: Annotates the cleaned interaction data with query IDs by matching the text users were looking at with the text from the query logs.

---

## Data Processing Pipeline

### Step 0: (Optional) Timezone Fixes
- **Note:** This was utilized when database timestamps had varying local timezones and had to be fixed. Now, the database and raw user gazing data should have entries within a consistent GMT timezone, so this phase should be unecessary.
- **Scripts**: `fix-timezone/`
- **Input**: Raw database files obtained from our online data (e.g., `query_logs_table`) and stored in csv format in some directory (e.g., `to-fix-data/`). These files have timezone inconsistency issues.
- **Process**: The script uses the `user_timezones.json` file to unify all timestamp information into the GMT format for timezone consistency.
- **Usage**:
  ```bash
  python gmt-timezone-converter.py
  ```

### Step 1: Initial Data Formatting

-   **Script**: `step-0-data-format.py`
-   **Input**: Raw `rel_*.csv` files located in a specified directory (e.g., `to-fix-data/`). These files often have formatting errors where text containing commas has been split across multiple columns.
-   **Process**: The script recursively finds all relevant CSV files, corrects the column structure by rejoining text that was improperly split(because commas were not escaped), and properly escapes quotes(since I surround all text in quotes to begin with). The corrected files overwrite the originals in place.
-   **Usage**:
    ```bash
    python step-0-data-format.py <path_to_data_directory>
    ```

### Step 2: Extracting Query Logs

-   **Script**: `step-1-extract-queries.py`
-   **Input**: A master CSV log file containing all user queries and LLM responses (`full_query_logs_table.csv`). This helps structurally organize and efficiently access the associated queries and metadata associated with each user and task combination without having to re-read our original query logs table.
-   **Process**: This script reads the master log and extracts all relevant fields for each query (`user_id`, `task_id`, `query_id`, `user_query`, `llm_response_1`, `llm_response_2`, and `query_timestamp`). It then organizes this information into a structured JSON file, grouped by user and task, and sorted by timestamp.
-   **Output**: `query_data.json`

### Step 3: Matching Gaze Data with Queries

-   **Script**: `step-2-match-gaze-queries.py`
-   **Input**:
    1.  The formatted `rel_*.csv` files from Step 0.
    2.  The `query_data.json` file from Step 1.
-   **Process**: This is the core matching script. It iterates through each row of the interaction data. Using the character index and a small window of surrounding text provided in the gaze data, it finds the corresponding LLM response text in `query_data.json`. Each row is then annotated with the matched `query_id`. For non-standard entries like when the user isn't looking at the screen or when they look at our experimentally provided prompt(instructing them how to perform their tasks), I used clearly defined query_ids like -1 and -2 which don't occur in the true dataset and additional boolean flags to properly convey this binary information for better model training.
-   **Output**: The script generates new annotated CSV files with the suffix `-query_id_assigned.csv`. These files are placed in the same directory as the input files and contain the original data plus additional columns for analysis.

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

## How to Run the Pipeline:
0. Note: Timezone is now consistent GMT across all timestamps so we skip step 0: timezone fixes.
1.  Place all raw user data (e.g., `P1/Task1/rel_gaze.csv`) into a main data directory (e.g., `to-fix-data/`).
2.  Place the master query log (`full_query_logs_table.csv`) in the project's root directory.
3.  Execute the scripts in order:

    ```bash
    # Step 0: Fix the raw CSV files
    python step-0-data-format.py to-fix-data/

    # Step 1: Generate the query data JSON file
    python step-1-extract-queries.py

    # Step 2: Match gaze data to queries and generate annotated files
    python step-2-match-gaze-queries.py

    # Step 3: Extract pairwise behavioral features
    python src/pairwise/step-0-feature_eng_pipeline.py
    ```

4.  The final, annotated data will be available as `*-query_id_assigned.csv` files within their original subdirectories.
5.  Pairwise features for preference prediction will be in `extracted_features.csv`
