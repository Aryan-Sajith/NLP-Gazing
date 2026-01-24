# User Metrics and LLM Output Preference Alignment Research

## Overview

The main goal of this project is to determine if user usage metrics(like gazing, mouse movements, and such) can be utilized to better predict user preferences for LLM outputs.

## ⚡ Quick Start

```bash
# Step 1: Extract queries from master log
python src/pipelines/extract_queries.py

# Step 2: Match behavioral data to queries
python src/pipelines/match_gaze.py

# Step 3: Extract behavioral features
python src/pipelines/extract_features.py

# Optional: Analyze worker quality
python src/pipelines/analyze_quality.py
```

## 📁 Required Files and Folder Structure

### Input Files Required:
1. **`full_query_logs_table.csv`** - Master CSV log file containing all user queries and LLM responses (place in project root)
2. **User behavioral data** - Raw `rel_*.csv` files (gaze and mouse tracking data) in `user_behavior/` directory

### Folder Structure:
```
project_root/
├── full_query_logs_table.csv    # Required: Master query logs
├── user_behavior/                # Required: Behavioral data directory
│   └── {user_id}/               # Each user has their own subdirectory
│       └── {task_id}/           # Task-specific directories
│           ├── rel_gaze_one.csv      # Pairwise gaze data (response 1)
│           ├── rel_gaze_two.csv      # Pairwise gaze data (response 2)
│           ├── rel_mouse_left.csv    # Pairwise mouse data (response 1)
│           ├── rel_mouse_right.csv   # Pairwise mouse data (response 2)
│           ├── rel_gaze.csv          # Pointwise gaze data
│           └── rel_mouse.csv         # Pointwise mouse data
└── output/                       # Generated: Pipeline outputs
    ├── query_data.json          # Step 1 output
    ├── extracted_features.csv   # Step 3 output
    ├── low_quality_workers.csv  # Quality analysis output
    └── worker_quality_report.txt
```

### ⚠️ Important Notes:
- The refactored codebase has replaced the old numbered scripts with modular pipelines
- All outputs now go to the `output/` directory
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

## Refactored Code Structure

The codebase has been completely refactored following software engineering best practices:

```
src/
├── config/              # Configuration and constants
│   └── constants.py
├── models/              # Data structures
│   ├── query.py
│   ├── behavioral_data.py
│   └── features.py
├── io/                  # File input/output operations
│   ├── query_reader.py
│   ├── query_writer.py
│   ├── behavioral_reader.py
│   ├── behavioral_writer.py
│   └── feature_writer.py
├── processing/          # Core business logic
│   ├── query_extractor.py
│   ├── gaze_matcher.py
│   └── feature_extractor.py
├── quality/             # Quality analysis
│   └── worker_analyzer.py
├── utils/               # Shared utilities
│   ├── timestamp.py
│   ├── text_matching.py
│   └── file_discovery.py
└── pipelines/           # Pipeline orchestration
    ├── extract_queries.py
    ├── match_gaze.py
    ├── extract_features.py
    └── analyze_quality.py
```

### Key Improvements:
- **Modular Design**: Clear separation of concerns with single-responsibility modules
- **Readable Code**: Descriptive names that explain intent (e.g., `QueryExtractor` vs "step-1")
- **Type Safety**: Data classes and type hints throughout
- **Maintainability**: Easy to understand, test, and extend
- **Configurability**: Centralized constants, no hard-coded paths

---

## Data Processing Pipeline

### Step 1: Extracting Query Logs

-   **Script**: `src/pipelines/extract_queries.py`
-   **Input**: Master CSV log file containing all user queries and LLM responses (`full_query_logs_table.csv`)
-   **Process**: Reads the master log and extracts all relevant fields for each query. Organizes this information into a structured JSON file, grouped by user and task, sorted by timestamp.
-   **Output**: `output/query_data.json`

### Step 2: Matching Gaze Data with Queries

-   **Script**: `src/pipelines/match_gaze.py`
-   **Input**:
    1.  The original `rel_*.csv` files from `user_behavior/`
    2.  The `query_data.json` file from Step 1
-   **Process**: Iterates through each row of the interaction data. Using the character index and surrounding text window, it finds the corresponding LLM response text. Each row is annotated with the matched `query_id`. Non-standard entries (not looking at screen, looking at experimental prompt) are assigned special query IDs (-2, -1).
-   **Output**: Annotated CSV files with suffix `_query_id_assigned.csv` in the same directories as input files

### Step 3: Feature Extraction

-   **Script**: `src/pipelines/extract_features.py`
-   **Input**: The annotated `*_query_id_assigned.csv` files from Step 2
-   **Process**: Extracts 426 behavioral features per pairwise comparison from user gaze and mouse tracking data
-   **Output**: `output/extracted_features.csv` - Consolidated CSV with one row per comparison

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

After completing the core data processing pipeline, the project includes a specialized **pairwise feature engineering pipeline** for predicting user preferences between competing LLM responses.

### Overview

-   **Script**: `src/pipelines/extract_features.py`
-   **Objective**: Extract behavioral features from user gaze and mouse tracking data to predict which of two LLM responses a user prefers
-   **Output**: A single consolidated CSV file (`output/extracted_features.csv`) with one row per pairwise comparison

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
python src/pipelines/extract_queries.py

# Step 2: Match gaze data to queries and generate annotated files
python src/pipelines/match_gaze.py

# Step 3: Extract pairwise and pointwise behavioral features
python src/pipelines/extract_features.py

# Optional: Analyze worker quality
python src/pipelines/analyze_quality.py
```

### Output Files:
- `output/query_data.json` - Structured query data (from Step 1)
- `user_behavior/*/*_query_id_assigned.csv` - Annotated behavioral data files (from Step 2)
- `output/extracted_features.csv` - Features for preference prediction (from Step 3)
- `output/low_quality_workers.csv` - Worker quality report (optional)
- `output/worker_quality_report.txt` - Detailed quality analysis (optional)

---
