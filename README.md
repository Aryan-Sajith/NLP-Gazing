# Phase 1: CSV Data Formatting and Preprocessing

This document outlines the initial data processing steps taken to clean and standardize raw CSV data. The primary tool for this process is a Python script designed to handle common CSV formatting issues.

---

### 1. Script Development for Data Cleaning

A Python script (`src/step-0-data-format.py`) was developed to automate the cleaning of raw data files.

**Key Features:**

* **Recursive File Search:** The script can recursively search through a specified directory to find all files matching a given name (e.g., `rel_gaze_one.csv`).
* **In-Place Formatting:** It processes each file and replaces the original with a corrected version, ensuring the data is standardized.
* **Error Handling:** The script includes error handling to prevent data loss and reports any issues encountered during processing.

### 2. Addressed Technical Challenges

The raw gaze data suffered from several formatting inconsistencies that prevented reliable parsing. The script was specifically designed to overcome these challenges.

#### Challenge 1: Inconsistent Column Count

* **Problem:** The third column, intended for text data, often contained commas. This caused standard CSV parsers to incorrectly split the text across multiple columns, resulting in rows with more than the expected six columns.
* **Solution:** The script identifies rows with more than six columns and intelligently concatenates the extra fields back into the third column, using a `, ` separator. This restores the intended structure of `[x_coord, y_coord, text, id_number, timestamp1, timestamp2]`.

#### Challenge 2: Improper Quote Escaping

* **Problem:** Text fields containing double quotes (`"`) were not properly escaped when using (`"`) to consistently surround all text. According to CSV standards, any double quote within a quoted field must be doubled (`""`) to be parsed correctly.
* **Solution:** A function was implemented to handle CSV quote escaping. It wraps the entire text field in double quotes and replaces any internal double quotes with two double quotes (`""`). This ensures that CSV parsers can correctly interpret the data without errors.

---

# Phase 2: Query Data Extraction

### 1. Script Development for Query Processing

A Python script (`src/step-1-extract-queries.py`) was developed to extract and structure query data from the raw CSV logs.

**Key Features:**
* **CSV to JSON Conversion:** Processes the `one_llm_query_logs_table.csv` file to extract user queries, LLM responses, and timestamps
* **Hierarchical Data Structure:** Organizes data by user ID and task ID for efficient lookup during gaze matching
* **Timestamp Parsing:** Converts query timestamps to Unix format for synchronization with gaze data

---

# Phase 3: Gaze Data and Query Matching

### 1. Script Development for Data Correlation

A sophisticated Python script (`src/step-2-match-gaze-queries.py`) was developed to correlate gaze tracking data with specific user queries based on timestamps and text content matching.

**Key Features:**
* **Recursive File Processing:** Automatically discovers and processes files in user/task directory structures
* **Dual-Condition Matching:** Uses both timestamp thresholds and text content verification for query transitions
* **Confirmation System:** Validates query transitions by checking subsequent gaze entries to prevent false positives
* **Flexible Pattern Matching:** Supports multiple file patterns and customizable confirmation counts

### 2. Addressed Technical Challenges

The gaze-to-query matching process presented several complex challenges that required iterative refinement of the matching algorithm.

#### Challenge 1: Premature Query Transitions

* **Problem:** Initial implementations switched queries immediately upon timestamp conditions, leading to incorrect associations when users briefly looked at content from upcoming queries.
* **Solution:** Implemented a confirmation system that requires a specified number of subsequent gaze entries to match the target query content before confirming a transition.

#### Challenge 2: Insufficient Text Matching

* **Problem:** Simple string matching failed to handle variations in text formatting, case sensitivity, and partial matches between gaze data and query content.
* **Solution:** Developed a robust text matching function that performs case-insensitive substring matching with minimum length requirements (4+ characters) to avoid false positives from short, common words.

#### Challenge 3: Placeholder Entry Handling

* **Problem:** Gaze data contains placeholder entries (-1, -1) that don't represent actual user attention, causing noise in the matching process.
* **Solution:** Modified the confirmation system to skip placeholder entries entirely, only considering valid gaze coordinates when confirming query transitions.

#### Challenge 4: End-of-File Boundary Conditions

* **Problem:** Confirmation checks near the end of gaze files couldn't find enough subsequent entries, causing valid transitions to be rejected.
* **Solution:** Implemented adaptive confirmation that accepts transitions based on available entries when fewer than the required confirmation count remain in the file.

---

## Next Steps
The next phase will involve analyzing the matched gaze-query patterns to determine if user gazing metrics can help predict llm preference. For now only singular llm tasks are being processed, I will also look to process and utilize multi-llm queries in this next phase.