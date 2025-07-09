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

## Next Steps

The next phase of this project will involve running the developed script on the complete dataset. Following the data cleaning, the processed attention metrics will be matched with their associated user queries to prepare for further analysis. This README will be updated upon completion of these steps.
