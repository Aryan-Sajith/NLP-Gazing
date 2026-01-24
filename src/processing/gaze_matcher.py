"""Match gaze data to query responses"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from src.config.constants import (
    EXPERIMENTAL_PROMPT, QUERY_ID_NOT_LOOKING, QUERY_ID_PROMPT, 
    QUERY_ID_DEFAULT, PAIRWISE_FILES, POINTWISE_FILES
)
from src.io.behavioral_reader import BehavioralReader
from src.io.behavioral_writer import BehavioralWriter
from src.utils.text_matching import match_window_to_text


class GazeMatcher:
    """Matches gaze/mouse data to query responses"""
    
    def __init__(self, data_dir: Path, query_json_path: Path):
        self.data_dir = data_dir
        self.query_json_path = query_json_path
        self.reader = BehavioralReader()
        self.writer = BehavioralWriter()
        self.query_data = self._load_query_data()
    
    def _load_query_data(self) -> dict:
        """Load query data from JSON"""
        try:
            with open(self.query_json_path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            print(f"Error loading query data: {e}")
            return {}
    
    def match_all(self):
        """Process all behavioral CSV files and annotate with query IDs"""
        processed_count = 0
        
        for src_file in self.data_dir.rglob("rel_*.csv"):
            # Skip already processed files
            if "_query_id_assigned" in src_file.name:
                continue
            
            # Check if this is a file we should process
            if src_file.name not in (PAIRWISE_FILES + POINTWISE_FILES):
                continue
            
            # Extract user_id and task_id from path
            try:
                parts = src_file.parts
                user_id = parts[-3]
                task_id = parts[-2]
            except (ValueError, IndexError):
                continue
            
            # Get task queries
            task_block = self.query_data.get(user_id, {}).get(task_id, [])
            if not task_block:
                continue
            
            # Process this file
            if self._process_file(src_file, user_id, task_id, task_block):
                processed_count += 1
        
        print(f"\nFinished: {processed_count} files annotated with query IDs")
    
    def _process_file(self, src_file: Path, user_id: str, task_id: str, 
                     task_block: List[dict]) -> bool:
        """Process a single behavioral CSV file"""
        try:
            print(f"Processing {user_id}/{task_id}/{src_file.name}...")
            
            # Determine response mapping
            responses = self._get_response_mapping(src_file, task_block)
            
            if not responses:
                print(f"  No responses found for {src_file.name}")
                return False
            
            # Read raw CSV
            raw_rows = self.reader.read_raw_csv(src_file)
            if not raw_rows:
                return False
            
            # Annotate rows
            annotated_rows = self._annotate_rows(raw_rows, responses)
            
            # Write output
            output_file = src_file.with_name(f"{src_file.stem}_query_id_assigned.csv")
            self.writer.write_annotated_csv(output_file, annotated_rows)
            
            print(f"  ✓ Annotated {len(annotated_rows)-1} rows")
            return True
            
        except Exception as e:
            print(f"  ✗ Error processing {src_file}: {e}")
            return False
    
    def _get_response_mapping(self, src_file: Path, 
                             task_block: List[dict]) -> List[Tuple[int, str]]:
        """Determine which responses to use based on file type"""
        is_pointwise = src_file.name in POINTWISE_FILES
        
        if is_pointwise:
            # Use only llm_response_1
            return [
                (q["query_id"], q.get("llm_response_1", ""))
                for q in task_block
                if q.get("llm_response_1")
            ]
        else:
            # For pairwise: pick response based on file name
            if src_file.stem.endswith("_two") or src_file.stem.endswith("_right"):
                resp_key = "llm_response_2"
            else:
                resp_key = "llm_response_1"
            
            return [
                (q["query_id"], q.get(resp_key, ""))
                for q in task_block
                if q.get(resp_key)
            ]
    
    def _annotate_rows(self, raw_rows: List[list], 
                      responses: List[Tuple[int, str]]) -> List[list]:
        """Annotate CSV rows with query IDs"""
        output_rows = [[
            "x", "y", "window", "centre_idx", "rel_ts", "abs_ts",
            "query_id", "is_experimental_text", "is_not_looking"
        ]]
        
        # Track last query user was reading
        last_query_id = QUERY_ID_NOT_LOOKING
        
        for row in raw_rows:
            # Parse row based on column count
            if len(row) == 7:
                x, y, window, idx_str, rel_ts, _, abs_ts = row
            elif len(row) == 6:
                x, y, window, idx_str, rel_ts, abs_ts = row
            else:
                continue  # Skip invalid rows
            
            # Parse numeric values
            try:
                x_f, y_f = float(x), float(y)
            except (ValueError, TypeError):
                continue
            
            try:
                idx_i = int(float(idx_str)) if idx_str.strip() else -1
            except (ValueError, TypeError):
                idx_i = -1
            
            window = window.strip()
            
            # Determine query ID
            is_not_looking = x_f == -1 and y_f == -1
            is_exp_text = False
            query_id = QUERY_ID_DEFAULT
            
            if is_not_looking:
                # Assign to last query user was reading
                query_id = last_query_id
            elif match_window_to_text(EXPERIMENTAL_PROMPT, window, idx_i):
                # Looking at prompt
                query_id = QUERY_ID_PROMPT
                is_exp_text = True
            else:
                # Try to match to a query response
                for qid, text in responses:
                    if match_window_to_text(text, window, idx_i):
                        query_id = qid
                        last_query_id = qid
                        break
            
            output_rows.append([
                x, y, window, idx_i, rel_ts, abs_ts,
                query_id, str(is_exp_text).lower(), str(is_not_looking).lower()
            ])
        
        return output_rows
